// XSPH viscosity pass (Macklin & Muller 2013, Algorithm 1 line 22):
//
// Smooths the velocity field toward the neighborhood average:
// v_new_i = v_i + c * sum_{j != i} (v_j - v_i) * W_poly6(p_i - p_j, h)
//
// c controls how strongly: 0 means no viscosity, 1 means full averaging.
//
// To avoid a Gauss-Seidel race (thread i reads velocity[j] while thread j overwrites it),
// the corrected velocity is written to the scratch field. velocityFromScratchCS then copies
// scratch back to velocity.
//
// Per the paper's ordering, this pass uses the OLD positions (position has not been committed
// yet — updatePositionCS runs after this).
//
// Root signature:
//   CBV(b0)                        -- ComputeCb
//   DescriptorTable(UAV(u0..u6))   -- particle field buffers: u0 = position (read), u1 = velocity (read), u6 = scratch (write)
//   DescriptorTable(UAV(u7..u8))   -- grid buffers: u7 = cellCount, u8 = cellPrefixSum
//   DescriptorTable(UAV(u9..u15))  -- sorted particle field buffers (unused here)

#define ViscosityRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 7)), DescriptorTable(UAV(u7, numDescriptors = 2)), DescriptorTable(UAV(u9, numDescriptors = 7))"

#include "SphKernels.hlsli" // Poly6

cbuffer ComputeCb : register(b0)
{
    float dt; // offset 0 (4 bytes): simulation timestep in seconds
    uint numParticles; // offset 4 (4 bytes): total particle count
    float h; // offset 8 (4 bytes): SPH smoothing radius
    float rho0; // offset 12 (4 bytes): rest density
    float3 boxMin; // offset 16 (12 bytes): simulation box minimum corner (world space)
    float epsilon; // offset 28 (4 bytes): constraint force mixing relaxation
    float3 boxMax; // offset 32 (12 bytes): simulation box maximum corner (world space)
    float viscosity; // offset 44 (4 bytes): XSPH viscosity coefficient c
    float sCorrK; // offset 48 (4 bytes): artificial pressure k
    float sCorrDeltaQ; // offset 52 (4 bytes): artificial pressure deltaq
    float sCorrN; // offset 56 (4 bytes): artificial pressure n
    float vorticityEpsilon; // offset 60 (4 bytes): vorticity confinement strength coefficient
    float3 externalForce; // offset 64 (12 bytes): horizontal force from arrow keys (acceleration, m/s^2)
    uint fountainEnabled; // offset 76 (4 bytes): 1 = fountain jet active, 0 = off
};

#include "GridUtils.hlsli" // posToCell(), cellIndex(), gridDims()

RWStructuredBuffer<float3> position : register(u0);
RWStructuredBuffer<float3> velocity : register(u1);
RWStructuredBuffer<float3> scratch : register(u6);
RWStructuredBuffer<uint> cellCount : register(u7);
RWStructuredBuffer<uint> cellPrefixSum : register(u8);

[RootSignature(ViscosityRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    float3 pi = position[i]; // committed position after finalizeCS
    float3 vi = velocity[i]; // velocity after finalizeCS: (p* - p_old) / dt

    float3 xsphSum = float3(0, 0, 0);

    // The grid was built from predictedPositions, but this pass uses old positions.
    // See vorticityCS for the justification of this approximation.
    int3 myCell = posToCell(pi);
    int3 dims = gridDims();
    for (int dz = -1; dz <= 1; dz++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++)
    {
        int3 nc = myCell + int3(dx, dy, dz);
        if (nc.x < 0 || nc.x >= dims.x ||
            nc.y < 0 || nc.y >= dims.y ||
            nc.z < 0 || nc.z >= dims.z)
            continue;

        uint ci = cellIndex(nc);
        uint count = cellCount[ci];

        for (uint s = 0; s < count; s++)
        {
            uint j = cellPrefixSum[ci] + s;
            if (j == i)
                continue;

            float3 r = pi - position[j];
            float3 vj = velocity[j];

            // (v_j - v_i) * W: neighbor's velocity contribution weighted by proximity.
            // The full XSPH formula (Schechter & Bridson 2012) is:
            //   sum_j (m_j / rho_j) * (v_j - v_i) * W(r_ij, h)
            // m_j = 1 is dropped as a uniform constant; it only scales the sum and is
            // absorbed into the viscosity coefficient c.
            // 1/rho_j is also dropped. Unlike m_j, rho_j varies per particle, so omitting it
            // changes the relative weighting of neighbors and is not trivially justified.
            // The assumption is that PBF keeps rho_j ≈ rho0 for all j (incompressibility),
            // making 1/rho_j approximately uniform. Under that assumption it too is absorbed
            // into c, and the formula reduces to what we compute here.
            xsphSum += (vj - vi) * Poly6(r, h);
        }
    }

    scratch[i] = vi + viscosity * xsphSum;
}
