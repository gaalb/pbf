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
// In: position, velocity, cellCount, cellPrefixSum
// Out: scratch (new velocity)

#define ViscosityRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 7)), DescriptorTable(UAV(u7, numDescriptors = 2))"

#include "ComputeCb.hlsli"

#include "SphKernels.hlsli" // Poly6

#include "GridUtils.hlsli" // posToCell(), cellIndex(), gridDim()

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
    NeighborCells nCells = NeighborCellIndices(pi);
    for (uint c = 0; c < nCells.count; c++)
    {
        uint ci = nCells.indices[c];
        uint count = cellCount[ci];

        for (uint s = 0; s < count; s++)
        {
            uint j = cellPrefixSum[ci] + s;
            if (j == i)
                continue;

            float3 r = pi - position[j];
            
            float r2 = dot(r, r);
            
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
            xsphSum += (vj - vi) * Poly6(r, r2, h);
        }
    }

    scratch[i] = vi + viscosity * xsphSum;
}
