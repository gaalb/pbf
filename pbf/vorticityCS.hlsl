// For each particle i, estimates the local velocity field curl / vorticity via SPH.
//   omega_i = sum_j (v_j - v_i) x grad_W_spiky( p_i - p_j, h)
//
// Result is stored in omega, and consumed by confinementCS in the next pass.
// Per the paper's ordering, this pass uses updated velocity (from updateVelocityCS) but
// the OLD positions (updatePositionCS has not run yet).
//
// In: cellCount, cellPrefixSum, position, velocity
// Out: omega
#define VorticityRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 7)), DescriptorTable(UAV(u7, numDescriptors = 2))"

#include "SphKernels.hlsli" // SpikyGrad

#include "ComputeCb.hlsli"

#include "GridUtils.hlsli" // posToCell(), cellIndex(), gridDim()

RWStructuredBuffer<float3> position : register(u0);
RWStructuredBuffer<float3> velocity : register(u1);
RWStructuredBuffer<float3> omega : register(u5);
RWStructuredBuffer<uint> cellCount : register(u7);
RWStructuredBuffer<uint> cellPrefixSum : register(u8);

[RootSignature(VorticityRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    float3 pi = position[i]; // committed position from finalizeCS
    float3 vi = velocity[i]; // post-constraint velocity from finalizeCS

    float3 omegaAccum = float3(0, 0, 0); // accumulates the vorticity vector for particle i

    // The grid was built from predictedPositions, but this pass uses old positions.
    // The displacement between old and predicted is small relative to h, so the
    // grid is still a valid acceleration structure for finding neighbors here.
    int3 myCell = posToCell(pi);
    int dim = gridDim();
    for (int dz = -1; dz <= 1; dz++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++)
    {
        int3 nc = myCell + int3(dx, dy, dz);
        if (nc.x < 0 || nc.x >= dim ||
            nc.y < 0 || nc.y >= dim ||
            nc.z < 0 || nc.z >= dim)
            continue;

        uint ci = cellIndex(nc);
        uint count = cellCount[ci];

        for (uint s = 0; s < count; s++)
        {
            uint j = cellPrefixSum[ci] + s;
            if (j == i)
                continue;

            float3 r = pi - position[j];

            // The curl estimator uses the gradient of W with respect to p_j, not p_i.
            // grad_{p_j} W(p_i - p_j, h) = -grad_{p_i} W(p_i - p_j, h) = -SpikyGrad(r, h)
            // so the sign is negated compared to using SpikyGrad directly.
            omegaAccum += cross(velocity[j] - vi, -SpikyGrad(r, h));
        }
    }

    // store omega, confinementCS will read it from there
    omega[i] = omegaAccum;
}
