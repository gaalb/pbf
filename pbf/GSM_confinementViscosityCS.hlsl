  // GSM variant of confinementViscosityCS: preloads position, velocity, and omega
  // into group shared memory so that neighbor lookups within the same thread group
  // hit on-chip shared memory instead of going through the global memory hierarchy.
  //
  // The GSM check is: is neighbor j within [groupStart, groupStart + THREAD_GROUP_SIZE)?
  // If yes, read from GSM; otherwise fall back to the global UAV buffer.
  //
  // For neighbor fields that are NOT preloaded into GSM (cellCount, cellPrefixSum),
  // reads still go through global memory as before.

  #define ConfinementViscosityRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 7)), DescriptorTable(UAV(u7, numDescriptors = 2))"

  #include "SharedConfig.hlsli"
  #include "ComputeCb.hlsli"
  #include "SphKernels.hlsli"
  #include "GridUtils.hlsli"

RWStructuredBuffer<float3> position : register(u0);
RWStructuredBuffer<float3> velocity : register(u1);
RWStructuredBuffer<float3> omega : register(u5);
RWStructuredBuffer<float3> scratch : register(u6);
RWStructuredBuffer<uint> cellCount : register(u7);
RWStructuredBuffer<uint> cellPrefixSum : register(u8);

groupshared float3 gs_position[THREAD_GROUP_SIZE];
groupshared float3 gs_velocity[THREAD_GROUP_SIZE];
groupshared float3 gs_omega[THREAD_GROUP_SIZE];

[RootSignature(ConfinementViscosityRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
  void main(
      uint3 dispatchID : SV_DispatchThreadID,
      uint localIdx : SV_GroupIndex,
      uint3 groupID : SV_GroupID)
{
    uint i = dispatchID.x;
    uint groupStart = groupID.x * THREAD_GROUP_SIZE;

      // Preload per-particle data into GSM. Out-of-bounds threads write zero
      // so the sync doesn't stall on uninitialized data.
    gs_position[localIdx] = (i < numParticles) ? position[i] : (float3) 0;
    gs_velocity[localIdx] = (i < numParticles) ? velocity[i] : (float3) 0;
    gs_omega[localIdx] = (i < numParticles) ? omega[i] : (float3) 0;
    GroupMemoryBarrierWithGroupSync();

    if (i >= numParticles)
        return;

    float3 pi = gs_position[localIdx];
    float3 vi = gs_velocity[localIdx];
    float3 omegaI = gs_omega[localIdx];

    float3 eta = float3(0, 0, 0);
    float3 xsphSum = float3(0, 0, 0);

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

            bool inGSM = j >= groupStart && j < groupStart + THREAD_GROUP_SIZE;
            uint gj = j - groupStart;

            float3 pj = inGSM ? gs_position[gj] : position[j];
            float3 r = pi - pj;
            float r2 = dot(r, r);

              // Confinement: eta_i = sum_j |omega_j| * grad_W_spiky(r_ij)
            float3 omegaJ = inGSM ? gs_omega[gj] : omega[j];
            eta += length(omegaJ) * SpikyGrad(r, r2);

              // Viscosity: sum_j (v_j - v_i) * W_poly6(r_ij)
            float3 vj = inGSM ? gs_velocity[gj] : velocity[j];
            xsphSum += (vj - vi) * Poly6(r, r2);
        }
    }

      // Confinement
    float etaLen = length(eta);
    if (etaLen >= 1e-6)
    {
        float3 N = eta / etaLen;
        vi += dt * vorticityEpsilon * cross(N, omegaI);
    }

      // Viscosity + write
    scratch[i] = vi + viscosity * xsphSum;
}