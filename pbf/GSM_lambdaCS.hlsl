// For each particle i, this shader:
// Estimates local density rho_i by summing Poly6 kernel contributions from all j:
// rho_i = sum_j(W_poly6(p_i - p_j, h))  (m = 1)
//
// Evaluates the density constraint: C_i = rho_i / rho0 - 1
//
// Computes lambda_i used by deltaCS for position corrections:
// lambda_i = -C_i / ( sum_k(|grad_pk(C_i)|^2) + eps )
//
// The denominator sums the squared gradient of C_i with respect to every particle k.
// two cases depending on whether k == i or k == j:
//   k == i: grad_pi(C_i) = (1/rho0) * sum_{j != i}( grad_W_spiky(r_ij, h) )
//   k == j: grad_pj(C_i) = -(1/rho0) * grad_W_spiky(r_ij, h)
// eps prevents division by zero when a particle has no neighbors.
//
// In: predictedPosition, cellCount, cellPreficSum
// Out: lambda, density
#define LambdaRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 7)), DescriptorTable(UAV(u7, numDescriptors = 2))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"
#include "SphKernels.hlsli" // Poly6, SpikyGrad
#include "GridUtils.hlsli" // posToCell(), cellIndex()

RWStructuredBuffer<float3> predictedPosition : register(u2);
RWStructuredBuffer<float> lambda : register(u3);
RWStructuredBuffer<float> density : register(u4);
RWStructuredBuffer<uint> cellCount : register(u7);
RWStructuredBuffer<uint> cellPrefixSum : register(u8);


// group shared memory array, one slot per thread in group
// This memory lives on-chip (fast), and is shared by all threads in the same
// group. Basically a scratch pad we control explicitly.
groupshared float3 gs_predPos[THREAD_GROUP_SIZE];

[RootSignature(LambdaRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(
uint3 dispatchID : SV_DispatchThreadID, // unique ID for the thread
uint localIdx : SV_GroupIndex, // unique ID for the thread within the group
uint3 groupID : SV_GroupID) // unique ID for the thread group
{
    uint i = dispatchID.x;
    uint groupStart = groupID.x * THREAD_GROUP_SIZE; // global particle index where this group beginx
    
    // Preload predictedPosition into GSM. Each thread (particle) copies its predictedPosition
    // into the group shared memory. i<numParticles means the thread belongs to no particle,
    // but we cannot return yet, because all threads must participate in the sync below.
    gs_predPos[localIdx] = (i < numParticles) ? predictedPosition[i] : (float3) 0;

    GroupMemoryBarrierWithGroupSync();
    
    if (i >= numParticles)
        return;

    // The lambda calculation involves three "indexes", i, j, and k, which aren't clearly explained
    // in the original research article:
    // i is the index of the particle we're computing lambda for (the "self" particle)
    // j is the index of a neighbor particle that contributes to i's density and constraint gradient
    // k is the index over a set of particles that includes the self particle and all the neighbors,
    // therefore it has two cases: k=i and k=j

    float3 pi = gs_predPos[localIdx]; // cache from GSM instead of UAV read

    float rho = 0.0; // density estimate rho_i
    float3 gradI = float3(0,0,0); // accumulates sum_{j != i}( grad_W(r_ij) ) for the k=i case
    float gradSqSum = 0.0; // accumulates sum_k(|grad_pk(C_i)|^2) for the k=j case

    // Iterate over neighboring cells using the precomputed cell index list.
    NeighborCells nCells = NeighborCellIndices(pi);
    for (uint c = 0; c < nCells.count; c++)
    {
        uint ci = nCells.indices[c];
        uint count = cellCount[ci];

        for (uint s = 0; s < count; s++)
        {
            uint j = cellPrefixSum[ci] + s;

            // r points from neighbor j toward particle i (r_ij = p_i - p_j)
            // for each neighbor j, check if it falls within this group's range
            // if yes, read from GSM (fast, on-clip), if not, fall back to the 
            // global UAV buffer (slow, off-clip through cache hierarchy
            bool inGSM = j >= groupStart && j < groupStart + THREAD_GROUP_SIZE;
            float3 pj = inGSM ? gs_predPos[j - groupStart] : predictedPosition[j];
            float3 r = pi - pj;
            
            float r2 = dot(r, r);

            // Density: every particle j including i itself contributes.
            rho += Poly6(r, r2);

            if (j != i)
            {
                // k=j case: grad_pj(C_i) = -(1/rho0) * grad_W_spiky(r_ij, h)
                float3 gradW = SpikyGrad(r, r2);
                float3 gradJ = -(1.0 / RHO0) * gradW;
                gradSqSum += dot(gradJ, gradJ); // add |grad_pj(C_i)|^2 to denominator

                // Also accumulate gradW into gradI -- needed for the k=i term after the loop
                gradI += gradW;
            }
        }
    }

    // k=i case
    // grad_pi(C_i) = (1/rho0) * sum_{j != i}(grad_W_spiky(r_ij, h))
    // gradI now holds the raw sum; apply the 1/rho0 factor and add its squared magnitude.
    gradI /= RHO0;
    gradSqSum += dot(gradI, gradI); // add |grad_pi(C_i)|^2 to denominator

    // Density constraint value
    float C = rho / RHO0 - 1.0; // 0 at rest density, > 0 if compressed, < 0 if sparse

    // Store density for visualization (read by rendering shaders)
    density[i] = rho;

    // Newton step length lambdda
    lambda[i] = -C / (gradSqSum + epsilon);
}
