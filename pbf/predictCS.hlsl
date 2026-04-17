// Forward Euler prediction step: apply forces, correct velocity for collisions,
// then predict the new position.
//
// Merges applyForcesCS, collisionVelocityCS, and predictPositionCS into a single
// per-particle pass (none of the three reads any neighbor's data), eliminating
// two dispatches and two UAV barriers.
//
// Steps per particle i:
//   1. Apply external forces:  v += (gravity + externalForce) * dt
//   2. Velocity collision response (box walls + solid SDF):
//        - Zero the velocity component directed into the surface.
//        - Damp the remaining tangential velocity by the adhesion factor.
//   3. Predict position:  p* = position + v * dt
//
// The velocity correction in step 2 is baked into p* (step 3 reads the
// corrected v), so the constraint solver starts from a wall-safe prediction.
//
// In:  position, velocity
// Out: velocity, predictedPosition

#define PredictRootSig \
    "CBV(b0), " \
    "DescriptorTable(UAV(u0, numDescriptors = 7)), " \
    "DescriptorTable(SRV(t0, numDescriptors = 1)), " \
    "StaticSampler(s0, " \
        "filter = FILTER_MIN_MAG_MIP_LINEAR, " \
        "addressU = TEXTURE_ADDRESS_CLAMP, " \
        "addressV = TEXTURE_ADDRESS_CLAMP, " \
        "addressW = TEXTURE_ADDRESS_CLAMP)"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"
#include "SolidSdf.hlsli"

RWStructuredBuffer<float3> position          : register(u0);
RWStructuredBuffer<float3> velocity          : register(u1);
RWStructuredBuffer<float3> predictedPosition : register(u2);

[RootSignature(PredictRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    float3 pos = position[i];
    float3 v   = velocity[i];

    // --- Step 1: apply external forces ---
    float3 force = float3(0.0, -9.8, 0.0) + externalForce;

    if (fountainEnabled) {
        float3 extent = boxMax - boxMin;
        if (pos.x > boxMax.x - extent.x * 0.05 &&
            pos.z > boxMax.z - extent.z * 0.05 &&
            pos.y < boxMin.y + extent.y * 0.3)
            force.y += 250.0;
    }

    v += force * dt;

    // --- Step 2: velocity-level collision response ---

    // box walls
    if (pos.x <= boxMin.x) { v.x = max(v.x, 0.0f); v.y *= (1.0f - adhesion); v.z *= (1.0f - adhesion); }
    if (pos.x >= boxMax.x) { v.x = min(v.x, 0.0f); v.y *= (1.0f - adhesion); v.z *= (1.0f - adhesion); }
    if (pos.y <= boxMin.y) { v.y = max(v.y, 0.0f); v.x *= (1.0f - adhesion); v.z *= (1.0f - adhesion); }
    if (pos.y >= boxMax.y) { v.y = min(v.y, 0.0f); v.x *= (1.0f - adhesion); v.z *= (1.0f - adhesion); }
    if (pos.z <= boxMin.z) { v.z = max(v.z, 0.0f); v.x *= (1.0f - adhesion); v.y *= (1.0f - adhesion); }
    if (pos.z >= boxMax.z) { v.z = min(v.z, 0.0f); v.x *= (1.0f - adhesion); v.y *= (1.0f - adhesion); }

    // solid SDF
    float d = SampleSdf(pos);
    if (d < pushRadius) {
        float3 normal = normalize(SdfGradient(pos));
        float vn = dot(v, normal);
        if (vn < 0.0f) {
            v -= vn * normal;
            v *= (1.0f - adhesion);
        }
    }

    // --- Step 3: predict position ---
    velocity[i]          = v;
    predictedPosition[i] = pos + v * dt;
}
