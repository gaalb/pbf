// Applies velocity-level collision response. Because updateVelocityCS later
// overwrites velocity with (p* - position) / dt, any velocity edits made
// after the position prediction (using the external forces) is
// lost. By running here (after applyForcesCS, before predictPositionCS) the
// corrected velocity is baked into p* and survives.
//
// Two effects are applied when a particle is headed into a wall or solid:
//
// Normal zeroing: zero the velocity component pointing into the surface,
// preventing the particle from accelerating further into it.
//
// Adhesion (tangential damping): reduce the remaining velocity slightly to
// mimic wall friction / surface tension.
//
// Box wall contact is detected from the current (committed) position.
// Solid SDF contact is detected when the SDF value at the position is < 0.
//
// In:  position, velocity  (force-applied by applyForcesCS)
// Out: velocity

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"
#include "SolidSdf.hlsli"

#define CollisionVelocityRootSig \
    "CBV(b0), " \
    "DescriptorTable(UAV(u0, numDescriptors = 2), SRV(t0, numDescriptors = " STR(NUM_OBSTACLES) ")), " \
    "StaticSampler(s0, " \
        "filter = FILTER_MIN_MAG_MIP_LINEAR, " \
        "addressU = TEXTURE_ADDRESS_CLAMP, " \
        "addressV = TEXTURE_ADDRESS_CLAMP, " \
        "addressW = TEXTURE_ADDRESS_CLAMP)"

RWStructuredBuffer<float3> position : register(u0);
RWStructuredBuffer<float3> velocity : register(u1);

[RootSignature(CollisionVelocityRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    float3 pos = position[i];
    float3 v   = velocity[i];

    // box wall response
    if (pos.x <= boxMin.x) { v.x = max(v.x, 0.0f); v.y *= (1.0f - adhesion); v.z *= (1.0f - adhesion); }
    if (pos.x >= boxMax.x) { v.x = min(v.x, 0.0f); v.y *= (1.0f - adhesion); v.z *= (1.0f - adhesion); }
    if (pos.y <= boxMin.y) { v.y = max(v.y, 0.0f); v.x *= (1.0f - adhesion); v.z *= (1.0f - adhesion); }
    if (pos.y >= boxMax.y) { v.y = min(v.y, 0.0f); v.x *= (1.0f - adhesion); v.z *= (1.0f - adhesion); }
    if (pos.z <= boxMin.z) { v.z = max(v.z, 0.0f); v.x *= (1.0f - adhesion); v.y *= (1.0f - adhesion); }
    if (pos.z >= boxMax.z) { v.z = min(v.z, 0.0f); v.x *= (1.0f - adhesion); v.y *= (1.0f - adhesion); }

    // solid SDF response — apply velocity correction for each obstacle independently
    // The committed position should be clear of the solid (corrected last frame by
    // collisionPredictedPositionCS). We check d < pushRadius to handle the first frame,
    // a newly moved solid, and numerical slip.
    for (int obstIdx = 0; obstIdx < NUM_OBSTACLES; obstIdx++) {
        float d = SampleSdf(obstIdx, pos);
        if (d < pushRadius) {
            float3 normal = normalize(SdfGradient(obstIdx, pos)); // outward surface normal in world space
            float vn = dot(v, normal);
            if (vn < 0.0f) { // particle heading into the solid
                // zero the inward component: normal points in the push direction,
                // vn is negative, so vn*normal points towards the middle of the object,
                // which means it must be substracted from v, which also points inward
                v -= vn * normal;
                v *= (1.0f - adhesion); // damp the remaining (tangential) velocity
            }
        }
    }

    velocity[i] = v;
}
