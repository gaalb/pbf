// Clamps predictedPosition to the simulation box and pushes it outside any
// solid obstacle described by the SDF volume texture.
//
// Called every solver iteration (same cadence as the old collisionCS), but
// deliberately does NOT touch velocity. Velocity solid response is handled
// once per simulation step by collisionVelocityCS, which runs before
// predictPositionCS so that the correction actually survives into p*.
//
// For the solid: if the predicted position is inside (SDF < 0), the particle
// is displaced along the SDF gradient so it sits on the surface.
//
// In:  predictedPosition
// Out: predictedPosition

#define CollisionPositionRootSig \
    "CBV(b0), " \
    "DescriptorTable(UAV(u0, numDescriptors = 7)), " \
    "DescriptorTable(SRV(t0, numDescriptors = 1)), " \
    "StaticSampler(s0, " \
        "filter = FILTER_MIN_MAG_MIP_LINEAR, " \
        "addressU = TEXTURE_ADDRESS_CLAMP, " \
        "addressV = TEXTURE_ADDRESS_CLAMP, " \
        "addressW = TEXTURE_ADDRESS_CLAMP)"

#include "ComputeCb.hlsli"
#include "SolidSdf.hlsli"

RWStructuredBuffer<float3> predictedPosition : register(u2);

[RootSignature(CollisionPositionRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    float3 p = predictedPosition[i];

    // Solid push-out: move p along the outward SDF gradient until d >= pushRadius.
    float d = SampleSdf(p);
    if (d < pushRadius) {
        float3 grad = SdfGradient(p); // always points in the
        // p must be pushed in the direction of grad, if d was negative, then
        // the distance to be pushed is radius+|d|, else it's radius-|d|
        p += (pushRadius - d) * normalize(grad);
    }
    
    // clamp to bounding box
    p = clamp(p, boxMin, boxMax);

    predictedPosition[i] = p;
}
