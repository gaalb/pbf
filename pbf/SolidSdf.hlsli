#ifndef SOLID_SDF_HLSLI
#define SOLID_SDF_HLSLI

// Shared SDF sampling helpers for solid-obstacle collision compute shaders.
// Include this AFTER ComputeCb.hlsli (needs solidInvTransform, sdfMin, sdfMax).
// Declares sdf and sdfSampler at t0 / s0; the calling shader's root signature
// must expose DescriptorTable(SRV(t0)) and a matching StaticSampler(s0).

Texture3D<float> sdf : register(t0);
SamplerState sdfSampler : register(s0);

// Transform worldPos to object space, map to [0,1]^3 UV, and sample the SDF.
// Returns a large positive value when the point is outside the SDF bounding box
// so it is never mistaken for a penetration.
float SampleSdf(float3 worldPos)
{
    // SDF can be sampled in the object's local space, so transform
    // the world coordinate to object space coordinate
    float4 objPos = mul(solidInvTransform, float4(worldPos, 1.0f));
    // convert the local space into [0,1]^3 for UVW sampling
    float3 uvw    = (objPos.xyz - sdfMin) / (sdfMax - sdfMin); 
    if (any(uvw < 0.0f) || any(uvw > 1.0f)) return 1e4f; // outside SDF region: no solid here
    return sdf.SampleLevel(sdfSampler, uvw, 0); // sample the distance field: negative if inside object
}

// Central-difference gradient of the SDF in world space (unnormalized).
// The result points toward the surface inside the object, and away from it outside.
// Safe to call whenever SampleSdf returned a finite value (i.e. not the 1e4 sentinel),
// because eps (0.01) is much smaller than the 10%-padding between the mesh surface
// and the SDF bounding box, so none of the six offset samples will escape the box.
float3 SdfGradient(float3 worldPos)
{
    const float eps = 0.01f;
    return float3(
        SampleSdf(worldPos + float3(eps, 0, 0)) - SampleSdf(worldPos - float3(eps, 0, 0)),
        SampleSdf(worldPos + float3(0, eps, 0)) - SampleSdf(worldPos - float3(0, eps, 0)),
        SampleSdf(worldPos + float3(0, 0, eps)) - SampleSdf(worldPos - float3(0, 0, eps))
    ) * (0.5f / eps);
}

#endif
