#ifndef SOLID_SDF_HLSLI
#define SOLID_SDF_HLSLI

// Shared SDF sampling helpers for solid-obstacle collision compute shaders.
// Include this AFTER ComputeCb.hlsli (needs solidInvTransform, sdfMin, sdfMax).
// Declares sdf and sdfSampler at t0 / s0; the calling shader's root signature
// must expose DescriptorTable(SRV(t0)) and a matching StaticSampler(s0).

// Comment out to fall back to the 6-tap numeric gradient (useful for comparison).
#define USE_PRECOMPUTED_GRADIENTS

Texture3D<float4> sdf : register(t0); // R=distance, GBA=gradient xyz (object space)
SamplerState sdfSampler : register(s0);

// Returns the uniform object-to-world scale encoded in solidInvTransform.
// The linear part of the world-to-object matrix scales distances by 1/solidScale,
// so the magnitude of the transformed x-axis unit vector gives 1/solidScale.
float SolidScale()
{
    return 1.0f / length(mul((float3x3)solidInvTransform, float3(1.0f, 0.0f, 0.0f)));
}

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
    return sdf.SampleLevel(sdfSampler, uvw, 0).r; // .r = distance channel; negative if inside object
}

// Gradient of the SDF in world space (unnormalized result from numeric path; normalised from precomputed).
// The result points outward (away from the solid surface, toward increasing SDF values).
float3 SdfGradient(float3 worldPos)
{
#ifdef USE_PRECOMPUTED_GRADIENTS
    float4 objPos = mul(solidInvTransform, float4(worldPos, 1.0f));
    float3 uvw    = (objPos.xyz - sdfMin) / (sdfMax - sdfMin);
    float3 grad   = sdf.SampleLevel(sdfSampler, uvw, 0).yzw; // object-space outward gradient
    // Transform object-space direction to world space.
    // solidInvTransform uploaded row-major from C++ appears transposed in HLSL, so its
    // upper-left 3x3 equals the forward rotation R — same as Egg's mul(normal, modelMatInv).
    return normalize(mul(grad, (float3x3)solidInvTransform));
#else
    const float eps = 0.01f;
    return float3(
        SampleSdf(worldPos + float3(eps, 0, 0)) - SampleSdf(worldPos - float3(eps, 0, 0)),
        SampleSdf(worldPos + float3(0, eps, 0)) - SampleSdf(worldPos - float3(0, eps, 0)),
        SampleSdf(worldPos + float3(0, 0, eps)) - SampleSdf(worldPos - float3(0, 0, eps))
    ) * (0.5f / eps);
#endif
}

#endif
