#ifndef SOLID_SDF_HLSLI
#define SOLID_SDF_HLSLI

// Shared SDF sampling helpers for solid-obstacle collision compute shaders.
// Include this AFTER ComputeCb.hlsli (needs obstacles[i].invTransform/sdfMin/sdfMax).
// Declares sdf[MAX_OBSTACLES] at t0..t(MAX_OBSTACLES-1) and sdfSampler at s0.
// The calling shader's root signature must expose
//   DescriptorTable(SRV(t0, numDescriptors = MAX_OBSTACLES))   <- update numDescriptors when MAX_OBSTACLES changes
// and a matching StaticSampler(s0).

// Comment out to fall back to the 6-tap numeric gradient (useful for comparison).
#define USE_PRECOMPUTED_GRADIENTS

Texture3D<float4> sdf[MAX_OBSTACLES] : register(t0); // per-obstacle: R=distance, GBA=gradient xyz (object space)
SamplerState sdfSampler : register(s0);

// Returns the uniform object-to-world scale for obstacle i.
// The linear part of the world-to-object matrix scales distances by 1/solidScale,
// so the magnitude of the transformed x-axis unit vector gives 1/solidScale.
float SolidScale(int i)
{
    return 1.0f / length(mul((float3x3)obstacles[i].invTransform, float3(1.0f, 0.0f, 0.0f)));
}

// Transform worldPos to object space of obstacle i, map to [0,1]^3 UV, and sample the SDF.
// Returns a large positive value when the point is outside the SDF bounding box
// so it is never mistaken for a penetration.
float SampleSdf(int i, float3 worldPos)
{
    float4 objPos = mul(obstacles[i].invTransform, float4(worldPos, 1.0f));
    float3 uvw    = (objPos.xyz - obstacles[i].sdfMin) / (obstacles[i].sdfMax - obstacles[i].sdfMin);
    if (any(uvw < 0.0f) || any(uvw > 1.0f)) return 1e4f; // outside SDF region: no solid here
    float d = sdf[i].SampleLevel(sdfSampler, uvw, 0).r; // .r = distance channel; negative if inside object
    return d * SolidScale(i); // d was in model space units, convert to world space units
}

// Gradient of the SDF for obstacle i in world space: points outward (away from surface).
float3 SdfGradient(int i, float3 worldPos)
{
#ifdef USE_PRECOMPUTED_GRADIENTS
    float4 objPos = mul(obstacles[i].invTransform, float4(worldPos, 1.0f));
    float3 uvw    = (objPos.xyz - obstacles[i].sdfMin) / (obstacles[i].sdfMax - obstacles[i].sdfMin);
    float3 grad   = sdf[i].SampleLevel(sdfSampler, uvw, 0).yzw; // object-space outward gradient
    // Transform object-space direction to world space.
    // invTransform uploaded row-major from C++ appears transposed in HLSL, so its
    // upper-left 3x3 equals the forward rotation R — same as Egg's mul(normal, modelMatInv).
    return normalize(mul(grad, (float3x3)obstacles[i].invTransform));
#else
    const float eps = 0.01f;
    float3 grad = float3(
        SampleSdf(i, worldPos + float3(eps, 0, 0)) - SampleSdf(i, worldPos - float3(eps, 0, 0)),
        SampleSdf(i, worldPos + float3(0, eps, 0)) - SampleSdf(i, worldPos - float3(0, eps, 0)),
        SampleSdf(i, worldPos + float3(0, 0, eps)) - SampleSdf(i, worldPos - float3(0, 0, eps))
    ) * (0.5f / eps);
    return normalize(grad);
#endif
}

#endif
