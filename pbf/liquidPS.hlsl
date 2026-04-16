// liquidPS.hlsl
// Pixel shader for the ray-marched liquid surface.
//
// Algorithm:
//   1. Ray-box intersection: find entry/exit t along the ray through boxMin..boxMax.
//   2. March from the entry point at step size H/2 (one voxel), sampling the
//      pre-computed density volume (Texture3D storing float4(rho, gradX, gradY, gradZ)).
//   3. Once density crosses the iso-surface threshold, binary-search (6 halvings)
//      to localize the surface position.
//   4. Sample the gradient at the surface point; the outward normal is -normalize(grad).
//   5. Blinn-Phong shading; output SV_Depth for correct occlusion against the solid.
//
// When the ray misses the box or never exceeds the threshold, the pixel is discarded.
// The liquid surface is drawn on the same depth buffer as the solid obstacle, so the
// solid correctly occludes (or is occluded by) the liquid.

#include "SharedConfig.hlsli"

#define LiquidRootSig \
    "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), " \
    "CBV(b0), " \
    "DescriptorTable(SRV(t0, numDescriptors = 1)), " \
    "StaticSampler(s0, " \
        "filter = FILTER_MIN_MAG_MIP_LINEAR, " \
        "addressU = TEXTURE_ADDRESS_CLAMP, " \
        "addressV = TEXTURE_ADDRESS_CLAMP, " \
        "addressW = TEXTURE_ADDRESS_CLAMP)"

cbuffer PerFrameCb : register(b0)
{
    float4x4 viewProjMat;     // world -> clip
    float4x4 rayDirMat;       // clip  -> world direction
    float4   cameraPos;       // world-space eye position
    float4   lightDir;        // xyz = direction toward light (normalized)
    float4   particleParams;  // x = rho0; must be here to match C++ layout
    uint     shadingMode;     // unused in this shader
    uint     minLOD;
    uint     maxLOD;
    float    _pad;
    float4   bbMin;           // xyz = adjustable boxMin, w = density iso-surface threshold
    float4   bbMax;           // xyz = adjustable boxMax, w = unused
};

// Density+gradient volume: float4(rho, gradX, gradY, gradZ) per voxel
Texture3D<float4> densityVol : register(t0);
SamplerState      samp       : register(s0);

struct VSOutput
{
    float4 position : SV_Position;
    float3 rayDir   : RAYDIR;
};

struct PSOutput
{
    float4 color : SV_Target;
    float  depth : SV_Depth;
};

// World-space position -> volume UVW in [0,1]^3 (maps GRID_MIN..GRID_MAX)
float3 worldToUvw(float3 pos)
{
    static const float3 gridMin = float3(-BOX_HALF_EXTENT, -BOX_HALF_EXTENT, -BOX_HALF_EXTENT);
    static const float3 gridMax = float3( BOX_HALF_EXTENT,  BOX_HALF_EXTENT,  BOX_HALF_EXTENT);
    return saturate((pos - gridMin) / (gridMax - gridMin));
}

[RootSignature(LiquidRootSig)]
PSOutput main(VSOutput input)
{
    PSOutput output;

    float3 origin = cameraPos.xyz;
    float3 dir    = normalize(input.rayDir);

    float3 boxMin = bbMin.xyz;
    float3 boxMax = bbMax.xyz;
    float  threshold = bbMin.w; // density iso-surface threshold

    // --- Ray-AABB intersection (slab method) ---
    float3 invDir = 1.0 / dir;
    float3 t0 = (boxMin - origin) * invDir;
    float3 t1 = (boxMax - origin) * invDir;
    float3 tNear = min(t0, t1);
    float3 tFar  = max(t0, t1);
    float tEnter = max(max(tNear.x, tNear.y), tNear.z);
    float tExit  = min(min(tFar.x,  tFar.y),  tFar.z);

    if (tEnter > tExit || tExit < 0.0)
        discard;

    tEnter = max(tEnter, 0.0); // start from camera if inside box

    // --- Ray march at one voxel per step ---
    // Voxel world-space size = (GRID_DIM * H) / VOL_DIM; adjusts automatically when VOL_DIM changes.
    float stepSize = (float(GRID_DIM) * H) / float(VOL_DIM);
    float tPrev    = tEnter;
    float tCur     = tEnter;
    bool  found    = false;

    // Maximum steps: enough to cross the full box diagonal at step size H/2
    float maxDist = tExit - tEnter;
    int   maxSteps = (int)(maxDist / stepSize) + 2;

    for (int i = 0; i < maxSteps; i++)
    {
        tCur = tEnter + float(i) * stepSize;
        if (tCur >= tExit)
            break;

        float3 pos = origin + dir * tCur;
        float  rho = densityVol.SampleLevel(samp, worldToUvw(pos), 0).x;

        if (rho >= threshold)
        {
            found = true;
            break;
        }
        tPrev = tCur;
    }

    if (!found)
        discard;

    // --- Binary search: 6 halvings to localize the surface ---
    float tLo = tPrev;
    float tHi = tCur;
    for (int b = 0; b < 6; b++)
    {
        float  tMid = (tLo + tHi) * 0.5;
        float3 pMid = origin + dir * tMid;
        float  rMid = densityVol.SampleLevel(samp, worldToUvw(pMid), 0).x;
        if (rMid < threshold)
            tLo = tMid;
        else
            tHi = tMid;
    }
    float tSurf = (tLo + tHi) * 0.5;
    float3 pSurf = origin + dir * tSurf;

    // --- Sample gradient at surface point for the normal ---
    float4 volSample = densityVol.SampleLevel(samp, worldToUvw(pSurf), 0);
    float3 gradRho   = volSample.yzw;

    float3 N;
    float gradLen = length(gradRho);
    if (gradLen < 1e-4)
    {
        // degenerate gradient (deep interior or empty); use view direction as fallback
        N = -dir;
    }
    else
    {
        N = -normalize(gradRho); // outward: points from liquid toward air
    }

    // --- Blinn-Phong shading ---
    float3 L = normalize(lightDir.xyz);
    float3 V = normalize(cameraPos.xyz - pSurf);
    float3 H_vec = normalize(L + V);

    static const float3 kLiquidColor = float3(0.05, 0.28, 0.75); // deep ocean blue
    static const float  kAmbient     = 0.18;
    static const float  kDiffuse     = 0.55;
    static const float  kSpecular    = 1.2;
    static const float  kShininess   = 80.0;

    float3 ambient  = kAmbient * kLiquidColor;
    float3 diffuse  = max(dot(N, L), 0.0) * kDiffuse * kLiquidColor;
    float  spec     = pow(max(dot(N, H_vec), 0.0), kShininess);
    float3 specular = kSpecular * spec * float3(1.0, 1.0, 1.0);

    output.color = float4(ambient + diffuse + specular, 1.0);

    // --- Depth: project surface point to NDC depth ---
    float4 clipPos = mul(viewProjMat, float4(pSurf, 1.0));
    output.depth   = clipPos.z / clipPos.w;

    return output;
}
