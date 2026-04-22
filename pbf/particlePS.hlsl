#include "SharedConfig.hlsli"

struct GSOutput
{
    float4 position : SV_Position; // window space position
    float2 uv : TEXCOORD; // ranges from (-1,-1) to (1,1), center is (0,0)
    float3 centerWorld : CENTERW; // particle center in world space (same for all 4 verts)
    float3 right : RIGHT; // billboard right axis in world space
    float3 up : UP; // billboard up axis in world space
    float density : DENSITY; // SPH density estimate for this particle
    uint lod : LOD;          // per-particle LOD value
};

cbuffer PerFrameCb : register(b0)
{
    float4x4 viewProjMat; // world space -> clip space  (view * projection)
    float4x4 rayDirMat; // NDC -> world space direction  (inverse of view-rotation * projection)
    float4 cameraPos; // camera position in world space
    float4 lightDir; // light direction in world space (should be normalized)
    float4 particleParams; // x = rho0 (density coloring), w = particle display radius
    uint shadingMode; // 0=unicolor, 1=density, 2=lod
    uint minLOD;  // minimum LOD value (far particles)
    uint maxLOD; // maximum LOD value (close particles, = solverIterations)
    float _pad;
};

float4 main(GSOutput input) : SV_Target
{
    // check if this pixel is inside the circle inscribed in the quad
    // uv was interpolated between (-1,-1) and (1,1), so length > 1 means outside the circle
    float r2 = dot(input.uv, input.uv);
    if (r2 > 1.0)
        discard; // throw away pixels outside the sphere silhouette

    // compute the fake sphere normal in world space
    // the billboard defines a local coordinate frame (right, up, forward)
    // uv.x and uv.y give us the position on the billboard face
    // nz is the "depth" component of the normal, derived from the sphere equation x^2 + y^2 + z^2 = 1
    float nz = sqrt(1.0 - r2);
    float3 forward = normalize(cameraPos.xyz - input.centerWorld);
    float3 normal = normalize(input.uv.x * input.right + input.uv.y * input.up + nz * forward);

    // base color: selected by shading mode
    float3 baseColor;

    if (shadingMode == SHADING_UNICOLOR)
    {
        // all particles the same color
        baseColor = float3(0.0, 1.0, 1.0);
    }
    else if (shadingMode == SHADING_DENSITY)
    {
        // Color by density: blue = sparse, green = rest density, red = compressed.
        // The PBF solver drives density close to rho0, so deviations are small —
        // we map the range [(1-range)*rho0, (1+range)*rho0] to the full color gradient
        // so that even minor density variations are clearly visible.
        float range = 0.15; // half-width of the visible density band around rho0
        float rho0 = particleParams.x;
        float t = saturate((input.density / rho0 - (1.0 - range)) / (2.0 * range));
        if (t < 0.5)
            baseColor = lerp(float3(0.0, 0.0, 1.0), float3(0.0, 1.0, 0.0), t * 2.0);       // blue (sparse) -> green (rest)
        else
            baseColor = lerp(float3(0.0, 1.0, 0.0), float3(1.0, 0.0, 0.0), (t - 0.5) * 2.0); // green (rest) -> red (compressed)
    }
    else // SHADING_LOD
    {
        // Color by LOD: blue = minLOD (far, few solver iterations), orange = maxLOD (close, many iterations).
        // Normalize the integer LOD value into [0,1] across the full [minLOD, maxLOD] range.
        float range = (float)(maxLOD - minLOD);
        float t = (range > 0.0) ? saturate((float(input.lod) - float(minLOD)) / range) : 0.0;
        baseColor = lerp(float3(0.2, 0.5, 1.0), float3(1.0, 0.5, 0.0), t); // blue -> orange
    }

    // Phong-Blinn lighting (same for all shading modes)

    // diffuse: scaled down so base color dominates over lighting variation
    float3 l = normalize(lightDir.xyz);
    float diffuse = max(dot(normal, l), 0.0) * 0.3;

    // specular: small highlight so particles still read as 3D
    float3 v = normalize(cameraPos.xyz - input.centerWorld);
    float3 h = normalize(l + v);
    float specular = pow(max(dot(normal, h), 0.0), 40.0) * 0.15;

    // high ambient so the base color is visible from all angles
    float ambient = 0.5;

    float3 finalColor = baseColor * (ambient + diffuse) + float3(1, 1, 1) * specular;

    return float4(finalColor, 1.0);
}