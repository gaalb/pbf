struct GSOutput
{
    float4 position : SV_Position; // window space position
    float2 uv : TEXCOORD; // ranges from (-1,-1) to (1,1), center is (0,0)
    float3 centerWorld : CENTERW; // particle center in world space (same for all 4 verts)
    float3 right : RIGHT; // billboard right axis in world space
    float3 up : UP; // billboard up axis in world space
};

cbuffer PerFrameCb : register(b0)
{
    float4x4 viewProjMat; // world space -> clip space  (view * projection)
    float4x4 rayDirMat; // NDC -> world space direction  (inverse of view-rotation * projection)
    float4 cameraPos; // camera position in world space
    float4 lightDir; // light direction in world space (should be normalized)
    float4 particleParams; // xyz are color, w is radius
};

float4 main(GSOutput input) : SV_Target
{
    // check if this pixel is inside the circle inscribed in the quad
    // uv was interpolated between (-1,-1) and (1,1), so length > 1 means outside the circle
    float r2 = dot(input.uv, input.uv);
    if (r2 > 1.0)
        discard; // throw away pixels outside the sphere silhouette

    // compute the fake sphere normal in world space
    // che billboard defines a local coordinate frame (right, up, forward)
    // uv.x and uv.y give us the position on the billboard face
    // nz is the "depth" component of the normal, derived from the sphere equation x^2 + y^2 + z^2 = 1
    float nz = sqrt(1.0 - r2);
    float3 forward = normalize(cameraPos.xyz - input.centerWorld);
    float3 normal = normalize(input.uv.x * input.right + input.uv.y * input.up + nz * forward);

    // Shading is phong-blinn
    float3 baseColor = particleParams.xyz; // base color of the particle

    // diffuse: how much the surface faces the light
    float3 L = normalize(lightDir.xyz);
    float diffuse = max(dot(normal, L), 0.0);

    // specular: bright highlight where the surface reflects light toward the camera
    float3 V = normalize(cameraPos.xyz - input.centerWorld);
    float3 H = normalize(L + V); // halfway vector between light and view
    float specular = pow(max(dot(normal, H), 0.0), 64.0); // shininess: 64

    // ambient: a small constant so shadowed areas aren't fully black
    float ambient = 0.15;

    float3 finalColor = baseColor * (ambient + diffuse) + float3(1, 1, 1) * specular * 0.5;

    return float4(finalColor, 1.0);
}