#ifndef COMPUTE_CB_HLSLI
#define COMPUTE_CB_HLSLI

// Per-simulation-step data uploaded to all PBF compute shaders before each dispatch.
// Fields that are compile-time constants (h, rho0, gridMin/Max, kernel coefficients,
// push radius, sCorrDeltaQ/N) have been removed and are now #defines in SharedConfig.hlsli.
// Only runtime-variable fields remain here.

cbuffer ComputeCb : register(b0)
{
    float dt;               // offset  0: simulation timestep in seconds
    uint numParticles;      // offset  4: total particle count; bounds check in every shader
    float sCorrK;           // offset  8: artificial pressure magnitude coefficient
    float vorticityEpsilon; // offset 12: vorticity confinement strength
    float3 boxMin;          // offset 16: simulation box minimum corner (adjustable, world space)
    float epsilon;          // offset 28: constraint force mixing relaxation
    float3 boxMax;          // offset 32: simulation box maximum corner (adjustable, world space)
    float viscosity;        // offset 44: XSPH viscosity coefficient
    float3 externalForce;   // offset 48: horizontal acceleration from arrow keys (m/s^2)
    uint fountainEnabled;   // offset 60: 1 = fountain jet active, 0 = off
    float adhesion;         // offset 64: tangential velocity damping on wall contact
    float _pad0;            // offset 68
    float _pad1;            // offset 72
    float _pad2;            // offset 76
    float4x4 solidInvTransform; // offset  80: world-to-object transform for SDF sampling
    float3 sdfMin;          // offset 144: object-space SDF AABB minimum corner
    float _padSdfMin;       // offset 156
    float3 sdfMax;          // offset 160: object-space SDF AABB maximum corner
    float _padSdfMax;       // offset 172
    // total: 176 bytes
};

#endif
