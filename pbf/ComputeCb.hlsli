#ifndef COMPUTE_CB_HLSLI
#define COMPUTE_CB_HLSLI

cbuffer ComputeCb : register(b0)
{
    float dt; // offset  0 (4 bytes): simulation timestep in seconds
    uint numParticles; // offset  4 (4 bytes): total particle count
    float h; // offset  8 (4 bytes): SPH smoothing radius
    float rho0; // offset 12 (4 bytes): rest density
    float3 boxMin; // offset 16 (12 bytes): simulation box minimum corner (world space)
    float epsilon; // offset 28 (4 bytes): constraint force mixing relaxation
    float3 boxMax; // offset 32 (12 bytes): simulation box maximum corner (world space)
    float viscosity; // offset 44 (4 bytes): XSPH viscosity coefficient c
    float sCorrK; // offset 48 (4 bytes): artificial pressure k
    float sCorrDeltaQ; // offset 52 (4 bytes): artificial pressure deltaq
    float sCorrN; // offset 56 (4 bytes): artificial pressure n
    float vorticityEpsilon; // offset 60 (4 bytes): vorticity confinement strength coefficient
    float3 externalForce; // offset 64 (12 bytes): horizontal force from arrow keys (acceleration, m/s^2)
    uint fountainEnabled; // offset 76 (4 bytes): 1 = fountain jet active, 0 = off
    float adhesion; // offset 80 (4 bytes): tangential velocity damping on wall contact (0 = frictionless, 1 = full stop)
};

#endif
