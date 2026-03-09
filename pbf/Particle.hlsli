// element type for the particle structured buffer on the GPU
// layout must exactly match the Particle struct in ParticleTypes.h
// in a StructuredBuffer, float3 is packed tightly (12 bytes), unlike cbuffer registers
struct Particle {
    float3 position;  // current (committed) position in world space
    float3 velocity;  // current velocity, updated at end of each PBF step
    float3 predictedPosition; // predicted position used during constraint solving (p*)
    float  lambda; // Newton step multiplier computed in lambdaCS, read in deltaCS
};
