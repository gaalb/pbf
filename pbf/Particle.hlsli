// element type for the particle structured buffer on the GPU
// layout must exactly match the Particle struct in ParticleTypes.h
struct Particle {
    float3 position; // current position in world space
    float3 velocity; // current velocity in world space
};
