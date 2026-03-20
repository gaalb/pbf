// element type for the particle structured buffer on the GPU
// layout must exactly match the Particle struct in ParticleTypes.h
// in a StructuredBuffer, float3 is packed tightly (12 bytes), unlike cbuffer registers
//
// Optimization note: scratch and omega are never live at the same time.
// scratch is written by deltaCS and consumed by positionFromScratchCS during the solver loop,
// then reused by viscosityCS and consumed by velocityFromScratchCS after confinement.
// omega is written by vorticityCS and consumed by confinementCS, which runs between those two uses.
// They could therefore share the same field to save 12 bytes per particle.
struct Particle {
    float3 position; // current (committed) position in world space
    float3 velocity; // current velocity, updated at end of each PBF step
    float3 predictedPosition; // predicted position used during constraint solving (p*)
    float lambda; // Lagrange multiplier computed in lambdaCS, read in deltaCS
    float density; // SPH density estimate written by lambdaCS, read by particle rendering shaders
    float3 omega; // vorticity vector written by vorticityCS, read by confinementCS
    float3 scratch; // scratch float3: used by deltaCS (position) and viscosityCS (velocity) to avoid Gauss-Seidel races
};
