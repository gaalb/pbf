// SPH kernel functions shared by lambdaCS and deltaCS.
//
// All kernels are parameterized by the smoothing radius h.
// Particles farther than h apart do not interact — the kernels return 0.
// As per Muller 2013, we use two kernels:
//   Poly6  — scalar, used for density estimation
//   SpikyGrad — vector, used for constraint gradient (position correction)
//TODO: double check the calculations here regarding gradients
// and their directions, in particular the "gradient with respect to smth"

#ifndef SPH_KERNELS_HLSLI
#define SPH_KERNELS_HLSLI

// Distance below which two particles are considered overlapping.
static const float EPSILON = 1e-6;

// When two particles overlap (r ~ 0), the spiky gradient is zero and they
// can never separate. This function returns a pseudo-random unit vector
// derived from the particle indices, giving overlapping particles a
// consistent but unique direction to push apart. Uses integer hashing
// instead of sin() to avoid GPU precision loss for large indices.
// The 127.5 offset guarantees no component is ever exactly zero
// (byte values are integers), so normalize() is always safe.
float3 overlapJitter(uint i, uint j)
{
    uint h = i * 0x1f1f1f1fu ^ j * 0x9e3779b9u;
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;

    return normalize(float3(
        float(h & 0xFF) - 127.5,
        float((h >> 8) & 0xFF) - 127.5,
        float((h >> 16) & 0xFF) - 127.5
    ));
}

// Poly6 kernel
float Poly6(float3 r, float h)
{
    float r2 = dot(r, r);
    float h2 = h * h;
    if (r2 > h2)
        return 0.0;
    float coeff = 315.0 / (64.0 * 3.14159265 * pow(h, 9.0));
    float diff = h2 - r2;
    return coeff * diff * diff * diff;
}


// Spiky kernel gradient
//
// The gradient was computed here:
// https://courses.grainger.illinois.edu/CS418/sp2023/text/sph.html
//
// with r = (pi - pj) as a vector
// grad(W_spiky(r, h)) = -45/=(pi*h^6)*(h-length(r))^2 * normalized(r)
float3 SpikyGrad(float3 r, float h)
{
    float rLen = length(r);

    // Guard: outside support radius contributes nothing.
    // rLen < EPSILON handles j == i (r = 0) to avoid divide-by-zero.
    if (rLen > h || rLen < EPSILON)
        return float3(0.0, 0.0, 0.0);

    float coeff = 45.0 / (3.14159265 * pow(h, 6.0));
    float diff = h - rLen;
    float3 rHat = r / rLen; // unit vector r

    // Negative: gradient points from i toward j (toward the neighbor)
    return -coeff * diff * diff * rHat;
}

#endif // SPH_KERNELS_HLSLI
