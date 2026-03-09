// SPH kernel functions shared by lambdaCS and deltaCS.
//
// All kernels are parameterized by the smoothing radius h.
// Particles farther than h apart do not interact — the kernels return 0.
// As per Muller 2013, we use two kernels:
//   Poly6       — scalar, used for density estimation
//   SpikyGrad   — vector, used for constraint gradient (position correction)
//TODO: double check the calculations here

#ifndef SPH_KERNELS_HLSLI
#define SPH_KERNELS_HLSLI

// Poly6 kernel
float Poly6(float3 r, float h)
{
    float r2 = dot(r, r);  // |r|^2 — cheaper than sqrt, and the formula uses r^2 directly
    float h2 = h * h;
    if (r2 > h2)
        return 0.0;
    float coeff = 315.0 / (64.0 * 3.14159265 * pow(h, 9.0));
    float diff = h2 - r2;
    return coeff * diff * diff * diff; 
}


// Spiky kernel gradient  (Desbrun & Cani 1996)
//
// The gradient of W_spiky with respect to p_i, where r = p_i - p_j:
//   grad_W(r, h) = -(45 / pi*h^6) * (h - |r|)^2 * (r / |r|)
//
// The negative sign means this points from i toward j -- in the direction
// that decreases |r|, which increases W (moving i closer to j increases density).
// This is the correct direction for use in the PBF position correction (Eq. 12):
//   delta_p_i = (1/rho0) * sum_j (lambda_i + lambda_j) * grad_W(r_ij, h)
// When lambda < 0 (too dense), the product is repulsive (pushes i away from j). Correct.
//
// r   -- vector from neighbor j toward particle i  (r = p_i - p_j)
// h   -- smoothing radius
float3 SpikyGrad(float3 r, float h)
{
    float rLen = length(r);

    // Guard: outside support radius contributes nothing.
    // rLen < 1e-6 handles j == i (r = 0) to avoid divide-by-zero.
    if (rLen > h || rLen < 1e-6)
        return float3(0.0, 0.0, 0.0);

    float coeff = 45.0 / (3.14159265 * pow(h, 6.0));
    float diff = h - rLen;
    float3 rHat = r / rLen; // unit vector from j toward i

    // Negative: gradient points from i toward j (toward the neighbor)
    return -coeff * diff * diff * rHat;
}

#endif // SPH_KERNELS_HLSLI
