// Fixed-point scale for the splat density pipeline.
// A Poly6 contribution of 1.0 density unit is stored as SPLAT_SCALE_U counts.
// Max single-particle Poly6(0) ≈ 6.41; max voxel density ≈ 420 (≈65 neighbors × Poly6(0)).
// 420 × 65536 ≈ 27.5 M << 2^32, so uint32 overflow is impossible in practice.
#define SPLAT_SCALE_F 65536.0f
#define SPLAT_SCALE_U 65536u
