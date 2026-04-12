#pragma once
// Contains the layout for the heaps PbfApp.h
// Two heaps to speak of:
// Main heap (SHADER_VISIBLE, HeapSlot::TOTAL descriptors)
// slot  0     CUBEMAP_SRV       t0 in bgPS        – skybox cubemap
// slot  1     PARTICLE_POS_SRV  t0 in particleVS  – position (overwritten each frame)
// slot  2     PARTICLE_DEN_SRV  t1 in particleVS  – density  (overwritten each frame)
// slot  3     PARTICLE_LOD_SRV  t2 in particleVS  – LOD      (overwritten each frame)
// slots 4–10  PARTICLE_FIELDS   u0..u6            – one UAV per ParticleField (PF_COUNT=7)
// slot 11     CELL_COUNT        u7                – per-cell particle count
// slot 12     CELL_PREFIX_SUM   u8                – exclusive prefix sum
// slots13–19  SORTED_FIELDS     u9..u15           – sorted particle field UAVs (PF_COUNT=7)
// slot 20     SDF_SRV           t0 in collision CSs – solid obstacle SDF Texture3D
// slot 21     PERM_UAV          u16               – permutation buffer
// slot 22     GROUP_SUM_UAV     (scratch)         – Blelloch prefix-sum group totals
// slot 23     LOD_UAV           (APBF)            – per-particle LOD countdown (uint)
// slot 24     LOD_REDUCTION_UAV (APBF)            – DTC min/max reduction scratch (2 uints)
//
// Snapshot staging heap (CPU-only, StagingSlot::TOTAL descriptors)
// slot  0     SNAPSHOT_POS_0    snapshotPosition[0] SRV
// slot  1     SNAPSHOT_POS_1    snapshotPosition[1] SRV
// slot  2     SNAPSHOT_DEN_0    snapshotDensity[0]  SRV
// slot  3     SNAPSHOT_DEN_1    snapshotDensity[1]  SRV
// slot  4     SNAPSHOT_LOD_0    snapshotLod[0]      SRV
// slot  5     SNAPSHOT_LOD_1    snapshotLod[1]      SRV


namespace HeapSlot {

    // Graphics SRVs (slots 0-3): overwritten each frame via CopyDescriptorsSimple
    constexpr UINT CUBEMAP_SRV        =  0;
    constexpr UINT PARTICLE_POS_SRV   =  1;
    constexpr UINT PARTICLE_DEN_SRV   =  2;
    constexpr UINT PARTICLE_LOD_SRV   =  3;

    // Particle field UAVs
    // Particle field f occupies slot (PARTICLE_FIELDS + f).  PF_COUNT consecutive slots.
    constexpr UINT PARTICLE_FIELDS    =  4;

    // Grid UAVs (u7..u8)
    // clearGridCS / countGridCS / lambdaCS / deltaCS etc. bind the table starting here.
    constexpr UINT CELL_COUNT         = 11;
    constexpr UINT CELL_PREFIX_SUM    = 12;

    // Sorted particle field UAVs
    // Sorted field f occupies slot (SORTED_FIELDS + f).  PF_COUNT consecutive slots.
    constexpr UINT SORTED_FIELDS      = 13;

    // Solid obstacle SDF SRV
    constexpr UINT SDF_SRV            = 20;

    // Permutation UAV (u16)
    constexpr UINT PERM_UAV           = 21;

    // Blelloch prefix-sum group-sum scratch UAV
    constexpr UINT GROUP_SUM_UAV      = 22;

    // Per-particle LOD countdown (uint per particle), written by lodCS
    constexpr UINT LOD_UAV            = 23;

    // LOD reduction buffer: 2 uints [minDTC bits, maxDTC bits], written by clearLodReductionCS + dtcReductionCS
    constexpr UINT LOD_REDUCTION_UAV  = 24;

    // Total size of the main shader-visible heap
    constexpr UINT TOTAL              = 25;

} // namespace HeapSlot

namespace StagingSlot {

    // Snapshot position SRVs indexed by double-buffer index (0 or 1)
    constexpr UINT SNAPSHOT_POS_0     =  0;
    constexpr UINT SNAPSHOT_POS_1     =  1;

    // Snapshot density SRVs indexed by double-buffer index (0 or 1)
    constexpr UINT SNAPSHOT_DEN_0     =  2;
    constexpr UINT SNAPSHOT_DEN_1     =  3;

    // Snapshot LOD SRVs indexed by double-buffer index (0 or 1)
    constexpr UINT SNAPSHOT_LOD_0     =  4;
    constexpr UINT SNAPSHOT_LOD_1     =  5;

    // Total size of the CPU-only snapshot staging heap
    constexpr UINT TOTAL              =  6;

} // namespace StagingSlot
