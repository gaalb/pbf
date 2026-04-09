#pragma once
// Contains the layout for the heaps PbfApp.h
// Two heaps to speak of:
// Main heap (SHADER_VISIBLE, HeapSlot::TOTAL descriptors)
// slot  0     CUBEMAP_SRV       t0 in bgPS        – skybox cubemap                        
// slot  1     PARTICLE_POS_SRV  t0 in particleVS  – position (overwritten each frame)       
// slot  2     PARTICLE_DEN_SRV  t1 in particleVS  – density  (overwritten each frame)       
// slots 3–9   PARTICLE_FIELDS   u0..u6            – one UAV per ParticleField (PF_COUNT=7)  
// slot 10     CELL_COUNT        u7                – per-cell particle count                 
// slot 11     CELL_PREFIX_SUM   u8                – exclusive prefix sum                    
// slots12–18  SORTED_FIELDS     u9..u15           – sorted particle field UAVs (PF_COUNT=7) 
// slot 19     SDF_SRV           t0 in collision CSs – solid obstacle SDF Texture3D         
// slot 20     PERM_UAV          u16               – permutation buffer                      
// slot 21     GROUP_SUM_UAV     (scratch)         – Blelloch prefix-sum group totals         
//
// Snapshot staging heap (CPU-only, StagingSlot::TOTAL descriptors)
// slot  0     SNAPSHOT_POS_0    snapshotPosition[0] SRV                                     
// slot  1     SNAPSHOT_POS_1    snapshotPosition[1] SRV                                    
// slot  2     SNAPSHOT_DEN_0    snapshotDensity[0]  SRV                                     
// slot  3     SNAPSHOT_DEN_1    snapshotDensity[1]  SRV                                     


namespace HeapSlot {

    // Graphics SRVs
    constexpr UINT CUBEMAP_SRV        =  0;  
    constexpr UINT PARTICLE_POS_SRV   =  1; 
    constexpr UINT PARTICLE_DEN_SRV   =  2;  

    // Particle field UAVs
    // Particle field f occupies slot (PARTICLE_FIELDS + f).  PF_COUNT consecutive slots.
    constexpr UINT PARTICLE_FIELDS    =  3;

    // Grid UAVs (u7..u8)
    // clearGridCS / countGridCS / lambdaCS / deltaCS etc. bind the table starting here.
    constexpr UINT CELL_COUNT         = 10;  
    constexpr UINT CELL_PREFIX_SUM    = 11;  

    // Sorted particle field UAVs 
    // Sorted field f occupies slot (SORTED_FIELDS + f).  PF_COUNT consecutive slots.
    constexpr UINT SORTED_FIELDS      = 12;

    // Solid obstacle SDF SRV 
    constexpr UINT SDF_SRV            = 19;  

    // Permutation UAV (u16)
    constexpr UINT PERM_UAV           = 20;

    // Blelloch prefix-sum group-sum scratch UAV
    constexpr UINT GROUP_SUM_UAV      = 21;

    // Total size of the main shader-visible heap
    constexpr UINT TOTAL              = 22;

} // namespace HeapSlot

namespace StagingSlot {

    // Snapshot position SRVs indexed by double-buffer index (0 or 1)
    constexpr UINT SNAPSHOT_POS_0     =  0;  
    constexpr UINT SNAPSHOT_POS_1     =  1;  

    // Snapshot density SRVs indexed by double-buffer index (0 or 1)
    constexpr UINT SNAPSHOT_DEN_0     =  2;  
    constexpr UINT SNAPSHOT_DEN_1     =  3;  

    // Total size of the CPU-only snapshot staging heap
    constexpr UINT TOTAL              =  4;

} // namespace StagingSlot
