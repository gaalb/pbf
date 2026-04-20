#pragma once
#include "Egg/Common.h"
#include "ComputeShader.h"
#include "GpuBuffer.h"
#include "DoubleBufferGpuBuffer.h"
#include "DescriptorAllocator.h"
#include "ParticleTypes.h"
#include "SharedConfig.hlsli"

// Encapsulates the spatial grid / sort subsystem: cellCount, cellPrefixSum, perm buffers,
// and all shaders for building a sorted spatial grid from particle positions.
//
// Allocates GPU resources and wires all descriptor tables into the main heap.
// Build()  dispatches the full grid-build + sort sequence on the given command list.
//
// The prefix sum is a uniform 5-pass Blelloch scan (requires GRID_DIM >= 64):
//
//   Pass 1 (N groups): local scan of cellCount -> cellPrefixSum   + groupSums
//   Pass 2 (M groups): local scan of groupSums -> groupPrefixSum  + superGroupSums
//   Pass 3 (1  group): global scan of superGroupSums  -> in-place
//   Pass 4 (M groups): propagate superGroupSums -> groupPrefixSum  (add offsets)
//   Pass 5 (N groups): propagate groupPrefixSum -> cellPrefixSum   (add offsets)
//
// N = numCells / ELEMENTS_PER_GROUP  (512 for 64^3, 4096 for 128^3)
// M = N / ELEMENTS_PER_GROUP         (1    for 64^3,    8  for 128^3)
//
// For GRID_DIM=64, M=1: passes 3 and 4 are trivially correct no-ops (add 0),
// but still dispatched — all 5 passes always run regardless of grid size.
// Passes 1 & 2 share prefixSumPass1CS.cso; passes 4 & 5 share prefixSumPass4CS.cso.
GG_CLASS(SpatialGrid)
    UINT numCells_ = 0;
    UINT numParticles_ = 0;
    UINT numPass1Groups_ = 0;   // N
    UINT numPass2Groups_ = 0;   // M

    GpuBuffer::P cellCountBuffer;
    GpuBuffer::P cellPrefixSumBuffer;
    GpuBuffer::P permBuffer;
    GpuBuffer::P groupSumBuffer;
    GpuBuffer::P groupPrefixSumBuffer;
    GpuBuffer::P superGroupSumBuffer;   // allocated with max(M, 2) elements (see pass 3 comment)

    ComputeShader::P clearGridShader;
    ComputeShader::P countGridShader;
    ComputeShader::P pass1Shader; // prefixSumPass1CS.cso — local scan of cellCount, N groups
    ComputeShader::P pass2Shader; // prefixSumPass1CS.cso — local scan of groupSums, M groups
    ComputeShader::P pass3Shader; // prefixSumPass3CS.cso — global scan of superGroupSums, 1 group
    ComputeShader::P pass4Shader; // prefixSumPass4CS.cso — propagate superGroupSums, M groups
    ComputeShader::P pass5Shader; // prefixSumPass4CS.cso — propagate groupPrefixSum, N groups
    ComputeShader::P sortShader;
    ComputeShader::P permutateShader;

public:
    // Allocate all GPU resources and wire descriptor tables.
    // particleFields must be a PF_COUNT-element array of the simulation's particle double-buffers.
    SpatialGrid(
        ID3D12Device* device,
        UINT numCells,
        UINT numParticles,
        DescriptorAllocator& mainAlloc,
        DescriptorAllocator& staticAlloc,
        D3D12_GPU_VIRTUAL_ADDRESS cbv,
        DoubleBufferGpuBuffer::P* particleFields)
    {
        using ResPtr = com_ptr<ID3D12Resource>*; // shorthand
        // One pass of a blelloch scan works on one group of cells, working out their
        // sum and exclusive prefix sum. The size of such a group is limited by the number
        // of threads we can launch in one thread group, which is 1024. This is why
		// we need multiple passes for large grids: each pass reduces the number of groups
        // by a factor of EPG.
        static constexpr UINT EPG = THREAD_GROUP_SIZE * 2;

        numCells_ = numCells;
        numParticles_ = numParticles;
        numPass1Groups_ = numCells / EPG; // N
        numPass2Groups_ = numPass1Groups_ / EPG; // M

		// lambda to copy a single descriptor from src (a static heap slot) to a main heap slot
        // [&] captures everything by reference, the lambda takes 2 parameters, slot and src, and copies
		// 1 descriptor from src to the main heap slot at index slot.
        auto copyToMainHeap = [&](UINT slot, D3D12_CPU_DESCRIPTOR_HANDLE src) {
            device->CopyDescriptorsSimple(1, mainAlloc.GetCpuHandle(slot), src,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        };

        // cell count buffer: how many particles can be found in each cell
        // it has numCell number of uints in it, starts in COMMON state, and is 
        // allocated on the default heap since it will be read/written by compute shaders,
        // this will be the case for most of the buffers below
        cellCountBuffer = GpuBuffer::Create( 
            device, numCells, sizeof(UINT),
            L"Cell Count Buffer", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
        cellCountBuffer->CreateUav(device, staticAlloc);

		// cell prefix sum buffer: the exclusive prefix sum of cellCountBuffer, 
        // used to scatter particles into the perm buffer. I.e. how many particles
        // can be found *before* the given cell (given an indexing).
		// starts in COMMON state, allocated on the default heap since it will be 
        // read/written by compute shaders
        cellPrefixSumBuffer = GpuBuffer::Create(
            device, numCells, sizeof(UINT),
            L"Cell Prefix Sum Buffer", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
        cellPrefixSumBuffer->CreateUav(device, staticAlloc);

		// group sum buffer: in pass 1, we break the grid into N groups of EPG cells, and 
        // each group sum is the total particle count of those EPG cells.
        groupSumBuffer = GpuBuffer::Create(
            device, numPass1Groups_, sizeof(UINT),
            L"Group Sum Buffer", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
        groupSumBuffer->CreateUav(device, staticAlloc);

        // group prefix sum buffer: the exclusive prefix sum of groupSums. Each entry tells you
        // how many particles exist across all groups *before* the given group. This is the
        // "global offset" that pass 5 adds into the local (intra-group) cell prefix sums,
        // turning them into the final cell prefix sums that span the entire grid.
        // Without this, each group's cellPrefixSum would only count particles within that
        // group — pass 5 uses groupPrefixSum to add in everything that came before.
        groupPrefixSumBuffer = GpuBuffer::Create(
            device, numPass1Groups_, sizeof(UINT),
            L"Group Prefix Sum Buffer", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
        groupPrefixSumBuffer->CreateUav(device, staticAlloc);

        // super group sum buffer: the total particle count of each super-group (a group of
        // EPG groups). After pass 3 scans them in-place, each entry instead holds how many
        // particles exist in all super-groups *before* this one — i.e., the global offset
        // that pass 4 propagates into groupPrefixSum.
        // Allocate max(M, 2) elements: pass 3 always scans at least 2 elements so that
        // PASS3_THREAD_COUNT >= 1 even when M=1 (GRID_DIM=64). The extra slot is zero-
        // initialized by D3D12 and never written by pass 2 or read by pass 4 in that case.
        UINT superGroupElems = std::max(numPass2Groups_, 2u);
        superGroupSumBuffer = GpuBuffer::Create(
            device, superGroupElems, sizeof(UINT),
            L"Super Group Sum Buffer", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
        superGroupSumBuffer->CreateUav(device, staticAlloc);

        // permutation buffer: maps each particle's current (unsorted) index to its sorted
        // destination index. sortCS fills it so that perm[i] = where particle i should end up
        // in the spatially-sorted order. permutateCS then reads perm[i] and copies every
        // particle field from position i to position perm[i], rearranging all fields into
        // cell-local order in one pass.
        permBuffer = GpuBuffer::Create(
            device, numParticles, sizeof(UINT),
            L"Permutation Buffer", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
        permBuffer->CreateUav(device, staticAlloc);

        UINT slot;
        // clearGridCS: UAV(u0) = 1 slot
		slot = mainAlloc.Allocate(1); // reserve 1 slot for the descriptor table used by this shader
        copyToMainHeap(slot, cellCountBuffer->GetUavCpuHandle()); // fill it with the appropraite descriptor
        clearGridShader = ComputeShader::Create(device, "Shaders/clearGridCS.cso", cbv,
			mainAlloc.GetGpuHandle(slot), // GPU handle to the start of this shader's contiguous descriptor region in the main heap
			std::vector<ResPtr>{ cellCountBuffer->GetResourcePtr() }, // input resources (for UAV barrier insertion)
			std::vector<ResPtr>{ cellCountBuffer->GetResourcePtr() }); // output resources (for UAV barrier insertion)

        // countGridCS: UAV(u0-1) = 2 slots
        // [0]=predictedPosition front, [1]=cellCount
        slot = mainAlloc.Allocate(2); // reserve 2 slots for the descriptor table used by this shader
        // particleFields is a is a PF_CONT length array of double buffered particle fields, passed in from PbfApp
        // because the spatial grid is responsible for building the grid from the predicted positions, and copying
        // the data to the sorted fields. The unsorted and sorted fields are the front and back targets
        // for this particle fields.
        // "registering" the countGridShader's 0th uav slot as the front target of particleFields[PF_PREDICTED_POSITION] means
        // that the uav which will actually be used to access the predicted positions will be the uav that provides
        // access to the front buffer of the particleFields[PF_PREDICTED_POSITION] double buffer
        // in the SpatialGrid class, we read from the front buffers of the double buffer, and based on the data read from
        // there, we construct the sorted version of that data in the backbuffer, where the other physics shaders can read it
        particleFields[PF_PREDICTED_POSITION]->registerFrontTarget(mainAlloc.GetCpuHandle(slot), false);
		copyToMainHeap(slot + 1, cellCountBuffer->GetUavCpuHandle()); // fill the second slot with the cell count buffer's UAV descriptor
        countGridShader = ComputeShader::Create(device, "Shaders/countGridCS.cso", cbv,
			mainAlloc.GetGpuHandle(slot), // GPU handle to the start of this shader's contiguous descriptor region in the main heap
            std::vector<ResPtr>{ particleFields[PF_PREDICTED_POSITION]->getFront()->GetResourcePtr(), 
                                    cellCountBuffer->GetResourcePtr() },
            std::vector<ResPtr>{ cellCountBuffer->GetResourcePtr() });
     
        // pass 1: prefixSumPass1CS — local scan of cellCount, N groups
        // UAV(u0-2) = 3 slots: [0]=cellCount, [1]=cellPrefixSum, [2]=groupSums      
        slot = mainAlloc.Allocate(3); // we use 3 uavs: need 3 slots in the main heap for descriptors
        copyToMainHeap(slot,     cellCountBuffer->GetUavCpuHandle()); // 0th slot: cell count
        copyToMainHeap(slot + 1, cellPrefixSumBuffer->GetUavCpuHandle()); // 1st slot: cell prefix sum
        copyToMainHeap(slot + 2, groupSumBuffer->GetUavCpuHandle()); // 2nd spot: group sum
        pass1Shader = ComputeShader::Create(device, "Shaders/prefixSumPass1_2CS.cso", cbv,
            mainAlloc.GetGpuHandle(slot), // GPU handle to the start of this shader's contiguous descriptor region in the main heap
            std::vector<ResPtr>{ cellCountBuffer->GetResourcePtr() }, // input: cellCount is read by pass 1
            std::vector<ResPtr>{ cellPrefixSumBuffer->GetResourcePtr(), // output: pass 1 writes the intra-group prefix sums
                                    groupSumBuffer->GetResourcePtr() }); // output: pass 1 writes each group's total sum

        // pass 2: prefixSumPass1CS reused — local scan of groupSums, M groups
        // UAV(u0-2) = 3 slots: [0]=groupSums, [1]=groupPrefixSum, [2]=superGroupSums
        slot = mainAlloc.Allocate(3); // reserve 3 slots for the descriptor table used by this shader
        copyToMainHeap(slot,     groupSumBuffer->GetUavCpuHandle()); // 0th slot: group sums (output of pass 1)
        copyToMainHeap(slot + 1, groupPrefixSumBuffer->GetUavCpuHandle()); // 1st slot: group prefix sum
        copyToMainHeap(slot + 2, superGroupSumBuffer->GetUavCpuHandle()); // 2nd slot: super group sums
        pass2Shader = ComputeShader::Create(device, "Shaders/prefixSumPass1_2CS.cso", cbv,
            mainAlloc.GetGpuHandle(slot), // GPU handle to the start of this shader's contiguous descriptor region in the main heap
            std::vector<ResPtr>{ groupSumBuffer->GetResourcePtr() }, // input: groupSums are read by pass 2
            std::vector<ResPtr>{ groupPrefixSumBuffer->GetResourcePtr(), // output: pass 2 writes the intra-super-group prefix sums
                                    superGroupSumBuffer->GetResourcePtr() }); // output: pass 2 writes each super-group's total sum

        // pass 3: prefixSumPass3CS — global scan of superGroupSums, 1 group
        // UAV(u0) = 1 slot: [0]=superGroupSums (in-place exclusive scan)
        slot = mainAlloc.Allocate(1); // reserve 1 slot for the descriptor table used by this shader
        copyToMainHeap(slot, superGroupSumBuffer->GetUavCpuHandle()); // the single UAV: super group sums, read and written in-place
        pass3Shader = ComputeShader::Create(device, "Shaders/prefixSumPass3CS.cso", cbv,
            mainAlloc.GetGpuHandle(slot), // GPU handle to the start of this shader's contiguous descriptor region in the main heap
            std::vector<ResPtr>{ superGroupSumBuffer->GetResourcePtr() }, // input: super group sums (totals per super-group)
            std::vector<ResPtr>{ superGroupSumBuffer->GetResourcePtr() }); // output: same buffer, now holding global offsets per super-group

        // pass 4: prefixSumPass4CS — propagate superGroupSums into groupPrefixSum, M groups
        // UAV(u0-1) = 2 slots: [0]=groupPrefixSum, [1]=superGroupSums
        slot = mainAlloc.Allocate(2); // reserve 2 slots for the descriptor table used by this shader
        copyToMainHeap(slot,     groupPrefixSumBuffer->GetUavCpuHandle()); // 0th slot: group prefix sums (will have offsets added)
        copyToMainHeap(slot + 1, superGroupSumBuffer->GetUavCpuHandle()); // 1st slot: super group sums (global offsets from pass 3)
        pass4Shader = ComputeShader::Create(device, "Shaders/prefixSumPass4_5CS.cso", cbv,
            mainAlloc.GetGpuHandle(slot), // GPU handle to the start of this shader's contiguous descriptor region in the main heap
            std::vector<ResPtr>{ superGroupSumBuffer->GetResourcePtr(), // input: super-group offsets are read
                                    groupPrefixSumBuffer->GetResourcePtr() }, // input: intra-super-group prefix sums are read
            std::vector<ResPtr>{ groupPrefixSumBuffer->GetResourcePtr() }); // output: now holds global group prefix sums

        // pass 5: prefixSumPass4CS reused — propagate groupPrefixSum into cellPrefixSum, N groups
        // UAV(u0-1) = 2 slots: [0]=cellPrefixSum, [1]=groupPrefixSum
        slot = mainAlloc.Allocate(2); // reserve 2 slots for the descriptor table used by this shader
        copyToMainHeap(slot,     cellPrefixSumBuffer->GetUavCpuHandle()); // 0th slot: cell prefix sums (will have offsets added)
        copyToMainHeap(slot + 1, groupPrefixSumBuffer->GetUavCpuHandle()); // 1st slot: global group offsets (from pass 4)
        pass5Shader = ComputeShader::Create(device, "Shaders/prefixSumPass4_5CS.cso", cbv,
            mainAlloc.GetGpuHandle(slot), // GPU handle to the start of this shader's contiguous descriptor region in the main heap
            std::vector<ResPtr>{ groupPrefixSumBuffer->GetResourcePtr(), // input: global group offsets are read
                                    cellPrefixSumBuffer->GetResourcePtr() }, // input: intra-group cell prefix sums are read
            std::vector<ResPtr>{ cellPrefixSumBuffer->GetResourcePtr() }); // output: now holds the final global cell prefix sums

        // sortCS: UAV(u0-3) = 4 slots
        // [0]=predictedPosition front, [1]=cellCount, [2]=cellPrefixSum, [3]=perm
        slot = mainAlloc.Allocate(4); // reserve 4 slots for the descriptor table used by this shader
        particleFields[PF_PREDICTED_POSITION]->registerFrontTarget(mainAlloc.GetCpuHandle(slot), false); // 0th slot: predicted positions (to determine each particle's cell)
        copyToMainHeap(slot + 1, cellCountBuffer->GetUavCpuHandle()); // 1st slot: cell count, reused as per-cell atomic counter for claiming slots
        copyToMainHeap(slot + 2, cellPrefixSumBuffer->GetUavCpuHandle()); // 2nd slot: cell prefix sums (base offset per cell)
        copyToMainHeap(slot + 3, permBuffer->GetUavCpuHandle()); // 3rd slot: permutation buffer (sortCS writes each particle's sorted destination here)
        sortShader = ComputeShader::Create(device, "Shaders/sortCS.cso", cbv,
            mainAlloc.GetGpuHandle(slot), // GPU handle to the start of this shader's contiguous descriptor region in the main heap
            std::vector<ResPtr>{ particleFields[PF_PREDICTED_POSITION]->getFront()->GetResourcePtr(), // input: predicted positions are read to compute cell indices
                                    cellPrefixSumBuffer->GetResourcePtr(), // input: cell prefix sums are read for base offsets
                                    cellCountBuffer->GetResourcePtr() }, // input: cell counts are read via InterlockedAdd
            std::vector<ResPtr>{ permBuffer->GetResourcePtr(), // output: permutation table is written (perm[i] = sorted destination of particle i)
                                    cellCountBuffer->GetResourcePtr() }); // output: cell counts are mutated as atomic counters (side effect)

        // permutateCS: UAV(u0-14) = 15 slots
        // [0-6]=pf front (unsorted, source), [7-13]=pf back (sorted, destination), [14]=perm
        slot = mainAlloc.Allocate(15); // reserve 15 slots: 7 particle fields as source + 7 as destination + 1 perm buffer
        for (UINT f = 0; f < PF_COUNT; f++)
            particleFields[f]->registerFrontTarget(mainAlloc.GetCpuHandle(slot + f),     false); // slots 0-6: source UAVs (unsorted particle data, read)
        for (UINT f = 0; f < PF_COUNT; f++)
            particleFields[f]->registerBackTarget( mainAlloc.GetCpuHandle(slot + 7 + f), false); // slots 7-13: destination UAVs (sorted particle data, written)
        copyToMainHeap(slot + 14, permBuffer->GetUavCpuHandle()); // slot 14: permutation table (perm[i] = where particle i goes)
        std::vector<ResPtr> permIn, permOut;
        for (UINT f = 0; f < PF_COUNT; f++) permIn.push_back(particleFields[f]->getFront()->GetResourcePtr()); // input: all 7 unsorted particle field buffers
        permIn.push_back(permBuffer->GetResourcePtr()); // input: permutation table
        for (UINT f = 0; f < PF_COUNT; f++) permOut.push_back(particleFields[f]->getBack()->GetResourcePtr()); // output: all 7 sorted particle field buffers
        permutateShader = ComputeShader::Create(device, "Shaders/permutateCS.cso", cbv,
            mainAlloc.GetGpuHandle(slot), // GPU handle to the start of this shader's contiguous descriptor region in the main heap
            std::move(permIn), std::move(permOut)); // input: unsorted fields + perm, output: sorted fields

    }

    // Dispatch the full grid-build and sort sequence.
    // All 5 prefix-sum passes are always dispatched; for small grids (M=1) passes 3 and 4
    // are trivially correct no-ops.
    void Build(ID3D12GraphicsCommandList* cmd)
    {
        UINT numGroups     = (numParticles_ + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
        UINT numCellGroups = (numCells_     + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
        UINT N = numPass1Groups_;
        UINT M = numPass2Groups_;

        clearGridShader->dispatch_then_barrier(cmd, numCellGroups);
        countGridShader->dispatch_then_barrier(cmd, numGroups);

        pass1Shader->dispatch_then_barrier(cmd, N);
        pass2Shader->dispatch_then_barrier(cmd, M);
        pass3Shader->dispatch_then_barrier(cmd, 1);
        pass4Shader->dispatch_then_barrier(cmd, M);
        pass5Shader->dispatch_then_barrier(cmd, N);

        // Reset cellCount to use as per-cell atomic counters for sort.
        clearGridShader->dispatch_then_barrier(cmd, numCellGroups);

        sortShader->dispatch_then_barrier(cmd, numGroups);
        permutateShader->dispatch_then_barrier(cmd, numGroups);
    }

    GpuBuffer::P GetCellCountBuffer()  const { return cellCountBuffer; }
    GpuBuffer::P GetPrefixSumBuffer()  const { return cellPrefixSumBuffer; }
GG_ENDCLASS
