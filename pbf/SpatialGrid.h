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

		// lambda to copy a single descriptor from src (a static heap slot) to dst (a main heap slot)
        // [&] captures everything by reference, the lambda takes 2 parameters, slot and src, and copies
		// 1 descriptor from src to the main heap slot at index slot.
        auto copyToMainHeap = [&](UINT slot, D3D12_CPU_DESCRIPTOR_HANDLE src) {
            device->CopyDescriptorsSimple(1, mainAlloc.GetCpuHandle(slot), src,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        };

        cellCountBuffer = GpuBuffer::Create(
            device, numCells, sizeof(UINT),
            L"Cell Count Buffer", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
        cellCountBuffer->CreateUav(device, staticAlloc);

        cellPrefixSumBuffer = GpuBuffer::Create(
            device, numCells, sizeof(UINT),
            L"Cell Prefix Sum Buffer", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
        cellPrefixSumBuffer->CreateUav(device, staticAlloc);

        groupSumBuffer = GpuBuffer::Create(
            device, numPass1Groups_, sizeof(UINT),
            L"Group Sum Buffer", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
        groupSumBuffer->CreateUav(device, staticAlloc);

        groupPrefixSumBuffer = GpuBuffer::Create(
            device, numPass1Groups_, sizeof(UINT),
            L"Group Prefix Sum Buffer", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
        groupPrefixSumBuffer->CreateUav(device, staticAlloc);

        // Allocate max(M, 2) elements: pass 3 always scans at least 2 elements so that
        // PASS3_THREAD_COUNT >= 1 even when M=1 (GRID_DIM=64). The extra slot is zero-
        // initialized by D3D12 and never written by pass 2 or read by pass 4 in that case.
        UINT superGroupElems = std::max(numPass2Groups_, 2u);
        superGroupSumBuffer = GpuBuffer::Create(
            device, superGroupElems, sizeof(UINT),
            L"Super Group Sum Buffer", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
        superGroupSumBuffer->CreateUav(device, staticAlloc);

        permBuffer = GpuBuffer::Create(
            device, numParticles, sizeof(UINT),
            L"Permutation Buffer", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
        permBuffer->CreateUav(device, staticAlloc);

        UINT slot;
        // clearGridCS: UAV(u0) = 1 slot
        slot = mainAlloc.Allocate(1);
        copyToMainHeap(slot, cellCountBuffer->GetUavCpuHandle());
        clearGridShader = std::make_shared<ComputeShader>(device, "Shaders/clearGridCS.cso", cbv,
            mainAlloc.GetGpuHandle(slot),
            std::vector<ResPtr>{ cellCountBuffer->GetResourcePtr() },
            std::vector<ResPtr>{ cellCountBuffer->GetResourcePtr() });

        // countGridCS: UAV(u0-1) = 2 slots
        // [0]=predictedPosition front, [1]=cellCount
        slot = mainAlloc.Allocate(2);
        particleFields[PF_PREDICTED_POSITION]->registerFrontTarget(mainAlloc.GetCpuHandle(slot), false);
        copyToMainHeap(slot + 1, cellCountBuffer->GetUavCpuHandle());
        countGridShader = std::make_shared<ComputeShader>(device, "Shaders/countGridCS.cso", cbv,
            mainAlloc.GetGpuHandle(slot),
            std::vector<ResPtr>{ particleFields[PF_PREDICTED_POSITION]->getFront()->GetResourcePtr(),
                                    cellCountBuffer->GetResourcePtr() },
            std::vector<ResPtr>{ cellCountBuffer->GetResourcePtr() });
     
        // pass 1: prefixSumPass1CS — local scan of cellCount, N groups
        // UAV(u0-2) = 3 slots: [0]=cellCount, [1]=cellPrefixSum, [2]=groupSums      
        slot = mainAlloc.Allocate(3);
        copyToMainHeap(slot,     cellCountBuffer->GetUavCpuHandle());
        copyToMainHeap(slot + 1, cellPrefixSumBuffer->GetUavCpuHandle());
        copyToMainHeap(slot + 2, groupSumBuffer->GetUavCpuHandle());
        pass1Shader = std::make_shared<ComputeShader>(device, "Shaders/prefixSumPass1_2CS.cso", cbv,
            mainAlloc.GetGpuHandle(slot),
            std::vector<ResPtr>{ cellCountBuffer->GetResourcePtr() },
            std::vector<ResPtr>{ cellPrefixSumBuffer->GetResourcePtr(),
                                    groupSumBuffer->GetResourcePtr() });

        // pass 2: prefixSumPass1CS reused — local scan of groupSums, M groups
        // UAV(u0-2) = 3 slots: [0]=groupSums, [1]=groupPrefixSum, [2]=superGroupSums
        slot = mainAlloc.Allocate(3);
        copyToMainHeap(slot,     groupSumBuffer->GetUavCpuHandle());
        copyToMainHeap(slot + 1, groupPrefixSumBuffer->GetUavCpuHandle());
        copyToMainHeap(slot + 2, superGroupSumBuffer->GetUavCpuHandle());
        pass2Shader = std::make_shared<ComputeShader>(device, "Shaders/prefixSumPass1_2CS.cso", cbv,
            mainAlloc.GetGpuHandle(slot),
            std::vector<ResPtr>{ groupSumBuffer->GetResourcePtr() },
            std::vector<ResPtr>{ groupPrefixSumBuffer->GetResourcePtr(),
                                    superGroupSumBuffer->GetResourcePtr() });

        // pass 3: prefixSumPass3CS — global scan of superGroupSums, 1 group
        // UAV(u0) = 1 slot: [0]=superGroupSums
        slot = mainAlloc.Allocate(1);
        copyToMainHeap(slot, superGroupSumBuffer->GetUavCpuHandle());
        pass3Shader = std::make_shared<ComputeShader>(device, "Shaders/prefixSumPass3CS.cso", cbv,
            mainAlloc.GetGpuHandle(slot),
            std::vector<ResPtr>{ superGroupSumBuffer->GetResourcePtr() },
            std::vector<ResPtr>{ superGroupSumBuffer->GetResourcePtr() });

        // pass 4: prefixSumPass4CS — propagate superGroupSums into groupPrefixSum, M groups
        // UAV(u0-1) = 2 slots: [0]=groupPrefixSum, [1]=superGroupSums
        slot = mainAlloc.Allocate(2);
        copyToMainHeap(slot,     groupPrefixSumBuffer->GetUavCpuHandle());
        copyToMainHeap(slot + 1, superGroupSumBuffer->GetUavCpuHandle());
        pass4Shader = std::make_shared<ComputeShader>(device, "Shaders/prefixSumPass4_5CS.cso", cbv,
            mainAlloc.GetGpuHandle(slot),
            std::vector<ResPtr>{ superGroupSumBuffer->GetResourcePtr(),
                                    groupPrefixSumBuffer->GetResourcePtr() },
            std::vector<ResPtr>{ groupPrefixSumBuffer->GetResourcePtr() });

        // pass 5: prefixSumPass4CS reused — propagate groupPrefixSum into cellPrefixSum, N groups
        // UAV(u0-1) = 2 slots: [0]=cellPrefixSum, [1]=groupPrefixSum
        slot = mainAlloc.Allocate(2);
        copyToMainHeap(slot,     cellPrefixSumBuffer->GetUavCpuHandle());
        copyToMainHeap(slot + 1, groupPrefixSumBuffer->GetUavCpuHandle());
        pass5Shader = std::make_shared<ComputeShader>(device, "Shaders/prefixSumPass4_5CS.cso", cbv,
            mainAlloc.GetGpuHandle(slot),
            std::vector<ResPtr>{ groupPrefixSumBuffer->GetResourcePtr(),
                                    cellPrefixSumBuffer->GetResourcePtr() },
            std::vector<ResPtr>{ cellPrefixSumBuffer->GetResourcePtr() });

        // sortCS: UAV(u0-3) = 4 slots
        // [0]=predictedPosition front, [1]=cellCount, [2]=cellPrefixSum, [3]=perm
        slot = mainAlloc.Allocate(4);
        particleFields[PF_PREDICTED_POSITION]->registerFrontTarget(mainAlloc.GetCpuHandle(slot), false);
        copyToMainHeap(slot + 1, cellCountBuffer->GetUavCpuHandle());
        copyToMainHeap(slot + 2, cellPrefixSumBuffer->GetUavCpuHandle());
        copyToMainHeap(slot + 3, permBuffer->GetUavCpuHandle());
        sortShader = std::make_shared<ComputeShader>(device, "Shaders/sortCS.cso", cbv,
            mainAlloc.GetGpuHandle(slot),
            std::vector<ResPtr>{ particleFields[PF_PREDICTED_POSITION]->getFront()->GetResourcePtr(),
                                    cellPrefixSumBuffer->GetResourcePtr(),
                                    cellCountBuffer->GetResourcePtr() },
            std::vector<ResPtr>{ permBuffer->GetResourcePtr(),
                                    cellCountBuffer->GetResourcePtr() });

        // permutateCS: UAV(u0-14) = 15 slots
        // [0-6]=pf front, [7-13]=pf back, [14]=perm
        slot = mainAlloc.Allocate(15);
        for (UINT f = 0; f < PF_COUNT; f++)
            particleFields[f]->registerFrontTarget(mainAlloc.GetCpuHandle(slot + f),     false);
        for (UINT f = 0; f < PF_COUNT; f++)
            particleFields[f]->registerBackTarget( mainAlloc.GetCpuHandle(slot + 7 + f), false);
        copyToMainHeap(slot + 14, permBuffer->GetUavCpuHandle());
        std::vector<ResPtr> permIn, permOut;
        for (UINT f = 0; f < PF_COUNT; f++) permIn.push_back(particleFields[f]->getFront()->GetResourcePtr());
        permIn.push_back(permBuffer->GetResourcePtr());
        for (UINT f = 0; f < PF_COUNT; f++) permOut.push_back(particleFields[f]->getBack()->GetResourcePtr());
        permutateShader = std::make_shared<ComputeShader>(device, "Shaders/permutateCS.cso", cbv,
            mainAlloc.GetGpuHandle(slot), std::move(permIn), std::move(permOut));

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
