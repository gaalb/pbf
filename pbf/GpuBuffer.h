#pragma once
#include <Egg/Common.h>
#include <Egg/d3dx12.h>
#include "DescriptorAllocator.h"

// Wraps a D3D12 structured or raw buffer resource together with its UAV/SRV descriptors.
// CPU handles are for creating the UAV/SRV views; GPU handles are for binding to shaders. 
// They come in pairs because D3D12 requires both - CPU side to populate the descriptor, 
// GPU side for the command list to reference it.
GG_CLASS(GpuBuffer)
    com_ptr<ID3D12Resource> resource;
    UINT elementCount = 0;
    UINT stride  = 0;
    D3D12_RESOURCE_STATES currentState = D3D12_RESOURCE_STATE_COMMON;
    D3D12_CPU_DESCRIPTOR_HANDLE uavCpuHandle{};
    D3D12_GPU_DESCRIPTOR_HANDLE uavGpuHandle{};
    D3D12_CPU_DESCRIPTOR_HANDLE srvCpuHandle{};
    D3D12_GPU_DESCRIPTOR_HANDLE srvGpuHandle{};

public:
    GpuBuffer(ID3D12Device* device, UINT elems, UINT elemStride,
              const wchar_t* name,
              D3D12_RESOURCE_STATES initialState,
              D3D12_HEAP_TYPE heapType)
        : elementCount(elems), stride(elemStride), currentState(initialState)
    {
        UINT64 bufSize = (UINT64)elems * elemStride;
        D3D12_RESOURCE_FLAGS flags = (heapType == D3D12_HEAP_TYPE_DEFAULT)
            ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS // CS reads/writes from default heap uavs
			: D3D12_RESOURCE_FLAG_NONE; // readback buffers and upload buffers don't need UAV flags

        DX_API("Failed to create GPU buffer")
            device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(heapType),
				D3D12_HEAP_FLAG_NONE, // no special flags for the heap
				&CD3DX12_RESOURCE_DESC::Buffer(bufSize, flags), // CD3DX12 helper
                initialState, nullptr,
                IID_PPV_ARGS(resource.ReleaseAndGetAddressOf()));
        if (name) resource->SetName(name);
    }

    // Returns a stable pointer to the internal com_ptr, for use in ComputeShader inputs/outputs.
    // The address is stable for the lifetime of this object; SwapInternals changes the value
    // at this address without changing the address itself.
    com_ptr<ID3D12Resource>* GetResourcePtr() { return std::addressof(resource); }

    // Swap internals (resource + all handles) of two GpuBuffer objects in-place,
    // preserving each shared_ptr's identity so stored com_ptr* pointers remain valid.
    static void SwapInternals(P& a, P& b) {
        std::swap(a->resource, b->resource);
        std::swap(a->elementCount, b->elementCount);
        std::swap(a->stride, b->stride);
        std::swap(a->currentState, b->currentState);
        std::swap(a->uavCpuHandle, b->uavCpuHandle);
        std::swap(a->uavGpuHandle, b->uavGpuHandle);
        std::swap(a->srvCpuHandle, b->srvCpuHandle);
        std::swap(a->srvGpuHandle, b->srvGpuHandle);
    }

    // Auto-allocate one slot from alloc, create a structured UAV, and store the resulting
    // cpu/gpu handle pair as the canonical UAV for this buffer. Retrieve them later via
    // GetUavCpuHandle() / GetUavGpuHandle(). Call at most once per buffer; a second call
    // would overwrite the stored handles. For additional views at specific locations, use
    // CreateUavAt() instead.
    void CreateUav(ID3D12Device* device, DescriptorAllocator& alloc) {
        UINT slot = alloc.Allocate();
        uavCpuHandle = alloc.GetCpuHandle(slot);
        uavGpuHandle = alloc.GetGpuHandle(slot);
        CreateUavAt(device, uavCpuHandle, uavGpuHandle);
    }

    // Auto-allocate one slot from alloc, create a structured SRV, and store the resulting
    // cpu/gpu handle pair as the canonical SRV for this buffer. Retrieve them later via
    // GetSrvCpuHandle() / GetSrvGpuHandle(). Call at most once per buffer; a second call
    // would overwrite the stored handles. For additional views at specific locations, use
    // CreateSrvAt() instead.
    void CreateSrv(ID3D12Device* device, DescriptorAllocator& alloc) {
        UINT slot = alloc.Allocate();
        srvCpuHandle = alloc.GetCpuHandle(slot);
        srvGpuHandle = alloc.GetGpuHandle(slot);
        CreateSrvAt(device, srvCpuHandle, srvGpuHandle);
    }

    // Create a structured UAV at a caller-specified cpu/gpu handle pair.
    // Does NOT store the handles — the caller already holds them and is responsible
    // for keeping them. Use this for secondary / ad-hoc views (e.g. a CPU-only UAV
    // in a non-shader-visible heap) where GetUavCpuHandle/GpuHandle should not be affected.
    void CreateUavAt(ID3D12Device* device,
                     D3D12_CPU_DESCRIPTOR_HANDLE cpu,
                     D3D12_GPU_DESCRIPTOR_HANDLE gpu) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC d = {};
        d.Format = DXGI_FORMAT_UNKNOWN; // structured buffer, so format is unknown
        d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; // this is a buffer UAV, not a texture UAV
        d.Buffer.NumElements = elementCount;
        d.Buffer.StructureByteStride = stride;
        device->CreateUnorderedAccessView(resource.Get(), nullptr, &d, cpu);
        // handles not stored: caller holds cpu/gpu directly
    }

    // Create a structured SRV at a caller-specified cpu/gpu handle pair.
    // Does NOT store the handles — the caller already holds them and is responsible
    // for keeping them. Use this for secondary / ad-hoc views where
    // GetSrvCpuHandle/GpuHandle should not be affected.
    void CreateSrvAt(ID3D12Device* device,
                     D3D12_CPU_DESCRIPTOR_HANDLE cpu,
                     D3D12_GPU_DESCRIPTOR_HANDLE gpu) {
        D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
        d.Format = DXGI_FORMAT_UNKNOWN; // structured buffer, so format is unknown
        d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER; // this is a buffer SRV, not a texture SRV
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Buffer.NumElements = elementCount;
        d.Buffer.StructureByteStride = stride;
        device->CreateShaderResourceView(resource.Get(), &d, cpu);
        // handles not stored: caller holds cpu/gpu directly
    }

    void Transition(D3D12_RESOURCE_STATES destState, ID3D12GraphicsCommandList* cmdList) {
        if (currentState == destState) return;
        auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(resource.Get(), currentState, destState);
        cmdList->ResourceBarrier(1, &barrier);
        currentState = destState;
    }

    ID3D12Resource* Get() const { return resource.Get(); }
    D3D12_CPU_DESCRIPTOR_HANDLE GetUavCpuHandle() const { return uavCpuHandle; }
    D3D12_GPU_DESCRIPTOR_HANDLE GetUavGpuHandle() const { return uavGpuHandle; }
    D3D12_CPU_DESCRIPTOR_HANDLE GetSrvCpuHandle() const { return srvCpuHandle; }
    D3D12_GPU_DESCRIPTOR_HANDLE GetSrvGpuHandle() const { return srvGpuHandle; }
    UINT GetElementCount() const { return elementCount; }
    UINT GetStride()       const { return stride; }

GG_ENDCLASS
