#pragma once
#include <Egg/Common.h>
#include <Egg/d3dx12.h>
#include "DescriptorAllocator.h"

// Wraps a D3D12 2D or 3D texture resource together with its UAV/SRV/DSV descriptors.
GG_CLASS(GpuTexture)
    com_ptr<ID3D12Resource> resource;
    UINT width = 0, height = 0, depth = 1;
    DXGI_FORMAT resourceFormat = DXGI_FORMAT_UNKNOWN;
    bool is3D = false;
    D3D12_RESOURCE_STATES currentState = D3D12_RESOURCE_STATE_COMMON;
    D3D12_CPU_DESCRIPTOR_HANDLE uavCpuHandle{};
    D3D12_GPU_DESCRIPTOR_HANDLE uavGpuHandle{};
    D3D12_CPU_DESCRIPTOR_HANDLE srvCpuHandle{};
    D3D12_GPU_DESCRIPTOR_HANDLE srvGpuHandle{};
    // DSV is bound via OMSetRenderTarget, which takes a CPU handle, unlike UAV/SRV which are
	// bound via SetComputeRootDescriptorTable/SetGraphicsRootDescriptorTable requiring a GPU handle.
	// Since DSV never goes through a root descriptor table, we only need to store the CPU handle for it.
    D3D12_CPU_DESCRIPTOR_HANDLE dsvCpuHandle{}; 

public:
	GpuTexture() = default; // allow default construction; use static Create methods to create actual textures.

	// Static factory method to create a 2D texture. Use Create2DWithClearValue if you need a clear value.
    static P Create2D(ID3D12Device* device, UINT w, UINT h,
                      DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags,
                      const wchar_t* name, D3D12_RESOURCE_STATES initialState)
    {
		// Create a new GpuTexture object and fill in its metadata fields.
        P tex = std::make_shared<GpuTexture>();
        tex->width = w; tex->height = h; tex->depth = 1;
        tex->resourceFormat = format; tex->is3D = false;
        tex->currentState = initialState;

        DX_API("Failed to create 2D texture")
            device->CreateCommittedResource(
				&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), // default heap for GPU read/write access
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Tex2D(format, w, h, 1, 1, 1, 0, flags),
                initialState, nullptr, // nullptr here means no clear value
				IID_PPV_ARGS(tex->resource.ReleaseAndGetAddressOf())); // fill the com_ptr with the created resource
        if (name) tex->resource->SetName(name);
        return tex;
    }

    static P Create2DWithClearValue(ID3D12Device* device, UINT w, UINT h,
                                    DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags,
                                    const wchar_t* name, D3D12_RESOURCE_STATES initialState,
                                    const D3D12_CLEAR_VALUE* clearValue)
    {
        P tex = std::make_shared<GpuTexture>();
        tex->width = w; tex->height = h; tex->depth = 1;
        tex->resourceFormat = format; tex->is3D = false;
        tex->currentState = initialState;

        DX_API("Failed to create 2D texture")
            device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Tex2D(format, w, h, 1, 1, 1, 0, flags),
                initialState, clearValue,
                IID_PPV_ARGS(tex->resource.ReleaseAndGetAddressOf()));
        if (name) tex->resource->SetName(name);
        return tex;
    }

    static P Create3D(ID3D12Device* device, UINT w, UINT h, UINT d,
                      DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags,
                      const wchar_t* name, D3D12_RESOURCE_STATES initialState)
    {
        P tex = std::make_shared<GpuTexture>();
        tex->width = w; tex->height = h; tex->depth = d;
        tex->resourceFormat = format; tex->is3D = true;
        tex->currentState = initialState;

        DX_API("Failed to create 3D texture")
            device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Tex3D(format, w, h, (UINT16)d, 1, flags),
                initialState, nullptr,
                IID_PPV_ARGS(tex->resource.ReleaseAndGetAddressOf()));
        if (name) tex->resource->SetName(name);
        return tex;
    }

    com_ptr<ID3D12Resource>* GetResourcePtr() { return std::addressof(resource); }

    // Auto-allocate one slot from alloc, create a UAV (optionally with format override),
    // and store the resulting cpu/gpu handle pair as the canonical UAV for this texture.
    // Retrieve them later via GetUavCpuHandle() / GetUavGpuHandle(). Call at most once
    // per texture; a second call would overwrite the stored handles. For additional views
    // at specific locations (e.g. a CPU-only UAV in a non-shader-visible heap), use
    // CreateUavAt() instead.
    void CreateUav(ID3D12Device* device, DescriptorAllocator& alloc,
                   DXGI_FORMAT viewFormat = DXGI_FORMAT_UNKNOWN) {
        UINT slot = alloc.Allocate();
        uavCpuHandle = alloc.GetCpuHandle(slot);
        uavGpuHandle = alloc.GetGpuHandle(slot);
        CreateUavAt(device, uavCpuHandle, uavGpuHandle, viewFormat);
    }

    // Auto-allocate one slot from alloc, create an SRV (optionally with format override),
    // and store the resulting cpu/gpu handle pair as the canonical SRV for this texture.
    // Retrieve them later via GetSrvCpuHandle() / GetSrvGpuHandle(). Call at most once
    // per texture; a second call would overwrite the stored handles. For additional views
    // at specific locations, use CreateSrvAt() instead.
    void CreateSrv(ID3D12Device* device, DescriptorAllocator& alloc,
                   DXGI_FORMAT viewFormat = DXGI_FORMAT_UNKNOWN) {
        UINT slot = alloc.Allocate();
        srvCpuHandle = alloc.GetCpuHandle(slot);
        srvGpuHandle = alloc.GetGpuHandle(slot);
        CreateSrvAt(device, srvCpuHandle, srvGpuHandle, viewFormat);
    }

    // Auto-allocate one slot from dsvAlloc, create a DSV, and store the resulting cpu
    // handle as the canonical DSV for this texture. DSV is CPU-only (bound via
    // OMSetRenderTargets, not a root descriptor table), so only a cpu handle is kept.
    // Call at most once per texture; a second call would overwrite the stored handle.
    // For re-creating the view at an existing slot (e.g. on resize), use CreateDsvAt().
    void CreateDsv(ID3D12Device* device, DescriptorAllocator& dsvAlloc,
                   DXGI_FORMAT dsvFormat = DXGI_FORMAT_D32_FLOAT) {
        UINT slot = dsvAlloc.Allocate();
        CreateDsvAt(device, dsvAlloc.GetCpuHandle(slot), dsvFormat);
    }

    // Create a DSV at an already-allocated cpu handle and store it.
    // Unlike CreateUavAt/CreateSrvAt, this DOES store the handle: DSV is always
    // single-slot (no risk of a secondary view clobbering a canonical one), and
    // the resize path needs to rewrite the view into the same pre-allocated slot.
    void CreateDsvAt(ID3D12Device* device, D3D12_CPU_DESCRIPTOR_HANDLE cpu,
                     DXGI_FORMAT dsvFormat = DXGI_FORMAT_D32_FLOAT) {
        D3D12_DEPTH_STENCIL_VIEW_DESC d = {};
        d.Format = dsvFormat;
        d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        d.Flags = D3D12_DSV_FLAG_NONE;
        dsvCpuHandle = cpu;
        device->CreateDepthStencilView(resource.Get(), &d, dsvCpuHandle);
    }

    // Create a UAV at a caller-specified cpu/gpu handle pair, optionally reinterpreting
    // the resource through viewFormat (needed for R32_TYPELESS resources that must be
    // viewed as e.g. R32_UINT for UAV atomic writes and R32_FLOAT for SRV reads).
    // Does NOT store the handles — the caller already holds them and is responsible for
    // keeping them. Use this for secondary / ad-hoc views where
    // GetUavCpuHandle/GpuHandle should not be affected.
    void CreateUavAt(ID3D12Device* device,
                     D3D12_CPU_DESCRIPTOR_HANDLE cpu,
                     D3D12_GPU_DESCRIPTOR_HANDLE gpu,
                     DXGI_FORMAT viewFormat = DXGI_FORMAT_UNKNOWN) {
        DXGI_FORMAT fmt = (viewFormat != DXGI_FORMAT_UNKNOWN) ? viewFormat : resourceFormat;
        D3D12_UNORDERED_ACCESS_VIEW_DESC d = {};
        d.Format = fmt;
        if (is3D) {
            d.ViewDimension          = D3D12_UAV_DIMENSION_TEXTURE3D;
            d.Texture3D.MipSlice     = 0;
            d.Texture3D.FirstWSlice  = 0;
            d.Texture3D.WSize        = depth;
        } else {
            d.ViewDimension          = D3D12_UAV_DIMENSION_TEXTURE2D;
            d.Texture2D.MipSlice     = 0;
        }
        device->CreateUnorderedAccessView(resource.Get(), nullptr, &d, cpu);
        // handles not stored: caller holds cpu/gpu directly
    }

    // Create an SRV at a caller-specified cpu/gpu handle pair, optionally reinterpreting
    // the resource through viewFormat (needed for R32_TYPELESS resources).
    // Does NOT store the handles — the caller already holds them and is responsible for
    // keeping them. Use this for secondary / ad-hoc views where
    // GetSrvCpuHandle/GpuHandle should not be affected.
    void CreateSrvAt(ID3D12Device* device,
                     D3D12_CPU_DESCRIPTOR_HANDLE cpu,
                     D3D12_GPU_DESCRIPTOR_HANDLE gpu,
                     DXGI_FORMAT viewFormat = DXGI_FORMAT_UNKNOWN) {
        DXGI_FORMAT fmt = (viewFormat != DXGI_FORMAT_UNKNOWN) ? viewFormat : resourceFormat;
        D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
        d.Format                  = fmt;
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if (is3D) {
            d.ViewDimension                   = D3D12_SRV_DIMENSION_TEXTURE3D;
            d.Texture3D.MostDetailedMip       = 0;
            d.Texture3D.MipLevels             = 1;
            d.Texture3D.ResourceMinLODClamp   = 0.0f;
        } else {
            d.ViewDimension                   = D3D12_SRV_DIMENSION_TEXTURE2D;
            d.Texture2D.MostDetailedMip       = 0;
            d.Texture2D.MipLevels             = 1;
        }
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
    D3D12_CPU_DESCRIPTOR_HANDLE GetDsvCpuHandle() const { return dsvCpuHandle; }

GG_ENDCLASS
