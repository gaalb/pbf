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

    // Auto-allocate one slot and create a UAV (optionally with format override).
    void CreateUav(ID3D12Device* device, DescriptorAllocator& alloc,
                   DXGI_FORMAT viewFormat = DXGI_FORMAT_UNKNOWN) {
        UINT slot = alloc.Allocate();
        CreateUavAt(device, alloc.GetCpuHandle(slot), alloc.GetGpuHandle(slot), viewFormat);
    }

    // Auto-allocate one slot and create an SRV (optionally with format override).
    void CreateSrv(ID3D12Device* device, DescriptorAllocator& alloc,
                   DXGI_FORMAT viewFormat = DXGI_FORMAT_UNKNOWN) {
        UINT slot = alloc.Allocate();
        CreateSrvAt(device, alloc.GetCpuHandle(slot), alloc.GetGpuHandle(slot), viewFormat);
    }

    // Auto-allocate one slot in a DSV allocator and create a DSV.
    void CreateDsv(ID3D12Device* device, DescriptorAllocator& dsvAlloc,
                   DXGI_FORMAT dsvFormat = DXGI_FORMAT_D32_FLOAT) {
        UINT slot = dsvAlloc.Allocate();
        CreateDsvAt(device, dsvAlloc.GetCpuHandle(slot), dsvFormat);
    }

    // Create a DSV at an already-allocated CPU handle (used for resize to reuse existing slots).
    void CreateDsvAt(ID3D12Device* device, D3D12_CPU_DESCRIPTOR_HANDLE cpu,
                     DXGI_FORMAT dsvFormat = DXGI_FORMAT_D32_FLOAT) {
        D3D12_DEPTH_STENCIL_VIEW_DESC d = {};
        d.Format = dsvFormat;
        d.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        d.Flags = D3D12_DSV_FLAG_NONE;
        dsvCpuHandle = cpu;
        device->CreateDepthStencilView(resource.Get(), &d, dsvCpuHandle);
    }

    void CreateUavAt(ID3D12Device* device,
                     D3D12_CPU_DESCRIPTOR_HANDLE cpu,
                     D3D12_GPU_DESCRIPTOR_HANDLE gpu,
                     DXGI_FORMAT viewFormat = DXGI_FORMAT_UNKNOWN) {
        // If a view format was specified, use it, otherwise fall back to the texture's native format.
        // In some cases, we might want to view texture data through a different format when viewed
		// through different descriptors. This can only be done if the underyling resource format is 
        // compatible with the view format, which means we will create such textures with 
        // R32_TYPELESS as format. In such cases, we specify the format at descriptor creation time, rather
		// than resource creation time; we specify the viewFormat in the parameters of CreateUavAt/CreateSrvAt.
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
        uavCpuHandle = cpu;
        uavGpuHandle = gpu;
    }

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
        srvCpuHandle = cpu;
        srvGpuHandle = gpu;
    }

    ID3D12Resource* Get() const { return resource.Get(); }
    D3D12_CPU_DESCRIPTOR_HANDLE GetUavCpuHandle() const { return uavCpuHandle; }
    D3D12_GPU_DESCRIPTOR_HANDLE GetUavGpuHandle() const { return uavGpuHandle; }
    D3D12_CPU_DESCRIPTOR_HANDLE GetSrvCpuHandle() const { return srvCpuHandle; }
    D3D12_GPU_DESCRIPTOR_HANDLE GetSrvGpuHandle() const { return srvGpuHandle; }
    D3D12_CPU_DESCRIPTOR_HANDLE GetDsvCpuHandle() const { return dsvCpuHandle; }

GG_ENDCLASS
