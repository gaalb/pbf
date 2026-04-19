#pragma once
#include "GpuTexture.h"
#include <vector>
#include <string>

// Owns two GpuTexture resources and a list of main-heap descriptor slots that always
// reflect the "active" (front) or "inactive" (back) resource.
//
// Front  = the resource currently consumed by readers (e.g. compute shaders, graphics materials).
// Back   = the resource currently available for writers (e.g. render target output, snapshot copy destination).
//
// flip():
//   1. frontIdx ^= 1 — advances the front index to point to the opposite buffer. The physical
//      GpuTexture objects stay in place; only which one is considered "front" changes.
//   2. CopyDescriptorsSimple — copies the now-active descriptor from the static (CPU-only) heap
//      to every registered main-heap target slot so shaders see the new resource.
//
// registerFrontTarget: the slot receives the FRONT descriptor after every flip.
// registerBackTarget:  the slot receives the BACK  descriptor after every flip.
//   Both variants also perform an immediate initial copy so the slot is populated right away.
//
// Static heap: the CPU-only staging heap. Both UAV and SRV descriptors for both resources
// are created here at construction time and never modified. They serve as CopyDescriptorsSimple sources.
// Main heap: the GPU-visible heap. Target slots here are registered via registerFront/BackTarget(),
// and are updated on every flip to point to the currently active resource's descriptor in the static
// heap. The mental image should be that of a switchboard, where signals have origins, and targets,
// where we connect the signals. A double buffer is what allows us to have two sets of signal origins,
// which we can flip between. It keeps a list of the signal targets, i.e. the main heap slots that need
// to be updated on every flip. The origin is the static heap.
//
// resize2D() recreates both texture resources at new dimensions, rebuilds descriptors in-place at
// the same static-heap slots, then refreshes all registered main-heap target slots.
GG_CLASS(DoubleBufferGpuTexture)
    GpuTexture::P textures[2];
    UINT frontIdx = 0;
    ID3D12Device* device = nullptr;

    // Static-heap CPU handles allocated once at creation; reused by rebuildDescriptors on resize.
    D3D12_CPU_DESCRIPTOR_HANDLE staticUavCpu[2]{};
    D3D12_CPU_DESCRIPTOR_HANDLE staticSrvCpu[2]{};
    D3D12_CPU_DESCRIPTOR_HANDLE staticDsvCpu[2]{};

    DXGI_FORMAT resourceFormat = DXGI_FORMAT_UNKNOWN;
    DXGI_FORMAT uavViewFormat  = DXGI_FORMAT_UNKNOWN;
    DXGI_FORMAT srvViewFormat  = DXGI_FORMAT_UNKNOWN;
    DXGI_FORMAT dsvViewFormat  = DXGI_FORMAT_D32_FLOAT;
    D3D12_RESOURCE_FLAGS resourceFlags = D3D12_RESOURCE_FLAG_NONE;
    D3D12_RESOURCE_STATES initialState = D3D12_RESOURCE_STATE_COMMON;
    D3D12_CLEAR_VALUE storedClearValue{};
    bool hasClearValue = false;
    bool hasUav = false, hasSrv = false, hasDsv = false;
    std::wstring names[2];

    struct Target {
        D3D12_CPU_DESCRIPTOR_HANDLE mainCpuHandle; // the slot in the main heap to update on flip()
        bool isSrv;
        bool isFront;
    };
    std::vector<Target> targets; // which main heap slots observe this double buffer

    void rebuildDescriptors(UINT idx) {
        GpuTexture::P& tex = textures[idx];
        if (hasUav) tex->CreateUavAt(device, staticUavCpu[idx], D3D12_GPU_DESCRIPTOR_HANDLE{}, uavViewFormat);
        if (hasSrv) tex->CreateSrvAt(device, staticSrvCpu[idx], D3D12_GPU_DESCRIPTOR_HANDLE{}, srvViewFormat);
        if (hasDsv) tex->CreateDsvAt(device, staticDsvCpu[idx], dsvViewFormat);
    }

    // Wire the appropriate static heap descriptor (front or back, UAV or SRV) to the target slot in the main heap.
    void copyToTarget(const Target& t) const {
        const GpuTexture::P& tex = t.isFront ? textures[frontIdx] : textures[frontIdx ^ 1]; // the GpuTexture that holds the relevant static heap descriptors
        D3D12_CPU_DESCRIPTOR_HANDLE src = t.isSrv ? tex->GetSrvCpuHandle() : tex->GetUavCpuHandle(); // choose UAV vs SRV descriptor
        device->CopyDescriptorsSimple(1, t.mainCpuHandle, src, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }

public:
    DoubleBufferGpuTexture(
        ID3D12Device* device,
        UINT w, UINT h,
        DXGI_FORMAT resourceFormat, D3D12_RESOURCE_FLAGS flags,
        const wchar_t* name0, const wchar_t* name1,
        D3D12_RESOURCE_STATES initialState,
        const D3D12_CLEAR_VALUE* clearValue,
        DescriptorAllocator& staticAlloc,
        DescriptorAllocator* dsvAlloc,
        bool needUav, bool needSrv, bool needDsv,
        DXGI_FORMAT uavFmt = DXGI_FORMAT_UNKNOWN,
        DXGI_FORMAT srvFmt = DXGI_FORMAT_UNKNOWN,
        DXGI_FORMAT dsvFmt = DXGI_FORMAT_D32_FLOAT)
        : device(device),
          resourceFormat(resourceFormat),
          uavViewFormat(uavFmt),
          srvViewFormat(srvFmt),
          dsvViewFormat(dsvFmt),
          resourceFlags(flags),
          initialState(initialState),
          hasUav(needUav), hasSrv(needSrv), hasDsv(needDsv)
    {
        names[0] = name0 ? name0 : L"";
        names[1] = name1 ? name1 : L"";
        if (clearValue) { storedClearValue = *clearValue; hasClearValue = true; }

        for (UINT i = 0; i < 2; ++i) {
            const wchar_t* name = names[i].empty() ? nullptr : names[i].c_str();
            if (clearValue)
                textures[i] = GpuTexture::Create2DWithClearValue(device, w, h, resourceFormat, flags, name, initialState, clearValue);
            else
                textures[i] = GpuTexture::Create2D(device, w, h, resourceFormat, flags, name, initialState);

            if (needUav) {
                UINT slot = staticAlloc.Allocate();
                staticUavCpu[i] = staticAlloc.GetCpuHandle(slot);
                textures[i]->CreateUavAt(device, staticUavCpu[i], D3D12_GPU_DESCRIPTOR_HANDLE{}, uavFmt);
            }
            if (needSrv) {
                UINT slot = staticAlloc.Allocate();
                staticSrvCpu[i] = staticAlloc.GetCpuHandle(slot);
                textures[i]->CreateSrvAt(device, staticSrvCpu[i], D3D12_GPU_DESCRIPTOR_HANDLE{}, srvFmt);
            }
            if (needDsv) {
                textures[i]->CreateDsv(device, *dsvAlloc, dsvFmt);
                staticDsvCpu[i] = textures[i]->GetDsvCpuHandle();
            }
        }
    }

    // Register a slot that always receives the FRONT resource's descriptor.
    // Performs an immediate initial copy so the slot is ready before first use.
    void registerFrontTarget(D3D12_CPU_DESCRIPTOR_HANDLE mainCpuHandle, bool isSrv) {
        targets.push_back({ mainCpuHandle, isSrv, /*isFront=*/true }); // register wiring rules
        copyToTarget(targets.back()); // perform initial copy so the slot is populated right away
    }

    // Register a slot that always receives the BACK resource's descriptor.
    void registerBackTarget(D3D12_CPU_DESCRIPTOR_HANDLE mainCpuHandle, bool isSrv) {
        targets.push_back({ mainCpuHandle, isSrv, /*isFront=*/false }); // register wiring rules
        copyToTarget(targets.back()); // perform initial copy so the slot is populated right away
    }

    void flip() {
        frontIdx ^= 1; // swap which buffer is considered front
        for (const auto& t : targets) // update wiring
            copyToTarget(t);
    }

    void resize2D(UINT w, UINT h) {
        const D3D12_CLEAR_VALUE* cv = hasClearValue ? &storedClearValue : nullptr;
        for (UINT i = 0; i < 2; ++i) {
            const wchar_t* name = names[i].empty() ? nullptr : names[i].c_str();
            if (cv)
                textures[i] = GpuTexture::Create2DWithClearValue(device, w, h, resourceFormat, resourceFlags, name, initialState, cv);
            else
                textures[i] = GpuTexture::Create2D(device, w, h, resourceFormat, resourceFlags, name, initialState);
            rebuildDescriptors(i);
        }
        for (const auto& t : targets)
            copyToTarget(t);
    }

    GpuTexture::P getFront() const { return textures[frontIdx]; }
    GpuTexture::P getBack()  const { return textures[frontIdx ^ 1]; }

GG_ENDCLASS
