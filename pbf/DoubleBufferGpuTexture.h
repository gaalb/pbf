#pragma once
#include "GpuTexture.h"
#include <vector>
#include <string>

// Index-based double buffer for GpuTexture. flip() swaps frontIdx and copies descriptors to all
// registered main-heap target slots. resize2D() recreates both textures at new dimensions and
// overwrites the already-allocated static-heap descriptor slots in place.
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
        D3D12_CPU_DESCRIPTOR_HANDLE mainCpuHandle;
        bool isSrv;
        bool isFront;
    };
    std::vector<Target> targets;

    void rebuildDescriptors(UINT idx) {
        GpuTexture::P& tex = textures[idx];
        if (hasUav) tex->CreateUavAt(device, staticUavCpu[idx], D3D12_GPU_DESCRIPTOR_HANDLE{}, uavViewFormat);
        if (hasSrv) tex->CreateSrvAt(device, staticSrvCpu[idx], D3D12_GPU_DESCRIPTOR_HANDLE{}, srvViewFormat);
        if (hasDsv) tex->CreateDsvAt(device, staticDsvCpu[idx], dsvViewFormat);
    }

    void copyToTarget(const Target& t) const {
        const GpuTexture::P& tex = t.isFront ? textures[frontIdx] : textures[frontIdx ^ 1];
        D3D12_CPU_DESCRIPTOR_HANDLE src = t.isSrv ? tex->GetSrvCpuHandle() : tex->GetUavCpuHandle();
        device->CopyDescriptorsSimple(1, t.mainCpuHandle, src, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }

public:
    static P Create2D(
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
    {
        P db = std::make_shared<DoubleBufferGpuTexture>();
        db->device         = device;
        db->resourceFormat = resourceFormat;
        db->uavViewFormat  = uavFmt;
        db->srvViewFormat  = srvFmt;
        db->dsvViewFormat  = dsvFmt;
        db->resourceFlags  = flags;
        db->initialState   = initialState;
        db->hasUav = needUav; db->hasSrv = needSrv; db->hasDsv = needDsv;
        db->names[0] = name0 ? name0 : L"";
        db->names[1] = name1 ? name1 : L"";
        if (clearValue) { db->storedClearValue = *clearValue; db->hasClearValue = true; }

        for (UINT i = 0; i < 2; ++i) {
            const wchar_t* name = db->names[i].empty() ? nullptr : db->names[i].c_str();
            if (clearValue)
                db->textures[i] = GpuTexture::Create2DWithClearValue(device, w, h, resourceFormat, flags, name, initialState, clearValue);
            else
                db->textures[i] = GpuTexture::Create2D(device, w, h, resourceFormat, flags, name, initialState);

            if (needUav) {
                UINT slot = staticAlloc.Allocate();
                db->staticUavCpu[i] = staticAlloc.GetCpuHandle(slot);
                db->textures[i]->CreateUavAt(device, db->staticUavCpu[i], D3D12_GPU_DESCRIPTOR_HANDLE{}, uavFmt);
            }
            if (needSrv) {
                UINT slot = staticAlloc.Allocate();
                db->staticSrvCpu[i] = staticAlloc.GetCpuHandle(slot);
                db->textures[i]->CreateSrvAt(device, db->staticSrvCpu[i], D3D12_GPU_DESCRIPTOR_HANDLE{}, srvFmt);
            }
            if (needDsv) {
                db->textures[i]->CreateDsv(device, *dsvAlloc, dsvFmt);
                db->staticDsvCpu[i] = db->textures[i]->GetDsvCpuHandle();
            }
        }
        return db;
    }

    void registerFrontTarget(D3D12_CPU_DESCRIPTOR_HANDLE mainCpuHandle, bool isSrv) {
        targets.push_back({ mainCpuHandle, isSrv, /*isFront=*/true });
        copyToTarget(targets.back());
    }

    void registerBackTarget(D3D12_CPU_DESCRIPTOR_HANDLE mainCpuHandle, bool isSrv) {
        targets.push_back({ mainCpuHandle, isSrv, /*isFront=*/false });
        copyToTarget(targets.back());
    }

    void flip() {
        frontIdx ^= 1;
        for (const auto& t : targets)
            copyToTarget(t);
    }

    // Recreates both texture resources at new dimensions, rebuilds descriptors in-place at
    // the same static-heap slots, then refreshes all registered main-heap target slots.
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
