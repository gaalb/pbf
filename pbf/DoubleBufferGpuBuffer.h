#pragma once
#include "GpuBuffer.h"
#include <vector>

// Owns two GpuBuffer resources and a list of main-heap descriptor slots that always
// reflect the "active" (front) or "inactive" (back) resource.
//
// Front  = the resource currently consumed by readers (e.g. compute shaders, graphics materials).
// Back   = the resource currently available for writers (e.g. permutateCS output, snapshot copy destination).
//
// flip():
//   1. GpuBuffer::SwapInternals — exchanges the com_ptrs and all handles stored inside the two
//      GpuBuffer objects without moving the objects themselves. This preserves the stable-pointer
//      contract: ComputeShader stores com_ptr<ID3D12Resource>* addresses (via GetResourcePtr()),
//      which remain valid because SwapInternals changes the values-at-those-addresses, not the
//      addresses themselves.
//   2. CopyDescriptorsSimple — copies the now-active descriptor from the static (CPU-only) heap
//      to every registered main-heap target slot so shaders see the new resource.
//
// registerFrontTarget: the slot receives the FRONT descriptor after every flip.
// registerBackTarget:  the slot receives the BACK  descriptor after every flip.
//   Both variants also perform an immediate initial copy so the slot is populated right away.
//
// Static heap: the CPU-only staging heap. Both UAV and SRV descriptors for both resources
// are created here at construction time and never modified. They serve as CopyDescriptorsSimple sources.
GG_CLASS(DoubleBufferGpuBuffer)
    GpuBuffer::P front;
    GpuBuffer::P back;
    ID3D12Device* device = nullptr;

    struct Target {
        D3D12_CPU_DESCRIPTOR_HANDLE mainCpuHandle;
        bool isSrv;
        bool isFront;
    };
    std::vector<Target> targets;

    void copyToTarget(const Target& t) const {
        D3D12_CPU_DESCRIPTOR_HANDLE src;
        const GpuBuffer::P& buf = t.isFront ? front : back;
        src = t.isSrv ? buf->GetSrvCpuHandle() : buf->GetUavCpuHandle();
        device->CopyDescriptorsSimple(1, t.mainCpuHandle, src, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }

public:
    static P Create(
        ID3D12Device* device,
        UINT elems, UINT stride,
        const wchar_t* frontName, const wchar_t* backName,
        D3D12_RESOURCE_STATES initialState,
        DescriptorAllocator& staticAlloc,
        bool needUav, bool needSrv)
    {
        P db = std::make_shared<DoubleBufferGpuBuffer>();
        db->device = device;

        db->front = GpuBuffer::Create(device, elems, stride, frontName, initialState, D3D12_HEAP_TYPE_DEFAULT);
        db->back  = GpuBuffer::Create(device, elems, stride, backName,  initialState, D3D12_HEAP_TYPE_DEFAULT);

        if (needUav) {
            db->front->CreateUav(device, staticAlloc);
            db->back->CreateUav(device, staticAlloc);
        }
        if (needSrv) {
            db->front->CreateSrv(device, staticAlloc);
            db->back->CreateSrv(device, staticAlloc);
        }

        return db;
    }

    // Register a slot that always receives the FRONT resource's descriptor.
    // Performs an immediate initial copy so the slot is ready before first use.
    void registerFrontTarget(D3D12_CPU_DESCRIPTOR_HANDLE mainCpuHandle, bool isSrv) {
        targets.push_back({ mainCpuHandle, isSrv, /*isFront=*/true });
        copyToTarget(targets.back());
    }

    // Register a slot that always receives the BACK resource's descriptor.
    void registerBackTarget(D3D12_CPU_DESCRIPTOR_HANDLE mainCpuHandle, bool isSrv) {
        targets.push_back({ mainCpuHandle, isSrv, /*isFront=*/false });
        copyToTarget(targets.back());
    }

    void flip() {
        GpuBuffer::SwapInternals(front, back);
        for (const auto& t : targets)
            copyToTarget(t);
    }

    GpuBuffer::P getFront() const { return front; }
    GpuBuffer::P getBack()  const { return back;  }

GG_ENDCLASS
