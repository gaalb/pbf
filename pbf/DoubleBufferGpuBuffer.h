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
// Main heap: the GPU-visible heap. Target slots here are registered via registerFront/BackTarget(),
// and are updated on every flip to point to the currently active resource's descriptor in the static
// heap. The mental image should be that of a switchboard, where signals have origins, and targets,
// where we connect the signals. A double buffer is what allows us to have two sets of signal origins,
// which we can flip between. It keeps a list of the signal targets, i.e. the main heap slots that need 
// to be updated on every flip. The origin is the static heap.
GG_CLASS(DoubleBufferGpuBuffer)
    GpuBuffer::P buffers[2];
    UINT frontIdx = 0;
    ID3D12Device* device = nullptr;

    struct Target {
		D3D12_CPU_DESCRIPTOR_HANDLE mainCpuHandle; // the slot in the main heap to update on flip()
        bool isSrv;
        bool isFront;
    };
	std::vector<Target> targets; // which main heap slots observe this double buffer

	// Wire the appropriate static heap descriptor (front or back, UAV or SRV) to the target slot in the main heap.
    void copyToTarget(const Target& t) const {
		const GpuBuffer::P& buf = buffers[t.isFront ? frontIdx : frontIdx ^ 1]; // the GpuBuffer that holds the relevant static heap descriptors
        D3D12_CPU_DESCRIPTOR_HANDLE src; // the static heap descriptor to copy from
		src = t.isSrv ? buf->GetSrvCpuHandle() : buf->GetUavCpuHandle(); // choose UAV vs SRV descriptor
        device->CopyDescriptorsSimple(1, t.mainCpuHandle, src, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }

public:
    DoubleBufferGpuBuffer(
        ID3D12Device* device,
        UINT elems, UINT stride,
        const wchar_t* frontName, const wchar_t* backName,
        D3D12_RESOURCE_STATES initialState,
        DescriptorAllocator& staticAlloc)
        : device(device)
    {
        buffers[0] = GpuBuffer::Create(device, elems, stride, frontName, initialState, D3D12_HEAP_TYPE_DEFAULT);
        buffers[1] = GpuBuffer::Create(device, elems, stride, backName,  initialState, D3D12_HEAP_TYPE_DEFAULT);
        buffers[0]->CreateUav(device, staticAlloc);
        buffers[1]->CreateUav(device, staticAlloc);
        buffers[0]->CreateSrv(device, staticAlloc);
        buffers[1]->CreateSrv(device, staticAlloc);
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

    GpuBuffer::P getFront() const { return buffers[frontIdx]; }
    GpuBuffer::P getBack()  const { return buffers[frontIdx ^ 1]; }

GG_ENDCLASS
