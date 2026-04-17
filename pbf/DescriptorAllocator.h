#pragma once
#include <Egg/Common.h>
#include <Egg/d3dx12.h>
#include <cassert>

// Linear descriptor allocator over a single D3D12 descriptor heap.
// Slots are allocated sequentially.
GG_CLASS(DescriptorAllocator)
    com_ptr<ID3D12DescriptorHeap> heap;
    UINT descriptorSize = 0;
    UINT capacity = 0;
    UINT nextSlot = 0;
	bool shaderVisible = false;

public:
    DescriptorAllocator(ID3D12Device* device, D3D12_DESCRIPTOR_HEAP_TYPE type,
                        UINT cap, bool shaderVisible)
		: capacity(cap), shaderVisible(shaderVisible)
    {
		D3D12_DESCRIPTOR_HEAP_DESC desc = {}; // zero-initialize all fields
        desc.Type = type; // CBV_SRV_UAV,
		desc.NumDescriptors = cap; // how many descriptors we want to allocate in total
        desc.Flags = shaderVisible // can shaders access it?
            ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE // yes
            : D3D12_DESCRIPTOR_HEAP_FLAG_NONE; // no
        DX_API("Failed to create descriptor heap")
            device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(heap.GetAddressOf()));
		descriptorSize = device->GetDescriptorHandleIncrementSize(type); // bytes between descriptors
    }

    // Allocate one slot; returns its index.
    // Allocating means that the value returned is owned by whoever called
    // Allocate(), and that slot (number) will not be returned by future calls,
    // since the internal counter gets incremented.
    UINT Allocate() {
        assert(nextSlot < capacity && "DescriptorAllocator overflow");
        return nextSlot++;
    }

    // Allocate a contiguous block of count slots; returns the index of the first slot.
    UINT Allocate(UINT count) {
        assert(nextSlot + count <= capacity && "DescriptorAllocator overflow");
		UINT first = nextSlot; // save the index before incrementing
        nextSlot += count; // increment
        return first; // return the value saved before incrementing
    }

	// CPU handle: used by the CPU to create descriptor views (CBV/SRV/UAV/DSV),
    // and as source/destination for CopyDescriptorsSimple, every heap has one.
    D3D12_CPU_DESCRIPTOR_HANDLE GetCpuHandle(UINT slot) const {
        return CD3DX12_CPU_DESCRIPTOR_HANDLE(
            heap->GetCPUDescriptorHandleForHeapStart(), slot, descriptorSize);
    }

	// GPU handle: passed to command lists via SetComputeRootDescriptorTable 
    // or SetGraphicsRootDescriptorTable so shaders can actually read the descriptor
	// at draw/dispatch time. Only valid if the heap was created with shader visibility.
    D3D12_GPU_DESCRIPTOR_HANDLE GetGpuHandle(UINT slot) const {
        return shaderVisible ? CD3DX12_GPU_DESCRIPTOR_HANDLE(
            heap->GetGPUDescriptorHandleForHeapStart(), slot, descriptorSize)
			: CD3DX12_GPU_DESCRIPTOR_HANDLE{}; // return null handle if not shader visible
    }

    // Underlying heap pointer
    ID3D12DescriptorHeap* GetHeap() const { return heap.Get(); }
    UINT GetDescriptorSize() const { return descriptorSize; }

GG_ENDCLASS
