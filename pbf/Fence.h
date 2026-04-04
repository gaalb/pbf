#pragma once
#include "Egg/Common.h"

class Fence {
	com_ptr<ID3D12Fence> fence;
	Microsoft::WRL::Wrappers::Event event; // Win32 event object for CPU waits
public:
	void createResources(com_ptr<ID3D12Device> device, D3D12_FENCE_FLAGS flags = D3D12_FENCE_FLAG_NONE) {
		DX_API("Fence")
			device->CreateFence(0, flags, IID_PPV_ARGS(&fence));

		// create win32 event: auto-reset, initially unset, Attach transfers ownership
		event.Attach(CreateEventEx(nullptr, FALSE, FALSE, EVENT_ALL_ACCESS));
	}

	// note: asynchronous call from CPU POV
	void signal(com_ptr<ID3D12CommandQueue> queue, uint64_t value) {
		queue->Signal(fence.Get(), value); // place a signal command in the queue: fence reaches value when queue reaches this point
	}

	// note: blocking call - CPU waits until fence reaches value
	void cpuWait(uint64_t value) {
		if (fence->GetCompletedValue() < value) {
			DX_API("Waiting for fence")
				fence->SetEventOnCompletion(value, event.Get());
			WaitForSingleObject(event.Get(), INFINITE);
		}
	}

	// stalls the GPU queue when it reaches this command, until the fence reaches value (CPU is not stalled)
	void gpuWait(com_ptr<ID3D12CommandQueue> queue, uint64_t value) {
		queue->Wait(fence.Get(), value);
	}

	// Returns the underlying fence pointer, for use where the raw interface is needed.
	ID3D12Fence* get() {
		return fence.Get();
	}
};
