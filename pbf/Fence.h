#pragma once
#include "Egg/Common.h"

class Fence {
	com_ptr<ID3D12Fence> fence; // the underlying D3D fence
	Microsoft::WRL::Wrappers::Event event; // Win32 event object for CPU waits
public:
	void createResources(com_ptr<ID3D12Device> device, D3D12_FENCE_FLAGS flags = D3D12_FENCE_FLAG_NONE) {
		DX_API("Fence")
			device->CreateFence(0, flags, IID_PPV_ARGS(&fence));

		// create win32 event: auto-reset, initially unset, Attach transfers ownership
		event.Attach(CreateEventEx(nullptr, FALSE, FALSE, EVENT_ALL_ACCESS));
	}

	// asynchronous call from the CPU's POV: places a signal in the queue,
	// when the GPU reaches this point, it signals the fence
	void signal(com_ptr<ID3D12CommandQueue> queue, uint64_t value) {
		DX_API("Failed to signal fence")
			queue->Signal(fence.Get(), value);
	}

	// blocking call from the CPU POV- CPU waits until fence reaches value
	void cpuWait(uint64_t value) {
		if (fence->GetCompletedValue() < value) { // fence not reached yet
			DX_API("Waiting for fence") // D3D signals the win32 event when reached
				fence->SetEventOnCompletion(value, event.Get());
			WaitForSingleObject(event.Get(), INFINITE); // and then we wait for that signal: BLOCK
		} // fast path: else branch, missing -> no-op
	}

	// stalls the GPU queue when it reaches this command, until the fence reaches value (CPU is not stalled)
	void gpuWait(com_ptr<ID3D12CommandQueue> queue, uint64_t value) {
		queue->Wait(fence.Get(), value);
	}

	// Returns the underlying fence pointer, for use where the raw interface is needed.
	ID3D12Fence* get() {
		return fence.Get();
	}

	// Returns the most recently completed fence value (non-blocking CPU query).
	uint64_t getCompletedValue() {
		return fence->GetCompletedValue();
	}

	void destroy() {
		event.Close();
	}
};
