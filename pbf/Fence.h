#pragma once
#include "Egg/Common.h"

class Fence {
	com_ptr<ID3D12Fence> fence; // the underlying D3D fence
	uint64_t valueSignalled; // The last value we have signalled, and can be waited on.
	Microsoft::WRL::Wrappers::Event event; // Win32 event object for CPU waits
public:
	void createResources(com_ptr<ID3D12Device> device, D3D12_FENCE_FLAGS flags = D3D12_FENCE_FLAG_NONE) {
		valueSignalled = 0;
		DX_API("Fence") // 0 initial value, no flags probably, value saved to fence
			device->CreateFence(valueSignalled, flags, IID_PPV_ARGS(&fence));

		// create win32 event: auto-reset, initially unset, Attach transfers ownership
		event.Attach(CreateEventEx(nullptr, FALSE, FALSE, EVENT_ALL_ACCESS));
	}

	// note: asynchronous call from CPU POV
	void signal(com_ptr<ID3D12CommandQueue> queue, uint64_t valueToSignal) {
		queue->Signal(fence.Get(), valueToSignal); // set fence value once queue reaches this
		valueSignalled = valueToSignal;
	}

	// note: blocking call
	void cpuWait() {
		if (fence->GetCompletedValue() < valueSignalled) { // fence not reached yet
			DX_API("Waiting for fence") // else D3D signals the win32 event when reached
				fence->SetEventOnCompletion(valueSignalled, event.Get());
			WaitForSingleObject(event.Get(), INFINITE); // and then we wiat for that signal: BLOCK
		} // fast path: else branch, missing -> no-op
	}

	// stalls gpu when the queue reaches this, but not cpu
	void gpuWait(com_ptr<ID3D12CommandQueue> queue) {
		queue->Wait(fence.Get(), valueSignalled);
	}

	// Returns the underlying fence pointer, for use with commandQueue->Wait(fence, value).
	ID3D12Fence* getFence() {
		return fence.Get();
	}

	// Returns the most recently completed fence value (non-blocking CPU query).
	uint64_t getCompletedValue() {
		return fence->GetCompletedValue();
	}

	void destroy() {
		cpuWait();
		event.Close();
	}
};
