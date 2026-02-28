#pragma once

#include <Egg/SimpleApp.h>

// SimpleApp gives us:
// command allocator and command list for recording GPU commands
// depth stencil buffer 
// PSO manager
// frame synchronization (WaitForPreviousFrame)
// basic render that populates command list, executes it, presents and syncs
class PbfApp : public Egg::SimpleApp {
protected:
	// this is the method we must implement from SimpleApp, Render() calls it,
	// this is where we record all GPU commands to construct one frame
	virtual void PopulateCommandList() override {
		// reset the command allocator, freeing the memory used by the previous frame's commands
		// this can only be done after the GPU finished executing those commands 
		commandAllocator->Reset();

		// command list must be reset before we start recording commands into it
		// second param is initial pipeline state, don't need it yet
		commandList->Reset(commandAllocator.Get(), nullptr);

		// tell the GPU what region of the screen to draw to
		commandList->RSSetViewports(1, &viewPort); // the visible area (full window)
		commandList->RSSetScissorRects(1, &scissorRect); // the clipping rectangle (also full window)

		// transition the current back buffer from "present" state to "render target" state so we can draw into it
		commandList->ResourceBarrier(1, // number of barriers
			&CD3DX12_RESOURCE_BARRIER::Transition( // helper function to create a transition barrier
				renderTargets[swapChainBackBufferIndex].Get(), // resource: the current back buffer, identified by the swap chain's current back buffer index
				D3D12_RESOURCE_STATE_PRESENT, // before: the back buffer was last used for presentation
				D3D12_RESOURCE_STATE_RENDER_TARGET)); // after: we want to render into the back buffer

		// get a CPU handle to the current back buffer's render target view (RTV)
		CD3DX12_CPU_DESCRIPTOR_HANDLE rHandle( // helper function to calculate a handle with an offset
			rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart(), // start of the RTV heap
			swapChainBackBufferIndex, // which back buffer
			rtvDescriptorHandleIncrementSize); // byte offset between entries

		// get a CPU handle to the depth stencil view (DSV)
		CD3DX12_CPU_DESCRIPTOR_HANDLE dsvHandle(dsvHeap->GetCPUDescriptorHandleForHeapStart());

		// set the render target and depth buffer as the output for draw calls
		commandList->OMSetRenderTargets(1, &rHandle, FALSE, &dsvHandle);

		// clear the screen to a solid color
		const float clearColor[] = { 0.0f, 0.2f, 0.4f, 1.0f };
		commandList->ClearRenderTargetView(rHandle, clearColor, 0, nullptr);

		// clear the depth buffer to 1.0 (maximum depth = far plane)
		commandList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

		//tTransition the back buffer back to "present" state so the swap chain can display it
		commandList->ResourceBarrier(1, // number of barriers
			&CD3DX12_RESOURCE_BARRIER::Transition( // helper function to create a transition barrier
				renderTargets[swapChainBackBufferIndex].Get(), // resource: the current back buffer, identified by the swap chain's current back buffer index
				D3D12_RESOURCE_STATE_RENDER_TARGET, // before: we just rendered into the back buffer
				D3D12_RESOURCE_STATE_PRESENT)); // after: we want to present the back buffer
	
		// close the command list, no more commands can be recorded until the next Reset()
		DX_API("Failed to close command list")
			commandList->Close();
	}

	virtual void Update(float dt, float T) override {
		// TODO: camera and simulation
	}
};