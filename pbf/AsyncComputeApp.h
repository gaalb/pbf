#pragma once

#include <Egg/App.h>
#include <Egg/PsoManager.h>
#include <Egg/d3dx12.h>
#include "Fence.h"

// AsyncComputeApp: base class for applications that run a graphics queue and a compute queue
// in parallel. Both queues are managed coequally: same structure, same sync discipline.
//
// Graphics queue: commandQueue (inherited from App), commandAllocator, commandList.
//   Tracked by graphicsFrame (increments each time the graphics queue is signaled).
// Compute queue:  computeCommandQueue, computeAllocator, computeList.
//   Tracked by computeFrame (increments each time the compute queue is signaled).
//
// Sync helpers:
//   cpuWaitForGraphics(N)    -- CPU blocks until graphics queue completes frame N
//   cpuWaitForCompute(N)     -- CPU blocks until compute queue completes frame N
//   graphicsWaitForCompute(N)-- graphics queue GPU-stalls until compute fence reaches frame N
//
// The graphics queue is created externally (main.cpp) and injected via SetCommandQueue().
// The compute queue is created here in CreateResources().
class AsyncComputeApp : public Egg::App {
protected:
	// --- Graphics queue ---
	com_ptr<ID3D12CommandAllocator>     commandAllocator;
	com_ptr<ID3D12GraphicsCommandList>  commandList;
	com_ptr<ID3D12Resource>             depthStencilBuffer;
	com_ptr<ID3D12DescriptorHeap>       dsvHeap;
	Egg::PsoManager::P                  psoManager;
	Fence                               graphicsFence;
	uint64_t                            graphicsFrame = 0;

	// --- Compute queue ---
	com_ptr<ID3D12CommandQueue>         computeCommandQueue;
	com_ptr<ID3D12CommandAllocator>     computeAllocator;
	com_ptr<ID3D12GraphicsCommandList>  computeList;
	Fence                               computeFence;
	uint64_t                            computeFrame = 0;

	// CPU blocks until graphicsFence reaches frame N.
	// Also updates swapChainBackBufferIndex (like the old WaitForPreviousFrame).
	void cpuWaitForGraphics(uint64_t frame) {
		graphicsFence.cpuWait(frame);
		previousSwapChainBackBufferIndex = swapChainBackBufferIndex;
		swapChainBackBufferIndex = swapChain->GetCurrentBackBufferIndex();
	}

	// CPU blocks until computeFence reaches frame N.
	void cpuWaitForCompute(uint64_t frame) {
		computeFence.cpuWait(frame);
	}

	// Graphics queue GPU-stalls until computeFence reaches frame N (CPU is not blocked).
	void graphicsWaitForCompute(uint64_t frame) {
		computeFence.gpuWait(commandQueue, frame);
	}

	// Optional hook for subclasses that want a single-list graphics recording pattern.
	// Not pure virtual: applications that override Render() directly can leave this empty.
	virtual void PopulateCommandList() {}

public:
	virtual void CreateResources() override {
		// App::CreateResources() is intentionally not called: its fence is not used here.
		// We create our own fences for each queue below.

		psoManager = Egg::PsoManager::Create(device);

		// Graphics command list (DIRECT type: can record draws, dispatches, and copies)
		DX_API("Failed to create graphics command allocator")
			device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
				IID_PPV_ARGS(commandAllocator.GetAddressOf()));
		DX_API("Failed to create graphics command list")
			device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
				commandAllocator.Get(), nullptr, IID_PPV_ARGS(commandList.GetAddressOf()));
		commandList->Close();

		// Compute queue (COMPUTE type: dispatches and copies only, dedicated hardware queue on most GPUs)
		D3D12_COMMAND_QUEUE_DESC computeQueueDesc = {};
		computeQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
		computeQueueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
		computeQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
		computeQueueDesc.NodeMask = 0;
		DX_API("Failed to create compute command queue")
			device->CreateCommandQueue(&computeQueueDesc, IID_PPV_ARGS(computeCommandQueue.GetAddressOf()));
		DX_API("Failed to create compute command allocator")
			device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
				IID_PPV_ARGS(computeAllocator.GetAddressOf()));
		DX_API("Failed to create compute command list")
			device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
				computeAllocator.Get(), nullptr, IID_PPV_ARGS(computeList.GetAddressOf()));
		computeList->Close();

		// One fence per queue: explicit frame values on both signal and wait sides
		graphicsFence.createResources(device);
		computeFence.createResources(device);
	}

	virtual void CreateSwapChainResources() override {
		Egg::App::CreateSwapChainResources(); // sets up viewPort, scissorRect, RTVs

		D3D12_DESCRIPTOR_HEAP_DESC dsHeapDesc = {};
		dsHeapDesc.NumDescriptors = 1;
		dsHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
		dsHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
		DX_API("Failed to create depth stencil descriptor heap")
			device->CreateDescriptorHeap(&dsHeapDesc, IID_PPV_ARGS(dsvHeap.GetAddressOf()));

		D3D12_CLEAR_VALUE depthOptimizedClearValue = {};
		depthOptimizedClearValue.Format = DXGI_FORMAT_D32_FLOAT;
		depthOptimizedClearValue.DepthStencil.Depth = 1.0f;
		depthOptimizedClearValue.DepthStencil.Stencil = 0;

		DX_API("Failed to create depth stencil buffer")
			device->CreateCommittedResource(
				&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
				D3D12_HEAP_FLAG_NONE,
				&CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_D32_FLOAT,
					scissorRect.right, scissorRect.bottom, 1, 0, 1, 0,
					D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
				D3D12_RESOURCE_STATE_DEPTH_WRITE,
				&depthOptimizedClearValue,
				IID_PPV_ARGS(depthStencilBuffer.GetAddressOf()));

		depthStencilBuffer->SetName(L"Depth Stencil Buffer");

		D3D12_DEPTH_STENCIL_VIEW_DESC depthStencilDesc = {};
		depthStencilDesc.Format = DXGI_FORMAT_D32_FLOAT;
		depthStencilDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
		depthStencilDesc.Flags = D3D12_DSV_FLAG_NONE;
		device->CreateDepthStencilView(depthStencilBuffer.Get(), &depthStencilDesc,
			dsvHeap->GetCPUDescriptorHandleForHeapStart());
	}

	virtual void ReleaseSwapChainResources() override {
		depthStencilBuffer.Reset();
		dsvHeap.Reset();
		Egg::App::ReleaseSwapChainResources();
	}

	virtual void ReleaseResources() override {
		psoManager = nullptr;
		computeList.Reset();
		computeAllocator.Reset();
		computeCommandQueue.Reset();
		commandList.Reset();
		commandAllocator.Reset();
		Egg::App::ReleaseResources();
	}

	virtual void Resize(int width, int height) override {
		// Drain both queues before resizing swap chain buffers and depth buffer
		cpuWaitForCompute(computeFrame);
		cpuWaitForGraphics(graphicsFrame);
		Egg::App::Resize(width, height);
	}

	virtual void Destroy() override {
		// Drain both queues before releasing any resources
		cpuWaitForCompute(computeFrame);
		cpuWaitForGraphics(graphicsFrame);
		// Fences cleaned up automatically by Fence's RAII event handle
		ReleaseResources();
		Egg::App::Destroy();
	}
};
