#pragma once

#include <Egg/SimpleApp.h>
#include "ConstantBufferTypes.h"
#include <Egg/Cam/FirstPerson.h>
#include <Egg/ConstantBuffer.hpp>


// SimpleApp gives us:
// command allocator and command list for recording GPU commands
// depth stencil buffer 
// PSO manager
// frame synchronization (WaitForPreviousFrame)
// basic render that populates command list, executes it, presents and syncs
class PbfApp : public Egg::SimpleApp {
protected:
	Egg::Cam::FirstPerson::P camera; // WASD + mouse movement camera
	Egg::ConstantBuffer<PerFrameCb> perFrameCb; // constant buffer uploaded to GPU each frame

	Egg::Mesh::Shaded::P particleMesh; // combines material + geometry + PSO into one drawable

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

		// draw commands here
		particleMesh->Draw(commandList.Get()); // 

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
		camera->Animate(dt); // update camera position and orientation based on user input
		perFrameCb->viewProjMat = // calculate the combined view-projection matrix and store it in the constant buffer
			camera->GetViewMatrix() * // view matrix: world space -> camera space
			camera->GetProjMatrix(); // projection matrix: camera space -> clip space
		perFrameCb->cameraPos = Egg::Math::Float4(camera->GetEyePosition(), 1.0f);
		perFrameCb.Upload(); // memcpy the data to the GPU-visible constant buffer
	}

public:
	virtual void CreateResources() override {
		Egg::SimpleApp::CreateResources(); // ccreates command allocator, command list, PSO manager, and sync objects (fence)
		perFrameCb.CreateResources(device.Get()); // Create the constant buffer on the GPU (upload heap, so CPU can write to it every frame)
		camera = Egg::Cam::FirstPerson::Create();
	}

	virtual void LoadAssets() override {
		// generate test particle positions:  create a small cube of particles so we can see something on screen
		std::vector<Egg::Math::Float3> positions;
		int gridSize = 10; // 10x10x10 = 1000 particles
		float spacing = 0.2f;  // distance between particles
		float offset = -(gridSize * spacing) / 2.0f; // center the cube around the origin

		for (int x = 0; x < gridSize; x++) {
			for (int y = 0; y < gridSize; y++) {
				for (int z = 0; z < gridSize; z++) {
					positions.push_back(Egg::Math::Float3(
						offset + x * spacing,
						offset + y * spacing,
						offset + z * spacing
					));
				}
			}
		}

		// loadCso reads the pre-compiled .cso binary into a blob
		com_ptr<ID3DBlob> vertexShader = Egg::Shader::LoadCso("Shaders/particleVS.cso"); // vertex shader
		com_ptr<ID3DBlob> pixelShader = Egg::Shader::LoadCso("Shaders/particlePS.cso"); // pixel shader

		// extract the root signature from the vertex shader
		// the [RootSignature(...)] attribute we defined in the HLSL gets embedded in the compiled blob
		com_ptr<ID3D12RootSignature> rootSig = Egg::Shader::LoadRootSignature(device.Get(), vertexShader.Get());

		// create a material to hold shaders, root signature, blend/rasterizer/depth state
		Egg::Mesh::Material::P material = Egg::Mesh::Material::Create();
		material->SetRootSignature(rootSig); 
		material->SetVertexShader(vertexShader);
		material->SetPixelShader(pixelShader);
		// enable depth testing so particles occlude each other correctly
		material->SetDepthStencilState(CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT));
		material->SetDSVFormat(DXGI_FORMAT_D32_FLOAT);
		// bind the per-frame constant buffer so the shader can access camera matrices
		material->SetConstantBuffer(perFrameCb);		

		// Upload the positions to a GPU vertex buffer
		Egg::Mesh::Geometry::P geometry = Egg::Mesh::VertexStreamGeometry::Create( // vertex stream geometry: array of vertices
			device.Get(), // D3D device, used to create GPU resources
			positions.data(), // pointer to vertex data
			(unsigned int)(positions.size() * sizeof(Egg::Math::Float3)), // total size in bytes
			(unsigned int)sizeof(Egg::Math::Float3)); // stride: bytes per vertex

		// Tell the input layout that each vertex has a POSITION with 3 floats
		geometry->AddInputElement({
			"POSITION", // semantic name — must match the HLSL input
			0, // semantic index
			DXGI_FORMAT_R32G32B32_FLOAT, // format: 3x 32-bit floats
			0, // input slot
			0,// byte offset within the vertex (position is first, so 0)
			D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, // this data is per-vertex, not per-instance
			0 // instance step rate (0 for per-vertex data)
			});

		// set topology to point list — each vertex is drawn as a single point
		geometry->SetTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);

		// mesh = material + geometry + PSO (created by PSO manager based on the material's root signature, shaders, and states)
		particleMesh = Egg::Mesh::Shaded::Create(psoManager, material, geometry);
	}

	// When the window is resized, update the camera's aspect ratio
	virtual void CreateSwapChainResources() override {
		Egg::SimpleApp::CreateSwapChainResources();
		if (camera) {
			camera->SetAspect(aspectRatio);
		}
	}

	// Forward window messages (keyboard, mouse) to the camera
	virtual void ProcessMessage(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) override {
		camera->ProcessMessage(hWnd, uMsg, wParam, lParam);
	}
};