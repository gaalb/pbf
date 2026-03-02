#pragma once

#include <Egg/SimpleApp.h>
#include "ConstantBufferTypes.h"
#include "ParticleTypes.h"
#include <Egg/Cam/FirstPerson.h>
#include <Egg/ConstantBuffer.hpp>
#include <Egg/TextureCube.h>
#include <Egg/Importer.h>
#include <Egg/Mesh/Prefabs.h>


// SimpleApp gives us:
// command allocator and command list for recording GPU commands
// depth stencil buffer 
// PSO manager
// frame synchronization (WaitForPreviousFrame)
// basic render that populates command list, executes it, presents and syncs
class PbfApp : public Egg::SimpleApp {
private:
	void LoadBackground() {
		// ImportTextureCube reads the dds file, and creates the two GPU resources for the texture cube :
		// 1. the default heap resource, which has 6 array slices for the cube map faces and is GPU-local (fast)
		// 2. the upload heap resource, which the CPU can write to, and is used for staging
		// the texture data is also copied to the upload heap by this function call, but not yet transferred to
		// the default heap until we call UploadResources(), as this step requires the command queue
		envTexture = Egg::Importer::ImportTextureCube(device.Get(), "../Media/cloudyNoon.dds");
		// create a Shader Resource View so the pixel shader can sample the cubemap
		// note that this is different from the SRV heap that we created earlier, this SRV fills one slot
		// in the SRV heap, specifically, slot (index) 0
		envTexture.CreateSRV(device.Get(), srvHeap.Get(), 0);
		// transfer texture data from upload heap to GPU-local memory
		UploadResources();

		// loadCso reads the pre-compiled .cso binary into a blob
		com_ptr<ID3DBlob> bgVertexShader = Egg::Shader::LoadCso("Shaders/bgVS.cso"); // vertex shader
		com_ptr<ID3DBlob> bgPixelShader = Egg::Shader::LoadCso("Shaders/bgPS.cso"); // pixel shader
		com_ptr<ID3D12RootSignature> bgRootSig = Egg::Shader::LoadRootSignature(device.Get(), bgVertexShader.Get());

		Egg::Mesh::Material::P bgMaterial = Egg::Mesh::Material::Create();
		bgMaterial->SetRootSignature(bgRootSig);
		bgMaterial->SetVertexShader(bgVertexShader);
		bgMaterial->SetPixelShader(bgPixelShader);
		// enable depth testing - the background writes z=0.999999, so particles (closer) will draw in front
		bgMaterial->SetDepthStencilState(CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT));
		bgMaterial->SetDSVFormat(DXGI_FORMAT_D32_FLOAT);
		// bind the per-frame constant buffer (root parameter 0)
		bgMaterial->SetConstantBuffer(perFrameCb);
		// bind the SRV heap containing the cubemap (root parameter 1, starting at descriptor index 0)
		bgMaterial->SetSrvHeap(1, srvHeap, 0);

		// The fullscreen quad from Egg's prefab library - 2 triangles covering the entire screen
		Egg::Mesh::Geometry::P bgGeometry = Egg::Mesh::Prefabs::FullScreenQuad(device.Get());

		backgroundMesh = Egg::Mesh::Shaded::Create(psoManager, bgMaterial, bgGeometry);
	}

	std::vector<Particle> GenerateParticles() {
		// create a small cube of particles so we can see something on screen
		std::vector<Particle> particles;
		float spacing = 0.2f; // distance between particles in world space
		float offset = -(gridSize * spacing) / 2.0f; // shift so the cube is centered at the origin

		for (int x = 0; x < gridSize; x++) {
			for (int y = 0; y < gridSize; y++) {
				for (int z = 0; z < gridSize; z++) {
					Particle p;
					p.position = Egg::Math::Float3(
						offset + x * spacing,
						offset + y * spacing,
						offset + z * spacing);
					p.velocity = Egg::Math::Float3(0.0f, 0.0f, 0.0f); // start at rest
					particles.push_back(p);
				}
			}
		}
		return particles;
	}

	void UploadParticles(const std::vector<Particle>& particles) {
		// CPU side
		void* pData; // will point to the upload buffer's CPU-accessible memory after mapping
		CD3DX12_RANGE readRange(0, 0); // 0,0 tells DX12 we will not read from this mapping, only write
		DX_API("Failed to map particle upload buffer")
			particleUploadBuffer->Map(
				0, // subresource index: 0 for buffers (only textures have multiple subresources)
				&readRange,// read range hint: (0,0) means we won't read any data back
				&pData); // output: CPU pointer to the upload buffer memory

		memcpy(pData, particles.data(), particles.size() * sizeof(Particle)); // copy all particle data from CPU vector into upload buffer

		particleUploadBuffer->Unmap( // we only use this buffer to upload data to the GPU, so we can unmap right after the copy
			0, // subresource index
			nullptr); // written range: nullptr means we wrote the entire buffer

		// GPU side
		DX_API("Failed to reset command allocator (UploadParticles)")
			commandAllocator->Reset(); // free memory from previous command list recording
		DX_API("Failed to reset command list (UploadParticles)")
			commandList->Reset(commandAllocator.Get(), nullptr); // begin recording; nullptr = no initial pipeline state needed for copy commands

		// transition particleBuffer from UAV to COPY_DEST:
		// the GPU enforces that a resource's state matches how it's being used.
		// it starts in UNORDERED_ACCESS (UAV), but CopyBufferRegion requires COPY_DEST.
		commandList->ResourceBarrier(1,
			&CD3DX12_RESOURCE_BARRIER::Transition(
				particleBuffer.Get(),
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS, // state before: UAV (how we created it)
				D3D12_RESOURCE_STATE_COPY_DEST)); // state after: ready to receive a copy

		commandList->CopyBufferRegion(
			particleBuffer.Get(), // destination: the default heap buffer
			0,  // destination offset in bytes: start from the beginning
			particleUploadBuffer.Get(), // source: the upload heap buffer we just filled
			0, // source offset in bytes: start from the beginning
			numParticles * sizeof(Particle)); // number of bytes to copy

		// transition particleBuffer back to UNORDERED_ACCESS so compute shaders can use it
		commandList->ResourceBarrier(1,
			&CD3DX12_RESOURCE_BARRIER::Transition(
				particleBuffer.Get(),
				D3D12_RESOURCE_STATE_COPY_DEST, // state before: we just copied into it
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS)); // state after: ready for compute shader read/write

		DX_API("Failed to close command list (UploadParticles)")
			commandList->Close(); // finalize the command list, no more commands can be added

		ID3D12CommandList* commandLists[] = { commandList.Get() };
		commandQueue->ExecuteCommandLists(_countof(commandLists), commandLists); // submit the copy commands to the GPU

		WaitForPreviousFrame(); // block CPU until GPU has finished the copy before we proceed
	}

	void LoadParticles() {
		std::vector<Particle> particles = GenerateParticles(); // generate initial particle positions and zero velocities
		UploadParticles(particles); // copy particle data from CPU to the GPU default heap buffer
		// loadCso reads the pre-compiled .cso binary into a blob
		com_ptr<ID3DBlob> vertexShader = Egg::Shader::LoadCso("Shaders/particleVS.cso");
		com_ptr<ID3DBlob> geometryShader = Egg::Shader::LoadCso("Shaders/particleGS.cso");
		com_ptr<ID3DBlob> pixelShader = Egg::Shader::LoadCso("Shaders/particlePS.cso");
		// extract the root signature from the vertex shader
		// the [RootSignature(...)] attribute we defined in the HLSL gets embedded in the compiled blob
		com_ptr<ID3D12RootSignature> rootSig = Egg::Shader::LoadRootSignature(device.Get(), vertexShader.Get());

		// create a material to hold shaders, root signature, blend/rasterizer/depth state
		Egg::Mesh::Material::P material = Egg::Mesh::Material::Create();
		material->SetRootSignature(rootSig);
		material->SetVertexShader(vertexShader);
		material->SetGeometryShader(geometryShader); // expand points into quads on the GPU
		material->SetPixelShader(pixelShader);
		// enable depth testing so particles occlude each other correctly
		material->SetDepthStencilState(CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT));
		material->SetDSVFormat(DXGI_FORMAT_D32_FLOAT);
		// bind the per-frame constant buffer so the shader can access camera matrices
		material->SetConstantBuffer(perFrameCb);

		// TODO: this VertexStreamGeometry will be removed in a later step when the VS reads from the structured buffer instead
		// temporarily using particles.data() so this compiles; stride is sizeof(Particle) since position is the first field
		Egg::Mesh::Geometry::P geometry = Egg::Mesh::VertexStreamGeometry::Create(
			device.Get(),
			particles.data(),
			(unsigned int)(particles.size() * sizeof(Particle)),
			(unsigned int)sizeof(Particle));

		// Tell the input layout that each vertex has a POSITION with 3 floats
		geometry->AddInputElement({
			"POSITION", // semantic name - must match the HLSL input
			0, // semantic index
			DXGI_FORMAT_R32G32B32_FLOAT, // format: 3x 32-bit floats
			0, // input slot
			0,// byte offset within the vertex (position is first, so 0)
			D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, // this data is per-vertex, not per-instance
			0 // instance step rate (0 for per-vertex data)
			});

		// set topology to point list - each vertex is drawn as a single point
		geometry->SetTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);

		// mesh = material + geometry + PSO (created by PSO manager based on the material's root signature, shaders, and states)
		particleMesh = Egg::Mesh::Shaded::Create(psoManager, material, geometry);
	}

protected:
	static const int gridSize = 10; // number of particles along one edge of the cube
	static const int numParticles = gridSize * gridSize * gridSize; // 1000

	Egg::Cam::FirstPerson::P camera; // WASD + mouse movement camera
	Egg::ConstantBuffer<PerFrameCb> perFrameCb; // constant buffer uploaded to GPU each frame
	Egg::Mesh::Shaded::P particleMesh; // combines material + geometry + PSO into one drawable
	Egg::Mesh::Shaded::P backgroundMesh; // fullscreen quad + cubemap shader
	Egg::TextureCube envTexture; // the cubemap texture
	com_ptr<ID3D12DescriptorHeap> srvHeap; // descriptor heap for shader-visible textures

	com_ptr<ID3D12Resource> particleBuffer; // default heap: GPU-local, UAV-accessible, lives for the duration of the app
	com_ptr<ID3D12Resource> particleUploadBuffer; // upload heap: used once to transfer initial particle data to the GPU

	// uploads textures from CPU to GPU. Must be called after importing textures.
	// similar to a render call: resets command list, records copy commands, executes, waits.
	void UploadResources() {
		// free the memory used by the previous frame's commands, must be done after GPU finished executing those commands
		DX_API("Failed to reset command allocator (UploadResources)")
			commandAllocator->Reset(); 
		
		// reset command list to start recording copy commands, no initial pipeline state needed for copy commands
		DX_API("Failed to reset command list (UploadResources)")
			commandList->Reset(commandAllocator.Get(), nullptr); 

		// record the copy commands that transfer texture data from upload heap to default heap
		envTexture.UploadResource(commandList.Get());

		DX_API("Failed to close command list (UploadResources)")
			commandList->Close();

		// execute the copy commands
		ID3D12CommandList* commandLists[] = { commandList.Get() };
		commandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);

		// wait for the GPU to finish copying before we release the upload buffers
		WaitForPreviousFrame();

		// the upload heap copies are done - we can free the temporary upload resources
		envTexture.ReleaseUploadResources();
	}

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
		backgroundMesh->Draw(commandList.Get()); // draw skybox at the back first
		particleMesh->Draw(commandList.Get()); // draw particles on top

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

		perFrameCb->viewProjTransform = // calculate the combined view-projection matrix and store it in the constant buffer
			camera->GetViewMatrix() * // view matrix: world space -> camera space
			camera->GetProjMatrix(); // projection matrix: camera space -> clip space
		perFrameCb->rayDirTransform = camera->GetRayDirMatrix(); // clip-space coords -> world-space view direction
		perFrameCb->cameraPos = Egg::Math::Float4(camera->GetEyePosition(), 1.0f);
		perFrameCb->lightDir = Egg::Math::Float4(0.5f, 1.0f, 0.3f, 0.0f); // light pointing down-left
		perFrameCb->particleParams = Egg::Math::Float4(0.9f, 0.1f, 0.7f, 0.08f); // x: particle radius 

		perFrameCb.Upload(); // memcpy the data to the GPU-visible constant buffer
	}

public:
	virtual void CreateResources() override {
		Egg::SimpleApp::CreateResources(); // creates command allocator, command list, PSO manager, and sync objects (fence)
		perFrameCb.CreateResources(device.Get()); // create the constant buffer on the GPU (upload heap, so CPU can write to it every frame)
		camera = Egg::Cam::FirstPerson::Create(); // create the camera, which will handle user input and calculate view/projection matrices

		// The texture cube lives in gpu memory, and will be created by ImportTextureCube as a committed resource
		// on the DEFAULT heap. We access it via a Shader Resource View (SRV), which is a type of descriptor. Descriptors
		// live in a descriptor heap, which has to be big enough to hold a descriptor for each of our resources, one of
		// which is the cube map texture.
		D3D12_DESCRIPTOR_HEAP_DESC dhd;
		dhd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE; // GPU can see these descriptors
		dhd.NodeMask = 0; // single GPU setup, so no node masking needed
		dhd.NumDescriptors = 1; // we've only got 1 texture for now
		dhd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; // constant buffer / shader resource / unordered access views

		DX_API("Failed to create SRV descriptor heap")
			device->CreateDescriptorHeap(&dhd, IID_PPV_ARGS(srvHeap.GetAddressOf())); // pointer to heap in srvHeap

		// create the particle buffer on the default heap (GPU-local, writable by compute shaders via UAV)
		const UINT64 bufferSize = numParticles * sizeof(Particle); // total size in bytes: 1000 particles * 24 bytes each
		const CD3DX12_HEAP_PROPERTIES defaultHeapProps(D3D12_HEAP_TYPE_DEFAULT); // DEFAULT heap: GPU-local fast memory, not CPU-accessible
		const CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(
			bufferSize,  // size of the buffer in bytes
			D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS); // UAV flag: required for compute shaders to write to this buffer
		DX_API("Failed to create particle buffer")
			device->CreateCommittedResource( // allocation call
				&defaultHeapProps, // allocate on the DEFAULT (GPU-local) heap
				D3D12_HEAP_FLAG_NONE, // no special heap flags
				&bufferDesc, 
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS, // initial state: UAV, matching how compute shaders will access it
				nullptr, // optimized clear value: only used for render targets / depth buffers, not buffers
				IID_PPV_ARGS(particleBuffer.ReleaseAndGetAddressOf())); // ReleaseAndGetAddressOf releases any existing object first
		particleBuffer->SetName(L"Particle Buffer"); // debug name for D3D12 validation layer

		// create the upload buffer: same size, upload heap so the CPU can write initial particle data into it
		// once the data is copied to the default heap, this buffer is no longer needed
		const CD3DX12_HEAP_PROPERTIES uploadHeapProps(D3D12_HEAP_TYPE_UPLOAD); // UPLOAD heap: CPU-writable, GPU-readable, slower than default heap
		const CD3DX12_RESOURCE_DESC uploadBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize); // same size, no UAV flag (CPU staging only)
		DX_API("Failed to create particle upload buffer")
			device->CreateCommittedResource(
				&uploadHeapProps, // allocate on the UPLOAD heap
				D3D12_HEAP_FLAG_NONE, // no special heap flags
				&uploadBufferDesc,  // buffer shape: size, no special flags
				D3D12_RESOURCE_STATE_GENERIC_READ, // upload heap resources must start in GENERIC_READ - the only valid state for upload heaps, means GPU can read from it
				nullptr, // no optimized clear value
				IID_PPV_ARGS(particleUploadBuffer.ReleaseAndGetAddressOf())); // ReleaseAndGetAddressOf releases any existing object first
		particleUploadBuffer->SetName(L"Particle Upload Buffer"); // debug name for D3D12 validation layer
	}

	virtual void LoadAssets() override {
		LoadBackground();
		LoadParticles();
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