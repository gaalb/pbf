#pragma once

#include <algorithm>
#include <Egg/SimpleApp.h>
#include "ConstantBufferTypes.h"
#include "ParticleTypes.h"
#include <Egg/Cam/FirstPerson.h>
#include <Egg/ConstantBuffer.hpp>
#include <Egg/TextureCube.h>
#include <Egg/Importer.h>
#include <Egg/Mesh/Prefabs.h>

using namespace Egg::Math;

// SimpleApp gives us:
// command allocator and command list for recording GPU commands
// depth stencil buffer 
// PSO manager
// frame synchronization (WaitForPreviousFrame)
// basic render that populates command list, executes it, presents and syncs
class PbfApp : public Egg::SimpleApp {
protected:
	// constants to tweak for different simulation setups
	// these values are based on the Macklin + Muller 2013 article
	const int gridX = 12, gridY = 20, gridZ = 12; // number of particles along each axis of the initial grid
	const int offsetX = 0, offsetY = 8, offsetZ = 0; // world space offset of the center of the initial particle grid
	const int numParticles = gridX * gridY * gridZ; // total number of particles in the simulation
	const int solverIterations = 4; // how many newton steps to take per frame
	// starting particle distance, which we also treat as the desired particle distance, to compute rest 
	// density and particle display size
	const float particleSpacing = 0.25f; 
	const float h = particleSpacing * 3.5f; // SPH smoothing radius, the larger the more neighbors each particle has
	// if the particles are spaced "d" apart, then one d sided cube contains one particle, meaning that
	// each particle is responsible for d^3 volume of fluid, meaning that with m=1, the density is 1/d^3
	const float rho0 = 1.0f / powf(particleSpacing, 3.0f); 
	// constraint force mixing relaxation parameter (Smith 2006), higher value = softer constraints
	const float epsilon = 5.0f; 
	// XSPH viscosity coefficient (Schechter and Bridson 2012), higher value = "thicker" fluid
	const float viscosity = 0.01f; // paper suggests 0.01
	// artificial purely repulsive pressure term (Monaghan 2000), reduces clumping while leaving room for surface tension
	const float sCorrK = 0.1f; // artificial pressure magnitude coefficient (paper: 0.1)
	const float sCorrDeltaQ = 0.2f * h; // reference distance for artificial pressure (paper: 0.1...0.3 * h)
	const float sCorrN = 4.0f; // exponent for artificial pressure (paper: 4)
	const Float4 particleParams = Float4(0.9f, 0.1f, 0.7f, 0.5*particleSpacing); // xyz are color, w is radius
	const float boxMoveSpeed = 4.0f; // world units per second for arrow key box translation

	// non-constant members
	Float3 boxMin = Float3(-2.0f, -2.0f, -2.0f); // simulation boundary minimum corner (world space)
	Float3 boxMax = Float3(2.0f, 100.0f, 2.0f); // simulation boundary maximum corner (world space)
	Egg::Cam::FirstPerson::P camera; // WASD + mouse movement camera

	Egg::ConstantBuffer<PerFrameCb> perFrameCb; // constant buffer uploaded to GPU each frame -> graphics data
	Egg::Mesh::Shaded::P particleMesh; // combines material + geometry + PSO into one drawable
	Egg::Mesh::Shaded::P backgroundMesh; // fullscreen quad + cubemap shader
	Egg::TextureCube envTexture; // the cubemap texture for the skybox background
	com_ptr<ID3D12DescriptorHeap> srvHeap; // descriptor heap for shader-visible resources
	com_ptr<ID3D12Resource> particleBuffer; // default heap: GPU-local, UAV-accessible
	com_ptr<ID3D12Resource> particleUploadBuffer; // upload heap: used once to transfer initial particle data to the GPU
	Egg::ConstantBuffer<ComputeCb> computeCb; // constant buffer uploaded to GPU each frame -> compute data
	// all four compute shaders share the same root signature layout: CBV(b0) + DescriptorTable(UAV(u0))
	// so we extract the root sig from one shader and reuse it for all four PSOs
	com_ptr<ID3D12RootSignature> computeRootSig;
	com_ptr<ID3D12PipelineState> predictPso; // apply gravity, compute position prediction p*
	com_ptr<ID3D12PipelineState> lambdaPso; // compute lambda per particle
	com_ptr<ID3D12PipelineState> deltaPso; // compute delta_p, update p*, clamp to bounding box
	com_ptr<ID3D12PipelineState> finalizePso; // commit p* to position, update velocity
	com_ptr<ID3D12PipelineState> viscosityPso; // XSPH velocity smoothing

	bool physicsRunning = false; // toggled by spacebar: when false, compute passes are skipped each frame
	bool arrowLeft = false, arrowRight = false, arrowUp = false, arrowDown = false; // arrow key held state for box translation


	void LoadBackground() {
		// ImportTextureCube reads the dds file, and creates the two GPU resources for the texture cube :
		// 1. the default heap resource, which has 6 array slices for the cube map faces and is GPU-local (fast)
		// 2. the upload heap resource, which the CPU can write to, and is used for staging
		// the texture data is also copied to the upload heap by this function call, but not yet transferred to
		// the default heap until we call UploadBackground(), as this step requires the command queue
		envTexture = Egg::Importer::ImportTextureCube(device.Get(), "../Media/cloudyNoon.dds");
		// create a Shader Resource View so the pixel shader can sample the cubemap
		// note that this is different from the SRV *heap* that we created earlier, this SRV fills one slot
		// in the SRV heap, specifically, slot (index) 0
		envTexture.CreateSRV(device.Get(), srvHeap.Get(), 0);
		// transfer texture data from upload heap to GPU-local memory
		UploadBackground();

		// loadCso reads the pre-compiled .cso binary into a blob
		com_ptr<ID3DBlob> bgVertexShader = Egg::Shader::LoadCso("Shaders/bgVS.cso"); // vertex shader
		com_ptr<ID3DBlob> bgPixelShader = Egg::Shader::LoadCso("Shaders/bgPS.cso"); // pixel shader
		com_ptr<ID3D12RootSignature> bgRootSig = Egg::Shader::LoadRootSignature(device.Get(), bgVertexShader.Get());

		// the material of a mesh is what handles shader configuration, which includes root signature, 
		// shader bytecode, pipeline state settings, and resource bindings (SRV/UAV/CBV)
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
		// the geometry of a mesh is what handles raw vertex data
		Egg::Mesh::Geometry::P bgGeometry = Egg::Mesh::Prefabs::FullScreenQuad(device.Get());

		// Mesh = material + geometry
		backgroundMesh = Egg::Mesh::Shaded::Create(psoManager, bgMaterial, bgGeometry);
	}

	std::vector<Particle> GenerateParticles() {
		// create and return an evenly spaced grid of particles so we can see something on screen
		std::vector<Particle> particles; 
		Float3 grid = Float3(gridX, gridY, gridZ);
		Float3 offset = -(grid * particleSpacing) / 2.0f; // shift so the cube is centered at the origin
		offset += Float3(offsetX, offsetY, offsetZ); // apply user-defined world space offset
		for (int x = 0; x < grid.x; x++) {
			for (int y = 0; y < grid.y; y++) {
				for (int z = 0; z < grid.z; z++) {
					Particle p;
					p.position = offset + Float3(x, y, z) * particleSpacing;
					p.velocity = Float3(0.0f, 0.0f, 0.0f); // start at rest
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
		DX_API("Failed to map particle upload buffer") // grab a CPU pointer to the memory of the upload buffer resource
			particleUploadBuffer->Map(
				0, // subresource index: 0 for buffers (only textures have multiple subresources)
				&readRange,// read range hint: (0,0) means we won't read any data back
				&pData); // output: CPU pointer to the upload buffer memory

		memcpy(pData, particles.data(), particles.size() * sizeof(Particle)); // copy all particle data from CPU vector into upload buffer

		particleUploadBuffer->Unmap( // we only use this buffer to upload data to the GPU, so we can unmap right after the copy
			0, // subresource index
			nullptr); // written range: nullptr means we wrote the entire buffer
		// after the unmap, pData is no longer valid

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

	void LoadCompute() {
		// load all five compiled compute shader blobs
		com_ptr<ID3DBlob> predictShader    = Egg::Shader::LoadCso("Shaders/predictCS.cso");
		com_ptr<ID3DBlob> lambdaShader     = Egg::Shader::LoadCso("Shaders/lambdaCS.cso");
		com_ptr<ID3DBlob> deltaShader      = Egg::Shader::LoadCso("Shaders/deltaCS.cso");
		com_ptr<ID3DBlob> finalizeShader   = Egg::Shader::LoadCso("Shaders/finalizeCS.cso");
		com_ptr<ID3DBlob> viscosityShader  = Egg::Shader::LoadCso("Shaders/viscosityCS.cso");

		// all five shaders embed the same root signature string, so we extract it once
		// from predictCS and reuse it for all five PSO descriptors
		computeRootSig = Egg::Shader::LoadRootSignature(device.Get(), predictShader.Get());

		// compute PSO descriptor: much simpler than a graphics PSO --
		// no rasterizer, blend state, render targets, or input layout; just root sig + shader bytecode
		D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
		psoDesc.pRootSignature = computeRootSig.Get();

		psoDesc.CS = CD3DX12_SHADER_BYTECODE(predictShader.Get()); // initialize descriptor with cd3dx12 helper
		DX_API("Failed to create predict compute PSO") // create pso using the descriptor and store it in a com_ptr
			device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(predictPso.GetAddressOf()));

		psoDesc.CS = CD3DX12_SHADER_BYTECODE(lambdaShader.Get());
		DX_API("Failed to create lambda compute PSO")
			device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(lambdaPso.GetAddressOf()));

		psoDesc.CS = CD3DX12_SHADER_BYTECODE(deltaShader.Get());
		DX_API("Failed to create delta compute PSO")
			device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(deltaPso.GetAddressOf()));

		psoDesc.CS = CD3DX12_SHADER_BYTECODE(finalizeShader.Get());
		DX_API("Failed to create finalize compute PSO")
			device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(finalizePso.GetAddressOf()));

		psoDesc.CS = CD3DX12_SHADER_BYTECODE(viscosityShader.Get());
		DX_API("Failed to create viscosity compute PSO")
			device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(viscosityPso.GetAddressOf()));
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
		// bind the particle SRV (slot 1 in srvHeap) to root parameter 1 so the VS can read particle positions
		// root parameter 1 = DescriptorTable(SRV(t0)) as declared in the ParticleRootSig in particleVS.hlsl
		// SetSrvHeap's third argument is a raw byte offset into the heap, not a descriptor slot index
		// so we must multiply the slot index by the descriptor increment size to get the correct byte offset
		UINT descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		material->SetSrvHeap(1, srvHeap, 1 * descriptorSize); // slot 1 * bytes-per-slot = byte offset to the particle SRV

		// NullGeometry: no vertex buffer - the VS fetches positions from the structured buffer using SV_VertexID
		// numParticles tells DrawInstanced how many vertices (and therefore SV_VertexID values) to generate
		Egg::Mesh::Geometry::P geometry = Egg::Mesh::NullGeometry::Create(numParticles);
		geometry->SetTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST); // each SV_VertexID maps to one point, expanded to a quad by the GS

		// mesh = material + geometry + PSO (created by PSO manager based on the material's root signature, shaders, and states)
		particleMesh = Egg::Mesh::Shaded::Create(psoManager, material, geometry);
	}

	void PrepareCommandList() {
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

		// make the SRV heap visible to the GPU for this command list, so shaders can access textures in it
		commandList->SetDescriptorHeaps(1, srvHeap.GetAddressOf()); 
	}

	void CloseCommandList() {
		//transition the back buffer back to "present" state so the swap chain can display it
		commandList->ResourceBarrier(1, // number of barriers
			&CD3DX12_RESOURCE_BARRIER::Transition( // helper function to create a transition barrier
				renderTargets[swapChainBackBufferIndex].Get(), // resource: the current back buffer, identified by the swap chain's current back buffer index
				D3D12_RESOURCE_STATE_RENDER_TARGET, // before: we just rendered into the back buffer
				D3D12_RESOURCE_STATE_PRESENT)); // after: we want to present the back buffer

		// close the command list, no more commands can be recorded until the next Reset()
		DX_API("Failed to close command list")
			commandList->Close();
	}

	void ComputePass() {
		UINT descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		CD3DX12_GPU_DESCRIPTOR_HANDLE uavHandle(
			srvHeap->GetGPUDescriptorHandleForHeapStart(),
			2, // slot index 2 = the particle UAV
			descriptorSize);

		// switch to the compute pipeline: root signature first, then PSO
		// (setting the root sig before the PSO is required - the PSO references the root sig)
		// all four shaders share the same root signature, so we set it once here.
		// root parameter bindings (CBV + UAV) also persist across PSO swaps as long
		// as we don't call SetComputeRootSignature again.
		commandList->SetComputeRootSignature(computeRootSig.Get());
		// bind the compute CB at root parameter 0 via its GPU virtual address directly.
		// this is a root CBV - it doesn't occupy a descriptor heap slot, the address goes straight into the root signature.
		commandList->SetComputeRootConstantBufferView(0, computeCb.GetGPUVirtualAddress());
		commandList->SetComputeRootDescriptorTable(1, uavHandle);

		// ceil(numParticles / 256) groups cover all particles; the shader discards extra threads
		UINT numGroups = (numParticles + 255) / 256;

		// apply gravity, store result in predictedPosition
		commandList->SetPipelineState(predictPso.Get());
		commandList->Dispatch(numGroups, 1, 1);
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(particleBuffer.Get()));

		// constraint solver loop
		for (int iter = 0; iter < solverIterations; iter++) {
			// compute lambda for each particle from predictedPositions
			commandList->SetPipelineState(lambdaPso.Get());
			commandList->Dispatch(numGroups, 1, 1);
			commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(particleBuffer.Get()));

			// compute delta_p from lambdas, update predictedPosition, clamp to box
			commandList->SetPipelineState(deltaPso.Get());
			commandList->Dispatch(numGroups, 1, 1);
			commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(particleBuffer.Get()));
		}

		// commit predictedPosition to position, derive velocity from displacement
		commandList->SetPipelineState(finalizePso.Get());
		commandList->Dispatch(numGroups, 1, 1);
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(particleBuffer.Get()));

		// XSPH viscosity: smooth velocity field toward neighborhood average
		// position reads are race-free here (finalizeCS finished, barrier issued above)
		commandList->SetPipelineState(viscosityPso.Get());
		commandList->Dispatch(numGroups, 1, 1);
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(particleBuffer.Get())); // finalize before graphics pass
	}

	void GraphicsPass() {
		backgroundMesh->Draw(commandList.Get()); // draw skybox at the back first
		particleMesh->Draw(commandList.Get()); // draw particles on top
	}

	// uploads textures from CPU to GPU. Must be called after importing textures.
	// similar to a render call: resets command list, records copy commands, executes, waits.
	void UploadBackground() {
		// free the memory used by the previous frame's commands, must be done after GPU finished executing those commands
		DX_API("Failed to reset command allocator (UploadBackground)")
			commandAllocator->Reset(); 
		
		// reset command list to start recording copy commands, no initial pipeline state needed for copy commands
		DX_API("Failed to reset command list (UploadBackground)")
			commandList->Reset(commandAllocator.Get(), nullptr); 

		// record the copy commands that transfer texture data from upload heap to default heap
		envTexture.UploadResource(commandList.Get());

		DX_API("Failed to close command list (UploadBackground)")
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
		PrepareCommandList(); // set up the command list for a new frame: reset, set viewport/scissor, transition back buffer to render target
		
		if (physicsRunning)
			ComputePass(); // dispatch the compute shader to update particle positions on the GPU

		GraphicsPass(); // draw the background and particles using the updated positions	

		CloseCommandList(); // transition back buffer to present state and close the command list
	}

	void UpdateBoundingBox(float dt) {
		// translate the bounding box on the XZ plane based on held arrow keys
		// left/right move along X, up/down move along Z (both corners shift by the same offset)
		Float3 boxShift = Float3(0.0f, 0.0f, 0.0f);
		if (arrowLeft) boxShift.x -= boxMoveSpeed * dt;
		if (arrowRight) boxShift.x += boxMoveSpeed * dt;
		if (arrowUp) boxShift.z += boxMoveSpeed * dt;
		if (arrowDown) boxShift.z -= boxMoveSpeed * dt;
		boxMin += boxShift;
		boxMax += boxShift;
	}

	void UpdatePerFrameCb() {
		perFrameCb->viewProjTransform = // calculate the combined view-projection matrix and store it in the constant buffer
			camera->GetViewMatrix() * // view matrix: world space -> camera space
			camera->GetProjMatrix(); // projection matrix: camera space -> clip space
		perFrameCb->rayDirTransform = camera->GetRayDirMatrix(); // clip-space coords -> world-space view direction
		perFrameCb->cameraPos = Egg::Math::Float4(camera->GetEyePosition(), 1.0f);
		perFrameCb->lightDir = Egg::Math::Float4(0.5f, 1.0f, 0.3f, 0.0f); // light pointing down-left
		perFrameCb->particleParams = particleParams;

		perFrameCb.Upload(); // memcpy the data to the GPU-visible constant buffer
	}

	void UpdateComputeCb(float dt) {
		computeCb->dt = dt;
		computeCb->numParticles = numParticles;
		computeCb->h = h;
		computeCb->rho0 = rho0;
		computeCb->boxMin = boxMin;
		computeCb->epsilon = epsilon;
		computeCb->boxMax = boxMax;
		computeCb->viscosity = viscosity;
		computeCb->sCorrK = sCorrK;
		computeCb->sCorrDeltaQ = sCorrDeltaQ;
		computeCb->sCorrN = sCorrN;
		computeCb.Upload();
	}

	virtual void Update(float dt, float T) override {
		dt = std::min(dt, 1.0f / 30.0f); // cap at 33ms: prevents energy spikes on window drag or stutter
		camera->Animate(dt); // update camera position and orientation based on user input

		UpdateBoundingBox(dt);

		UpdatePerFrameCb();
		
		UpdateComputeCb(dt);		
	}

	void CreateParticleBuffer() {
		// create the particle buffer on the default heap (GPU-local, writable by compute shaders via UAV)
		const UINT64 bufferSize = numParticles * sizeof(Particle); // total size in bytes
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
	}

	void CreateUploadBuffer() {
		const UINT64 bufferSize = numParticles * sizeof(Particle); // total size in bytes
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

	void CreateSrv() {
		// query how many bytes apart descriptor slots are in this heap type - varies by GPU
		UINT descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

		// particle SRV (slot 1): read-only view for the vertex shader
		D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
		srvDesc.Format = DXGI_FORMAT_UNKNOWN; // structured buffers always use UNKNOWN format (the stride defines the layout)
		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER; // this is a buffer, not a texture
		srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; // mandatory field for structured buffers; standard identity swizzle
		srvDesc.Buffer.FirstElement = 0; // start from the first element in the buffer
		srvDesc.Buffer.NumElements = numParticles; // total number of Particle elements accessible through this view
		srvDesc.Buffer.StructureByteStride = sizeof(Particle); // size of each element in bytes: tells the GPU how to step through the buffer
		srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE; // no special flags (RAW flag would be set here for byte-address buffers)

		CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle(
			srvHeap->GetCPUDescriptorHandleForHeapStart(), // base address of the descriptor heap
			1, // slot index 1
			descriptorSize); // stride between slots in bytes
		device->CreateShaderResourceView(particleBuffer.Get(), &srvDesc, srvHandle); // write the SRV descriptor into slot 1
	}

	void CreateUav() {
		// query how many bytes apart descriptor slots are in this heap type - varies by GPU
		UINT descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		// particle UAV (slot 2): read-write view for the compute shader
		D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
		uavDesc.Format = DXGI_FORMAT_UNKNOWN; // structured buffers always use UNKNOWN format
		uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; // this is a buffer, not a texture
		uavDesc.Buffer.FirstElement = 0; // start from the first element
		uavDesc.Buffer.NumElements = numParticles; // total number of Particle elements accessible through this view
		uavDesc.Buffer.StructureByteStride = sizeof(Particle); // size of each element in bytes
		uavDesc.Buffer.CounterOffsetInBytes = 0; // counter is used for append/consume buffers only - we don't use it
		uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE; // no special flags

		CD3DX12_CPU_DESCRIPTOR_HANDLE uavHandle(
			srvHeap->GetCPUDescriptorHandleForHeapStart(), // base address of the descriptor heap
			2, // slot index 2
			descriptorSize); // stride between slots in bytes
		device->CreateUnorderedAccessView(
			particleBuffer.Get(), // the resource this view points to
			nullptr, // no counter resource (only needed for append/consume buffers)
			&uavDesc,
			uavHandle); // write the UAV descriptor into slot 2
	}

	void CreateDescriptorHeap() {
		// The texture cube lives in gpu memory, and will be created by ImportTextureCube as a committed resource
		// on the DEFAULT heap. We access it via a Shader Resource View (SRV), which is a type of descriptor. Descriptors
		// live in a descriptor heap, which has to be big enough to hold a descriptor for each of our resources, one of
		// which is the cube map texture.
		// descriptor heap layout:
		//   slot 0: cubemap SRV       - sampled by the background pixel shader
		//   slot 1: particle SRV (t0) - read by the vertex shader to fetch particle positions
		//   slot 2: particle UAV (u0) - written by the compute shader to update particle positions and velocities
		D3D12_DESCRIPTOR_HEAP_DESC dhd;
		dhd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE; // GPU can see these descriptors
		dhd.NodeMask = 0; // single GPU setup, so no node masking needed
		dhd.NumDescriptors = 3; // slot 0: cubemap SRV, slot 1: particle SRV, slot 2: particle UAV
		dhd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; // heap type that holds CBVs, SRVs, and UAVs

		DX_API("Failed to create descriptor heap")
			device->CreateDescriptorHeap(&dhd, IID_PPV_ARGS(srvHeap.GetAddressOf()));
	}

public:
	virtual void CreateResources() override {
		Egg::SimpleApp::CreateResources(); // creates command allocator, command list, PSO manager, and sync objects (fence)
		perFrameCb.CreateResources(device.Get()); // create the constant buffer on the GPU (upload heap, so CPU can write to it every frame)
		computeCb.CreateResources(device.Get()); // create the compute constant buffer (upload heap: dt and numParticles written each frame)
		camera = Egg::Cam::FirstPerson::Create(); // create the camera, which will handle user input and calculate view/projection matrices

		CreateDescriptorHeap();

		CreateParticleBuffer();

		CreateUploadBuffer();
		
		CreateSrv();
		
		CreateUav();
	}

	virtual void LoadAssets() override {
		LoadBackground();
		LoadParticles();
		LoadCompute();
	}

	// When the window is resized, update the camera's aspect ratio
	virtual void CreateSwapChainResources() override {
		Egg::SimpleApp::CreateSwapChainResources();
		if (camera) {
			camera->SetAspect(aspectRatio);
		}
	}

	// Forward window messages (keyboard, mouse) to the camera, and handle app-level hotkeys
	virtual void ProcessMessage(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) override {
		camera->ProcessMessage(hWnd, uMsg, wParam, lParam);

		if (uMsg == WM_KEYDOWN && wParam == VK_SPACE)
			physicsRunning = !physicsRunning; // toggle physics simulation on/off

		// track arrow key held state for continuous box translation in Update()
		if (uMsg == WM_KEYDOWN) {
			if (wParam == VK_LEFT) arrowLeft = true;
			if (wParam == VK_RIGHT) arrowRight = true;
			if (wParam == VK_UP) arrowUp = true;
			if (wParam == VK_DOWN) arrowDown = true;
		}
		if (uMsg == WM_KEYUP) {
			if (wParam == VK_LEFT) arrowLeft = false;
			if (wParam == VK_RIGHT) arrowRight = false;
			if (wParam == VK_UP) arrowUp = false;
			if (wParam == VK_DOWN) arrowDown = false;
		}
	}
};