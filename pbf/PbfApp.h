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
#include <imgui.h>
#include <imgui_impl_win32.h>
#include <imgui_impl_dx12.h>

using namespace Egg::Math;

// SimpleApp gives us:
// command allocator and command list for recording GPU commands
// depth stencil buffer 
// PSO manager
// frame synchronization (WaitForPreviousFrame)
// basic render that populates command list, executes it, presents and syncs
class PbfApp : public Egg::SimpleApp {
protected:
	const int gridX = 50, gridY = 100, gridZ = 50; // number of particles along each axis of the initial grid
	const int offsetX = 0, offsetY = 8, offsetZ = 0; // world space offset of the center of the initial particle grid
	const int numParticles = gridX * gridY * gridZ; // total number of particles in the simulation

	// parameters that are tunable via ImGui each frame
	int solverIterations = 4; // how many newton steps to take per frame
	// starting particle distance, which we also treat as the desired particle distance, to compute rest
	// density and particle display size
	float particleSpacing = 0.25f;
	float hMultiplier = 3.5f; // h = particleSpacing * hMultiplier, the larger the more neighbors each particle has
	// constraint force mixing relaxation parameter (Smith 2006), higher value = softer constraints
	float epsilon = 5.0f;
	// XSPH viscosity coefficient (Schechter and Bridson 2012), higher value = "thicker" fluid
	float viscosity = 0.01f; // paper suggests 0.01
	// artificial purely repulsive pressure term (Monaghan 2000), reduces clumping while leaving room for surface tension
	float sCorrK = 0.05f; // artificial pressure magnitude coefficient (paper: 0.1)
	float vorticityEpsilon = 0.01f; // vorticity confinement strength (paper: 0.01)

	// parameters derived from tunable parameters
	float h = particleSpacing * hMultiplier; // SPH smoothing radius
	// if the particles are spaced "d" apart, then one d sided cube contains one particle, meaning that
	// each particle is responsible for d^3 volume of fluid, meaning that with m=1, the density is 1/d^3
	float rho0 = 1.0f / powf(particleSpacing, 3.0f);
	float sCorrDeltaQ = 0.2f * h; // reference distance for artificial pressure (paper: 0.1...0.3 * h)

	// non-tunable constants
	const float sCorrN = 4.0f; // exponent for artificial pressure (paper: 4)
	const Float3 particleColor = Float3(0.9f, 0.1f, 0.7f); // particle display color (RGB)
	const float externalAcceleration = 20.0f; // m/s^2, applied horizontally via arrow keys
	const Float3 boxMin = Float3(-10.0f, -5.0f, -10.0f); // simulation boundary minimum corner (world space)
	const Float3 boxMax = Float3(10.0f, 20.0f, 10.0f); // simulation boundary maximum corner (world space)
	// h is related to the grid size, so we must clampt it by clamping its components to sensible values
	const float hMultiplierMin = 2.0f; 
	const float hMultiplierMax = 4.0f; 
	const float particleSpacingMin = 0.1f;
	const float particleSpacingMax = 0.3f;

	// Spatial grid constants, derived from the parameter bounds above and the simulation box.
	// We reserve GPU buffers for the worst case scenario in terms of how many grid cells we're 
	// going to need. In order to not miss any SPH neighbors, Cell size = h (the SPH kernel radius).
	// since h = particleSpacing * hMultiplier, and both factors are clamped, h ranges from 
	// particleSpacingMin * hMultiplierMin to particleSpacingMax * hMultiplierMax. 
	// The number of grid cells is max when h is the smallests, particleSpacingMin * hMultiplierMin
	// We rebuild the grid every physics run (not every jacobi iteration), this way, we can ensure
	// that the grid is always the same size as h. Since the simulation aims to keep the particle
	// density around rho0, which is derived from the desired particle spacing, we can estimate
	// the number of particles per cell using the relation between particle spacing and h, which
	// is given by hMultiplier. The larger hMultiplier is, the more particles can fit in a cell,
	// with the worst case being hMultiplierMax^3. We add a generous 50% headroom to allow
	// for transient fluctuations in denisty.
	const float minCellSize = particleSpacingMin * hMultiplierMin; // smallest possible h = 0.1 * 2.0 = 0.2
	const UINT maxGridDimX = (UINT)ceilf((boxMax.x - boxMin.x) / minCellSize); 
	const UINT maxGridDimY = (UINT)ceilf((boxMax.y - boxMin.y) / minCellSize); 
	const UINT maxGridDimZ = (UINT)ceilf((boxMax.z - boxMin.z) / minCellSize); 
	const UINT maxNumCells = maxGridDimX * maxGridDimY * maxGridDimZ; 
	const UINT maxPerCell = (UINT)ceilf(hMultiplierMax * hMultiplierMax * hMultiplierMax * 1.5f); 

	// non-constant members
	Float3 externalForce = Float3(0.0f, 0.0f, 0.0f); // current external acceleration from arrow keys
	Egg::Cam::FirstPerson::P camera; // WASD + mouse movement camera
	Egg::ConstantBuffer<PerFrameCb> perFrameCb; // constant buffer uploaded to GPU each frame -> graphics data
	Egg::Mesh::Shaded::P particleMesh; // combines material + geometry + PSO into one drawable
	Egg::Mesh::Shaded::P backgroundMesh; // fullscreen quad + cubemap shader
	Egg::TextureCube envTexture; // the cubemap texture for the skybox background
	com_ptr<ID3D12DescriptorHeap> srvHeap; // descriptor heap for shader-visible resources
	com_ptr<ID3D12Resource> particleBuffer; // default heap: GPU-local, UAV-accessible
	com_ptr<ID3D12Resource> particleUploadBuffer; // upload heap: used once to transfer initial particle data to the GPU
	com_ptr<ID3D12Resource> cellCountBuffer; // default heap: uint per cell, stores how many particles are in each cell
	com_ptr<ID3D12Resource> cellParticlesBuffer; // default heap: flat 2D array [maxNumCells * maxPerCell], stores particle indices per cell
	com_ptr<ID3D12Resource> sortedParticleBuffer; // default heap: reorder target buffer, same size as particleBuffer
	com_ptr<ID3D12Resource> cellPrefixSumBuffer; // default heap: exclusive prefix sum of cellCount, used by reorderCS
	Egg::ConstantBuffer<ComputeCb> computeCb; // constant buffer uploaded to GPU each frame -> compute data
	// all compute shaders share the same root signature layout:
	//   CBV(b0) + DescriptorTable(UAV(u0, numDescriptors=5))
	// so we extract the root sig from one shader and reuse it for all PSOs.
	// The five UAVs are: u0 = particles, u1 = cellCount, u2 = cellParticles, u3 = sortedParticles, u4 = cellPrefixSum.
	// u0 = particle data
	// u1 = grid cell counts, i.e. how many particles are in each cell 
	// u2 = grid cell particle indices, i.e. which particles are in each cell (up to maxPerCell per cell)
	// u3 = sorted particle data, i.e. the same particles but scattered into grid order for better memory coherence during constraint solving
	// u4 = grid cell prefix sum, i.e. the exclusive prefix sum of cellCount, used to compute scatter offsets during reorder
	com_ptr<ID3D12RootSignature> computeRootSig;
	com_ptr<ID3D12PipelineState> predictPso; // apply gravity, compute position prediction p*
	com_ptr<ID3D12PipelineState> clearGridPso; // zero cellCount array
	com_ptr<ID3D12PipelineState> buildGridPso; // populate grid from predictedPositions
	com_ptr<ID3D12PipelineState> lambdaPso; // compute lambda per particle
	com_ptr<ID3D12PipelineState> deltaPso; // compute delta_p, update p*
	com_ptr<ID3D12PipelineState> collisionPso; // clamp predictedPosition to box, zero wall-normal velocity
	com_ptr<ID3D12PipelineState> positionFromScratchPso; // copy scratch -> predictedPosition (Jacobi commit during solver loop)
	com_ptr<ID3D12PipelineState> updateVelocityPso; // update velocity from displacement: v = (p* - x) / dt
	com_ptr<ID3D12PipelineState> vorticityPso; // estimate per-particle vorticity (curl of velocity), store in omega
	com_ptr<ID3D12PipelineState> confinementPso; // apply vorticity confinement force to velocity
	com_ptr<ID3D12PipelineState> viscosityPso; // XSPH velocity smoothing, writes to scratch
	com_ptr<ID3D12PipelineState> velocityFromScratchPso; // copy scratch -> velocity (Jacobi commit after viscosity)
	com_ptr<ID3D12PipelineState> updatePositionPso; // update position from predictedPosition (final step per paper)
	com_ptr<ID3D12PipelineState> prefixSumPso; // exclusive prefix sum of cellCount -> cellPrefixSum (for spatial reorder)
	com_ptr<ID3D12PipelineState> reorderPso; // scatter particles into sorted order by grid cell
	com_ptr<ID3D12DescriptorHeap> imguiSrvHeap; // dedicated 1-slot SRV heap for ImGui's font texture

	// Readback buffer for copying particle data to CPU (e.g. for diagnostics or export).
	// Usage: transition particleBuffer to COPY_SOURCE, CopyBufferRegion into this, transition back,
	// then Map/Unmap after the GPU finishes to read the data on CPU.
	com_ptr<ID3D12Resource> particleReadbackBuffer;
	std::vector<Particle> particleReadbackData;
	float avgDensity = 0.0f; // average particle density from previous frame's readback

	bool physicsRunning = false; // toggled by spacebar: when false, compute passes are skipped each frame
	bool arrowLeft = false, arrowRight = false, arrowUp = false, arrowDown = false; // arrow key held state for box translation
	int reorderInterval = 5; // how often to spatially reorder the particle buffer (every N physics frames)
	int framesSinceReorder = 0; // counts up to reorderInterval, then resets after a reorder


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
		// load all compiled compute shader blobs
		com_ptr<ID3DBlob> predictShader = Egg::Shader::LoadCso("Shaders/predictCS.cso");
		com_ptr<ID3DBlob> clearGridShader = Egg::Shader::LoadCso("Shaders/clearGridCS.cso");
		com_ptr<ID3DBlob> buildGridShader = Egg::Shader::LoadCso("Shaders/buildGridCS.cso");
		com_ptr<ID3DBlob> lambdaShader = Egg::Shader::LoadCso("Shaders/lambdaCS.cso");
		com_ptr<ID3DBlob> deltaShader = Egg::Shader::LoadCso("Shaders/deltaCS.cso");
		com_ptr<ID3DBlob> collisionShader = Egg::Shader::LoadCso("Shaders/collisionCS.cso");
		com_ptr<ID3DBlob> positionFromScratchShader = Egg::Shader::LoadCso("Shaders/positionFromScratchCS.cso");
		com_ptr<ID3DBlob> updateVelocityShader = Egg::Shader::LoadCso("Shaders/updateVelocityCS.cso");
		com_ptr<ID3DBlob> vorticityShader = Egg::Shader::LoadCso("Shaders/vorticityCS.cso");
		com_ptr<ID3DBlob> confinementShader = Egg::Shader::LoadCso("Shaders/confinementCS.cso");
		com_ptr<ID3DBlob> viscosityShader = Egg::Shader::LoadCso("Shaders/viscosityCS.cso");
		com_ptr<ID3DBlob> velocityFromScratchShader = Egg::Shader::LoadCso("Shaders/velocityFromScratchCS.cso");
		com_ptr<ID3DBlob> updatePositionShader = Egg::Shader::LoadCso("Shaders/updatePositionCS.cso");
		com_ptr<ID3DBlob> prefixSumShader = Egg::Shader::LoadCso("Shaders/prefixSumCS.cso");
		com_ptr<ID3DBlob> reorderShader = Egg::Shader::LoadCso("Shaders/reorderCS.cso");

		// all shaders embed the same root signature string, so we extract it once
		// from predictCS and reuse it for all PSO descriptors
		computeRootSig = Egg::Shader::LoadRootSignature(device.Get(), predictShader.Get());

		// compute PSO descriptor: much simpler than a graphics PSO --
		// no rasterizer, blend state, render targets, or input layout; just root sig + shader bytecode
		D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
		psoDesc.pRootSignature = computeRootSig.Get();

		psoDesc.CS = CD3DX12_SHADER_BYTECODE(predictShader.Get());
		DX_API("Failed to create predict compute PSO")
			device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(predictPso.GetAddressOf()));

		psoDesc.CS = CD3DX12_SHADER_BYTECODE(clearGridShader.Get());
		DX_API("Failed to create clearGrid compute PSO")
			device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(clearGridPso.GetAddressOf()));

		psoDesc.CS = CD3DX12_SHADER_BYTECODE(buildGridShader.Get());
		DX_API("Failed to create buildGrid compute PSO")
			device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(buildGridPso.GetAddressOf()));

		psoDesc.CS = CD3DX12_SHADER_BYTECODE(lambdaShader.Get());
		DX_API("Failed to create lambda compute PSO")
			device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(lambdaPso.GetAddressOf()));

		psoDesc.CS = CD3DX12_SHADER_BYTECODE(deltaShader.Get());
		DX_API("Failed to create delta compute PSO")
			device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(deltaPso.GetAddressOf()));

		psoDesc.CS = CD3DX12_SHADER_BYTECODE(collisionShader.Get());
		DX_API("Failed to create collision compute PSO")
			device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(collisionPso.GetAddressOf()));

		psoDesc.CS = CD3DX12_SHADER_BYTECODE(positionFromScratchShader.Get());
		DX_API("Failed to create positionFromScratch compute PSO")
			device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(positionFromScratchPso.GetAddressOf()));

		psoDesc.CS = CD3DX12_SHADER_BYTECODE(updateVelocityShader.Get());
		DX_API("Failed to create updateVelocity compute PSO")
			device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(updateVelocityPso.GetAddressOf()));

		psoDesc.CS = CD3DX12_SHADER_BYTECODE(vorticityShader.Get());
		DX_API("Failed to create vorticity compute PSO")
			device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(vorticityPso.GetAddressOf()));

		psoDesc.CS = CD3DX12_SHADER_BYTECODE(confinementShader.Get());
		DX_API("Failed to create confinement compute PSO")
			device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(confinementPso.GetAddressOf()));

		psoDesc.CS = CD3DX12_SHADER_BYTECODE(viscosityShader.Get());
		DX_API("Failed to create viscosity compute PSO")
			device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(viscosityPso.GetAddressOf()));

		psoDesc.CS = CD3DX12_SHADER_BYTECODE(velocityFromScratchShader.Get());
		DX_API("Failed to create velocityFromScratch compute PSO")
			device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(velocityFromScratchPso.GetAddressOf()));

		psoDesc.CS = CD3DX12_SHADER_BYTECODE(updatePositionShader.Get());
		DX_API("Failed to create updatePosition compute PSO")
			device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(updatePositionPso.GetAddressOf()));

		psoDesc.CS = CD3DX12_SHADER_BYTECODE(prefixSumShader.Get());
		DX_API("Failed to create prefixSum compute PSO")
			device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(prefixSumPso.GetAddressOf()));

		psoDesc.CS = CD3DX12_SHADER_BYTECODE(reorderShader.Get());
		DX_API("Failed to create reorder compute PSO")
			device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(reorderPso.GetAddressOf()));
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
			2, // slot index 2 = start of the UAV descriptor table (u0..u4)
			descriptorSize);

		// switch to the compute pipeline: root signature first, then PSO
		// (setting the root sig before the PSO is required - the PSO references the root sig)
		// all compute shaders share the same root signature, so we set it once here.
		// root parameter bindings (CBV + UAV table) persist across PSO swaps as long
		// as we don't call SetComputeRootSignature again.
		commandList->SetComputeRootSignature(computeRootSig.Get());
		// bind the compute CB at root parameter 0 via its GPU virtual address directly.
		// this is a root CBV - it doesn't occupy a descriptor heap slot, the address goes straight into the root signature.
		commandList->SetComputeRootConstantBufferView(0, computeCb.GetGPUVirtualAddress());
		// bind the descriptor table at root parameter 1: 5 consecutive UAVs starting at slot 2
		// (u0 = particles, u1 = cellCount, u2 = cellParticles, u3 = sortedParticles, u4 = cellPrefixSum)
		commandList->SetComputeRootDescriptorTable(1, uavHandle);

		// ceil(numParticles / 256) groups cover all particles; the shader discards extra threads
		UINT numGroups = (numParticles + 255) / 256;

		// prediction: apply gravity, compute p* = x + v*dt
		commandList->SetPipelineState(predictPso.Get());
		commandList->Dispatch(numGroups, 1, 1);
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(particleBuffer.Get()));

		// pre-stabilization (Koster & Kruger 2016): clamp predicted positions to the
		// simulation box and zero wall-normal velocity before building the grid.
		commandList->SetPipelineState(collisionPso.Get());
		commandList->Dispatch(numGroups, 1, 1);
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(particleBuffer.Get()));

		// build the spatial grid from predicted positions (used by all subsequent neighbor queries)
		// step 1: zero the cell counts
		UINT numCells = (UINT)ceilf((boxMax.x - boxMin.x) / h) * (UINT)ceilf((boxMax.y - boxMin.y) / h) * (UINT)ceilf((boxMax.z - boxMin.z) / h);
		UINT numCellGroups = (numCells + 255) / 256;
		commandList->SetPipelineState(clearGridPso.Get());
		commandList->Dispatch(numCellGroups, 1, 1);
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(cellCountBuffer.Get()));

		// step 2: each particle inserts itself into its cell via InterlockedAdd
		commandList->SetPipelineState(buildGridPso.Get());
		commandList->Dispatch(numGroups, 1, 1);
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(cellCountBuffer.Get()));
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(cellParticlesBuffer.Get()));

		// constraint solver loop
		for (int iter = 0; iter < solverIterations; iter++) {
			// compute lambda for each particle from predictedPositions
			commandList->SetPipelineState(lambdaPso.Get());
			commandList->Dispatch(numGroups, 1, 1);
			commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(particleBuffer.Get()));

			// writes corrected position to scratch (avoids Gauss-Seidel race)
			commandList->SetPipelineState(deltaPso.Get());
			commandList->Dispatch(numGroups, 1, 1);
			commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(particleBuffer.Get()));

			// commit scratch -> predictedPosition for next iteration
			commandList->SetPipelineState(positionFromScratchPso.Get());
			commandList->Dispatch(numGroups, 1, 1);
			commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(particleBuffer.Get()));

			// post-delta collision: clamp predictedPosition to the simulation box
			// so the next solver iteration sees valid positions
			commandList->SetPipelineState(collisionPso.Get());
			commandList->Dispatch(numGroups, 1, 1);
			commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(particleBuffer.Get()));
		}

		// update velocity from displacement (position stays old for vorticity/viscosity)
		commandList->SetPipelineState(updateVelocityPso.Get());
		commandList->Dispatch(numGroups, 1, 1);
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(particleBuffer.Get()));

		// vorticity estimation: compute curl(v) per particle, store in omega
		commandList->SetPipelineState(vorticityPso.Get());
		commandList->Dispatch(numGroups, 1, 1);
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(particleBuffer.Get()));

		// vorticity confinement: apply corrective force to velocity
		commandList->SetPipelineState(confinementPso.Get());
		commandList->Dispatch(numGroups, 1, 1);
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(particleBuffer.Get()));

		// XSPH viscosity: writes corrected velocity to scratch (avoids Gauss-Seidel race)
		commandList->SetPipelineState(viscosityPso.Get());
		commandList->Dispatch(numGroups, 1, 1);
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(particleBuffer.Get()));

		// commit scratch -> velocity
		commandList->SetPipelineState(velocityFromScratchPso.Get());
		commandList->Dispatch(numGroups, 1, 1);
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(particleBuffer.Get()));

		// update position from predictedPosition
		commandList->SetPipelineState(updatePositionPso.Get());
		commandList->Dispatch(numGroups, 1, 1);
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(particleBuffer.Get()));

		// spatial sort (reorder): rearrange the particle buffer so that particles
		// in the same grid cell are contiguous in memory, restoring cache coherence
		// for neighbor lookups. Uses the grid (cellCount + cellParticles) built earlier
		// this frame to drive the reorder, avoiding any mismatch between grid-time
		// and current positions. The grid was built from pre-solver predictedPositions,
		// but by now position has been updated by the solver. This is fine, because if
		// the solver converges, the predicted and actual positions are close spatially,
		// meaning that the grouping of particles into cells is still mostly valid.
		//
		// The heart and soul of this sorting is that when we walk the grid cell-by-cell,
		// we put the particles we find in them into the next open slot in the sorted buffer, 
		// (which we track with a prefix sum of the cell counts), where they are contigous in memory,
		// meaning that particles belonging to the same cell are contigous in memory, and are 
		// also more likely to be contiguous with particles that are in adjacent cells.
		framesSinceReorder++;
		if (framesSinceReorder >= reorderInterval) {
			framesSinceReorder = 0;

			// Step 1: prefix sum of cellCount -> cellPrefixSum
			// Tells us where each cell's particles start in the sorted buffer.
			// cellCount is still populated from the grid build earlier this frame.
			commandList->SetPipelineState(prefixSumPso.Get());
			commandList->Dispatch(1, 1, 1); // single thread
			commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(cellPrefixSumBuffer.Get()));

			// Step 2: scatter particles into sortedParticleBuffer in cell order.
			// Dispatched over (cell, slot) pairs: numCells * maxPerCell threads total.
			// Each thread copies one particle from particles[] to its sorted position, 
			// if there is a particle in the given cell's given spot
			UINT reorderThreads = numCells * maxPerCell;
			UINT reorderGroups = (reorderThreads + 255) / 256;
			commandList->SetPipelineState(reorderPso.Get());
			commandList->Dispatch(reorderGroups, 1, 1);
			commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(sortedParticleBuffer.Get()));

			// Step 4: copy sorted data back into the main particle buffer
			// sortedParticleBuffer becomes COPY_SOURCE, particleBuffer becomes COPY_DEST for the copy call
			commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition( 
				sortedParticleBuffer.Get(),
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE));
			commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
				particleBuffer.Get(),
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST));
			commandList->CopyBufferRegion(particleBuffer.Get(), 0,
				sortedParticleBuffer.Get(), 0, numParticles * sizeof(Particle));
			// then transition both back to UAV so the compute shaders can use it next frame
			commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
				sortedParticleBuffer.Get(),
				D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
			commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
				particleBuffer.Get(),
				D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
		}

		// Copy particle data to readback buffer for CPU access next frame
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
			particleBuffer.Get(),
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE));
		commandList->CopyBufferRegion(particleReadbackBuffer.Get(), 0,
			particleBuffer.Get(), 0, numParticles * sizeof(Particle));
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
			particleBuffer.Get(),
			D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
	}

	void GraphicsPass() {
		backgroundMesh->Draw(commandList.Get()); // draw skybox at the back first

		// The particle buffer's home state is UNORDERED_ACCESS, because that's what the compute
		// shaders need (they bind it as RWStructuredBuffer). The vertex shader, however, reads it
		// as a StructuredBuffer (SRV), which requires the resource to be in a shader-resource state.
		// DX12 enforces this so the GPU knows whether to expect writes (requiring cache flushes and
		// coherence) or only reads (allowing aggressive caching). We transition in before the draw
		// and back out to UAV afterward, so the rest of the pipeline always sees the home state.
		// Both NON_PIXEL_SHADER_RESOURCE and PIXEL_SHADER_RESOURCE bits are required because the
		// root signature's default DATA_STATIC_WHILE_SET_AT_EXECUTE descriptor volatility demands
		// the full SRV mask regardless of which shader stage actually reads the resource.
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
			particleBuffer.Get(),
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));

		particleMesh->Draw(commandList.Get()); // draw particles on top

		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
			particleBuffer.Get(),
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
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

	void ImGuiPass() {
		// begin a new ImGui frame, which gives us a clean slate to construct the UI for this frame
		ImGui_ImplDX12_NewFrame(); // tell ImGui about the new frame for DX12
		ImGui_ImplWin32_NewFrame(); // tell ImGui about the new frame for Win32 (input handling, time, etc)
		// the core library consumes the input state the backends just wrote and begins a new frame
		ImGui::NewFrame(); // after this we can create ImGui widgets for this frame


		// InputFloat/InputInt: text field with +/- stepper buttons. Type a value and press Enter.
		// The "step" argument is how much the +/- buttons change the value per click.
		// This is the immediate mode paradigm: we construct the same UI every frame, and ImGui handles the state internally.
		// InputFloat/Int reads the current value from the pointer, renders the widget into the draw list, and
		// writes the value back to the pointer if the user changed it
		ImGui::Begin("PBF Controls");
		ImGui::PushItemWidth(100); // set the input field width to 100 pixels (just the number box, not the label)
		//ImGui::Checkbox("Physics running (Space)", &physicsRunning);
		ImGui::InputInt("Solver iterations [4]", &solverIterations, 1); // step 1 per click
		ImGui::InputFloat("Particle spacing [0.25]", &particleSpacing, 0.01f, 0.1f, "%.4f");
		particleSpacing = std::clamp(particleSpacing, particleSpacingMin, particleSpacingMax);
		ImGui::InputFloat("h multiplier [3.5]", &hMultiplier, 0.1f, 0.5f, "%.2f");
		hMultiplier = std::clamp(hMultiplier, hMultiplierMin, hMultiplierMax);
		ImGui::InputFloat("Epsilon (relaxation) [5.0]", &epsilon, 0.5f, 1.0f, "%.2f");
		ImGui::InputFloat("Viscosity (XSPH) [0.01]", &viscosity, 0.005f, 0.01f, "%.4f");
		ImGui::InputFloat("Artificial pressure [0.05]", &sCorrK, 0.01f, 0.05f, "%.4f");
		ImGui::InputFloat("Vorticity epsilon [0.01]", &vorticityEpsilon, 0.005f, 0.01f, "%.4f");
		ImGui::InputInt("Reorder interval [5]", &reorderInterval, 10); // how often to sort particles by cell
		reorderInterval = std::max(reorderInterval, 1); // must be at least 1
		ImGui::PopItemWidth(); // restore default width for any subsequent widgets
		// show derived values as read-only text for reference
		ImGui::Separator(); // horizontal line to separate tunable parameters from derived values
		ImGui::Text("%d particles, h = %.4f, rho0 = %.2f",numParticles, h, rho0);
		ImGui::Text("%.1f FPS (%.2f ms)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
		ImGui::Text("avg density: %.2f (rho0: %.2f)", avgDensity, rho0);
		ImGui::End();

		// Finalizes the frame.ImGui takes all the widgets you defined since NewFrame(), performs layout
		// (positions, sizes, clipping), and produces an ImDrawData structure : a list of vertex buffers, index
		// buffers, and draw commands that describe exactly what triangles to draw and with what textures.No
		// GPU calls happen here - it's pure CPU-side geometry generation.
		ImGui::Render();
		// ImGui needs its own SRV heap bound (for the font texture), so we switch heaps here.
		// The scene's srvHeap was used during GraphicsPass; that's done, so this is safe.
		commandList->SetDescriptorHeaps(1, imguiSrvHeap.GetAddressOf());
		// This is where ImGui's geometry actually gets drawn. GetDrawData() returns the ImDrawData that
		// Render() produced.The D3D12 backend takes it and :
		//  1. Selects this frame's rotating vertex/index buffer pair (alternating between 2 sets for double buffering)
		//	2. Maps the upload buffers and copies ImGui's vertex + index data into them
		//	3. Sets its own root signature and PSO on the command list
		//	4. Sets the viewport, blend factor, and stencil ref
		//	5. For each draw command : sets the scissor rect(ImGui uses scissor for clipping), binds the font
		//		texture SRV, and issues an indexed draw call
		//
		//	After this returns, the command list contains all the triangles needed to render the UI panel, text,
		//	and input fields on top of our scene.
		ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), commandList.Get());
	}

	// this is the method we must implement from SimpleApp, Render() calls it,
	// this is where we record all GPU commands to construct one frame
	virtual void PopulateCommandList() override {
		PrepareCommandList(); // set up the command list for a new frame: reset, set viewport/scissor, transition back buffer to render target

		if (physicsRunning)
			ComputePass(); // dispatch the compute shader to update particle positions on the GPU

		GraphicsPass(); // draw the background and particles using the updated positions

		// draw the parameter tuning UI *after* the main scene so it appears on top
		ImGuiPass(); 

		CloseCommandList(); // transition back buffer to present state and close the command list
	}

	void UpdateExternalForce() {
		// build a horizontal acceleration vector from held arrow keys
		// left/right push along X, up/down push along Z
		externalForce = Float3(0.0f, 0.0f, 0.0f);
		if (arrowLeft) externalForce.x -= externalAcceleration;
		if (arrowRight) externalForce.x += externalAcceleration;
		if (arrowUp) externalForce.z += externalAcceleration;
		if (arrowDown) externalForce.z -= externalAcceleration;
	}

	// recompute values that depend on the primary tunables (particleSpacing, hMultiplier)
	// must be called each frame before uploading constant buffers, because sliders may have changed
	void RecomputeDerivedParams() {
		h = particleSpacing * hMultiplier;
		rho0 = 1.0f / powf(particleSpacing, 3.0f);
		sCorrDeltaQ = 0.2f * h;
	}

	void UpdatePerFrameCb() {
		perFrameCb->viewProjTransform = // calculate the combined view-projection matrix and store it in the constant buffer
			camera->GetViewMatrix() * // view matrix: world space -> camera space
			camera->GetProjMatrix(); // projection matrix: camera space -> clip space
		perFrameCb->rayDirTransform = camera->GetRayDirMatrix(); // clip-space coords -> world-space view direction
		perFrameCb->cameraPos = Egg::Math::Float4(camera->GetEyePosition(), 1.0f);
		perFrameCb->lightDir = Egg::Math::Float4(0.5f, 1.0f, 0.3f, 0.0f); // light pointing down-left
		perFrameCb->particleParams = Float4(rho0, 0.0f, 0.0f, 0.4f * particleSpacing); // x = rho0 (for density coloring in PS), w = particle display radius (for billboard sizing in GS)

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
		computeCb->vorticityEpsilon = vorticityEpsilon;
		computeCb->externalForce = externalForce;
		computeCb->maxPerCell = maxPerCell;
		computeCb.Upload();
	}

	virtual void Update(float dt, float T) override {
		dt = std::min(dt, 1.0f / 30.0f); // cap at 33ms: prevents energy spikes on window drag or stutter
		camera->Animate(dt); // update camera position and orientation based on user input

		UpdateExternalForce();

		RecomputeDerivedParams(); // recalculate h, rho0, sCorrDeltaQ from current slider values

		UpdatePerFrameCb();

		UpdateComputeCb(dt);

		
		// Read back previous frame's particle data (GPU is done after WaitForPreviousFrame)
		if (physicsRunning) { CalculateAvgDensity(); }
	}

	void CalculateAvgDensity() {
		// map the readback buffer to CPU memory and copy the particle data into a vector
		const UINT64 bufferSize = numParticles * sizeof(Particle);
		void* pData;
		CD3DX12_RANGE readRange(0, bufferSize);
		if (SUCCEEDED(particleReadbackBuffer->Map(0, &readRange, &pData))) {
			memcpy(particleReadbackData.data(), pData, bufferSize);
			CD3DX12_RANGE writeRange(0, 0);
			particleReadbackBuffer->Unmap(0, &writeRange);
		}

		// Compute average density from readback data
		double densitySum = 0.0;
		for (int i = 0; i < numParticles; i++)
			densitySum += particleReadbackData[i].density;
		avgDensity = static_cast<float>(densitySum / numParticles);
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

	void CreateParticleSrv() {
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

	void CreateParticleUav() {
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

	void CreateGridBuffers() {
		const CD3DX12_HEAP_PROPERTIES defaultHeapProps(D3D12_HEAP_TYPE_DEFAULT);

		// cellCount buffer: one uint per cell, indicating how many particles are in each
		// cell, zeroed each frame by clearGridCS, then incremented atomically by buildGridCS
		const UINT64 cellCountSize = maxNumCells * sizeof(UINT);
		const CD3DX12_RESOURCE_DESC cellCountDesc = CD3DX12_RESOURCE_DESC::Buffer(
			cellCountSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		DX_API("Failed to create cell count buffer")
			device->CreateCommittedResource(
				&defaultHeapProps, D3D12_HEAP_FLAG_NONE, &cellCountDesc,
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
				IID_PPV_ARGS(cellCountBuffer.ReleaseAndGetAddressOf()));
		cellCountBuffer->SetName(L"Cell Count Buffer");

		// cellParticles buffer: flat 2D array of particle indices, indicating which particles are 
		// in each cell. This is a flattened array, where the first maxPerCell entries belong to cell 0, 
		// the next maxPerCell entries belong to cell 1, and so on. Therefore, the "slot"th slot in the
		// "cellIndex"th cell is accessed as cellParticles[cellIndex * maxPerCell + slot]
		// The number of cells at a given time is determined by h, which can change at runtime, so
		// only the first cellCount[number_of_cells] entries per cell are valid; the rest are unused.
		const UINT64 cellParticlesSize = (UINT64)maxNumCells * maxPerCell * sizeof(UINT);
		const CD3DX12_RESOURCE_DESC cellParticlesDesc = CD3DX12_RESOURCE_DESC::Buffer(
			cellParticlesSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		DX_API("Failed to create cell particles buffer")
			device->CreateCommittedResource(
				&defaultHeapProps, D3D12_HEAP_FLAG_NONE, &cellParticlesDesc,
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
				IID_PPV_ARGS(cellParticlesBuffer.ReleaseAndGetAddressOf()));
		cellParticlesBuffer->SetName(L"Cell Particles Buffer");
	}

	void CreateGridUavs() {
		UINT descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

		// cellCount UAV (slot 3): one uint per cell
		D3D12_UNORDERED_ACCESS_VIEW_DESC cellCountUavDesc = {};
		cellCountUavDesc.Format = DXGI_FORMAT_UNKNOWN;
		cellCountUavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		cellCountUavDesc.Buffer.FirstElement = 0;
		cellCountUavDesc.Buffer.NumElements = maxNumCells;
		cellCountUavDesc.Buffer.StructureByteStride = sizeof(UINT);
		cellCountUavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
		CD3DX12_CPU_DESCRIPTOR_HANDLE cellCountHandle(
			srvHeap->GetCPUDescriptorHandleForHeapStart(), 3, descriptorSize);
		device->CreateUnorderedAccessView(cellCountBuffer.Get(), nullptr, &cellCountUavDesc, cellCountHandle);

		// cellParticles UAV (slot 4): flat array of maxNumCells * maxPerCell uints
		D3D12_UNORDERED_ACCESS_VIEW_DESC cellParticlesUavDesc = {};
		cellParticlesUavDesc.Format = DXGI_FORMAT_UNKNOWN;
		cellParticlesUavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		cellParticlesUavDesc.Buffer.FirstElement = 0;
		cellParticlesUavDesc.Buffer.NumElements = maxNumCells * maxPerCell;
		cellParticlesUavDesc.Buffer.StructureByteStride = sizeof(UINT);
		cellParticlesUavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
		CD3DX12_CPU_DESCRIPTOR_HANDLE cellParticlesHandle(
			srvHeap->GetCPUDescriptorHandleForHeapStart(), 4, descriptorSize);
		device->CreateUnorderedAccessView(cellParticlesBuffer.Get(), nullptr, &cellParticlesUavDesc, cellParticlesHandle);
	}

	void CreateSortBuffers() {
		const CD3DX12_HEAP_PROPERTIES defaultHeapProps(D3D12_HEAP_TYPE_DEFAULT);

		// sortedParticleBuffer: same size and flags as particleBuffer, used as
		// the scatter destination during spatial reordering
		const UINT64 particleSize = numParticles * sizeof(Particle);
		const CD3DX12_RESOURCE_DESC sortedParticleDesc = CD3DX12_RESOURCE_DESC::Buffer(
			particleSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		DX_API("Failed to create sorted particle buffer")
			device->CreateCommittedResource(
				&defaultHeapProps, D3D12_HEAP_FLAG_NONE, &sortedParticleDesc,
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
				IID_PPV_ARGS(sortedParticleBuffer.ReleaseAndGetAddressOf()));
		sortedParticleBuffer->SetName(L"Sorted Particle Buffer");

		// cellPrefixSumBuffer: one uint per cell, stores the exclusive prefix sum
		// of cellCount — i.e. where each cell's particles start in the sorted buffer
		const UINT64 prefixSumSize = maxNumCells * sizeof(UINT);
		const CD3DX12_RESOURCE_DESC prefixSumDesc = CD3DX12_RESOURCE_DESC::Buffer(
			prefixSumSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		DX_API("Failed to create cell prefix sum buffer")
			device->CreateCommittedResource(
				&defaultHeapProps, D3D12_HEAP_FLAG_NONE, &prefixSumDesc,
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
				IID_PPV_ARGS(cellPrefixSumBuffer.ReleaseAndGetAddressOf()));
		cellPrefixSumBuffer->SetName(L"Cell Prefix Sum Buffer");
	}

	void CreateSortUavs() {
		UINT descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

		// sortedParticles UAV (slot 5): same element layout as the main particle UAV
		D3D12_UNORDERED_ACCESS_VIEW_DESC sortedUavDesc = {};
		sortedUavDesc.Format = DXGI_FORMAT_UNKNOWN;
		sortedUavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		sortedUavDesc.Buffer.FirstElement = 0;
		sortedUavDesc.Buffer.NumElements = numParticles;
		sortedUavDesc.Buffer.StructureByteStride = sizeof(Particle);
		sortedUavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
		CD3DX12_CPU_DESCRIPTOR_HANDLE sortedHandle(
			srvHeap->GetCPUDescriptorHandleForHeapStart(), 5, descriptorSize);
		device->CreateUnorderedAccessView(sortedParticleBuffer.Get(), nullptr, &sortedUavDesc, sortedHandle);

		// cellPrefixSum UAV (slot 6): one uint per cell
		D3D12_UNORDERED_ACCESS_VIEW_DESC prefixSumUavDesc = {};
		prefixSumUavDesc.Format = DXGI_FORMAT_UNKNOWN;
		prefixSumUavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		prefixSumUavDesc.Buffer.FirstElement = 0;
		prefixSumUavDesc.Buffer.NumElements = maxNumCells;
		prefixSumUavDesc.Buffer.StructureByteStride = sizeof(UINT);
		prefixSumUavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
		CD3DX12_CPU_DESCRIPTOR_HANDLE prefixSumHandle(
			srvHeap->GetCPUDescriptorHandleForHeapStart(), 6, descriptorSize);
		device->CreateUnorderedAccessView(cellPrefixSumBuffer.Get(), nullptr, &prefixSumUavDesc, prefixSumHandle);
	}

	void CreateParticleReadbackBuffer() {
		// create the readback buffer: same size as the particle buffer, but on the 
		// readback heap so we can copy GPU data back to the CPU
		const UINT64 bufferSize = numParticles * sizeof(Particle);
		const CD3DX12_HEAP_PROPERTIES readbackHeapProps(D3D12_HEAP_TYPE_READBACK);
		const CD3DX12_RESOURCE_DESC readbackDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);
		DX_API("Failed to create particle readback buffer")
			device->CreateCommittedResource(
				&readbackHeapProps, D3D12_HEAP_FLAG_NONE, &readbackDesc,
				D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
				IID_PPV_ARGS(particleReadbackBuffer.ReleaseAndGetAddressOf()));
		particleReadbackBuffer->SetName(L"Particle Readback Buffer");
		particleReadbackData.resize(numParticles);
	}

	void CreateDescriptorHeap() {
		// The texture cube lives in gpu memory, and will be created by ImportTextureCube as a committed resource
		// on the DEFAULT heap. We access it via a Shader Resource View (SRV), which is a type of descriptor. Descriptors
		// live in a descriptor heap, which has to be big enough to hold a descriptor for each of our resources, one of
		// which is the cube map texture.
		// descriptor heap layout:
		//   slot 0: cubemap SRV       - sampled by the background pixel shader
		//   slot 1: particle SRV (t0) - read by the vertex shader to fetch particle positions
		//   slot 2: particle UAV (u0) - read/written by compute shaders (particle data)
		//   slot 3: cellCount UAV (u1) - per-cell particle count for the spatial grid
		//   slot 4: cellParticles UAV (u2) - particle indices per cell (flat 2D: cell * maxPerCell + slot)
		//   slot 5: sortedParticles UAV (u3) - reorder target for spatial sorting
		//   slot 6: cellPrefixSum UAV (u4) - exclusive prefix sum of cellCount (for reorder offsets)
		// The compute root signature binds slots 2-6 as a single descriptor table with 5 UAVs.
		D3D12_DESCRIPTOR_HEAP_DESC dhd;
		dhd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE; // GPU can see these descriptors
		dhd.NodeMask = 0; // single GPU setup, so no node masking needed
		dhd.NumDescriptors = 7;
		dhd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; // heap type that holds CBVs, SRVs, and UAVs

		DX_API("Failed to create descriptor heap")
			device->CreateDescriptorHeap(&dhd, IID_PPV_ARGS(srvHeap.GetAddressOf()));
	}

	void CreateImGuiDescriptorHeap() {
		// create a dedicated 1-slot SRV descriptor heap for ImGui's internal font texture.
		// this is separate from our scene's srvHeap so we don't have to change the existing layout.
		D3D12_DESCRIPTOR_HEAP_DESC imguiHeapDesc = {};
		imguiHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
		imguiHeapDesc.NumDescriptors = 1;
		imguiHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
		DX_API("Failed to create ImGui SRV descriptor heap")
			device->CreateDescriptorHeap(&imguiHeapDesc, IID_PPV_ARGS(imguiSrvHeap.GetAddressOf()));
	}
	
public:

	virtual void CreateResources() override {
		Egg::SimpleApp::CreateResources(); // creates command allocator, command list, PSO manager, and sync objects (fence)
		perFrameCb.CreateResources(device.Get()); // create the constant buffer on the GPU (upload heap, so CPU can write to it every frame)
		computeCb.CreateResources(device.Get()); // create the compute constant buffer (upload heap: dt and numParticles written each frame)
		camera = Egg::Cam::FirstPerson::Create(); // create the camera, which will handle user input and calculate view/projection matrices
		camera->SetSpeed(5.0f); // movement speed of the camera

		CreateImGuiDescriptorHeap();

		CreateDescriptorHeap();

		CreateParticleBuffer();

		CreateUploadBuffer();

		CreateGridBuffers();

		CreateParticleSrv();

		CreateParticleUav();

		CreateGridUavs();

		CreateSortBuffers();

		CreateSortUavs();

		CreateParticleReadbackBuffer();
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

	// Call once after CreateResources + LoadAssets, from main.cpp where the HWND is available.
	// Sets up ImGui context and its Win32 + D3D12 backends. At this point the D3D12 device, command queue, and
	//imguiSrvHeap all exist.
	void InitImGui(HWND hwnd) {
		IMGUI_CHECKVERSION(); // checks that the headers and compiled .lib are from the same version of ImGui
		// create the ImGui context, which stores ImGui's internal state and is needed before calling any ImGui functions
		ImGui::CreateContext(); 
		ImGui::StyleColorsDark();
		
		ImGui_ImplWin32_Init(hwnd); // Win32 backend: handles mouse position, keyboard input, cursor shape

		// D3D12 backend: renders ImGui's vertex/index buffers using our device and command queue.
		// We use the legacy single-descriptor path: one SRV for the font texture atlas.
		// Internally ImGui_ImplDX12_Init creates a root signature and PSO, allocates 
		// a two vertex/index buffers for swapping, creatres its own command allocator and command list,
		// writes the font texture srv into LegacySingleSrvCpuDescriptor and LegacySingleSrvGpuDescriptor
		ImGui_ImplDX12_InitInfo initInfo;
		initInfo.Device = device.Get();
		initInfo.CommandQueue = commandQueue.Get();
		initInfo.NumFramesInFlight = 2; // matches our double-buffered swap chain
		initInfo.RTVFormat = DXGI_FORMAT_R8G8B8A8_UNORM; // must match swap chain format
		initInfo.SrvDescriptorHeap = imguiSrvHeap.Get();
		initInfo.LegacySingleSrvCpuDescriptor = imguiSrvHeap->GetCPUDescriptorHandleForHeapStart();
		initInfo.LegacySingleSrvGpuDescriptor = imguiSrvHeap->GetGPUDescriptorHandleForHeapStart();
		ImGui_ImplDX12_Init(&initInfo);
	}

	void ShutdownImGui() {
		// Teardown in reverse order of initialization :
		// 1. ImGui_ImplDX12_Shutdown() - releases all D3D12 objects the backend created(PSOs, root
		//	  signatures, vertex / index buffers, command allocator, command list, font texture + its SRV)
		// 2. ImGui_ImplWin32_Shutdown() - unhooks from the window, clears input state
		// 3. ImGui::DestroyContext() - frees the global context(GImGui), setting it to nullptr.This is why
		//	  the GetCurrentContext() != nullptr guard in WindowProcess is necessary - messages arriving after
		//    this point must not call into ImGui.
		ImGui_ImplDX12_Shutdown();
		ImGui_ImplWin32_Shutdown();
		ImGui::DestroyContext();
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