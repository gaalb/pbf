#pragma once

#include <algorithm>
#include <Egg/SimpleApp.h>
#include "ConstantBufferTypes.h"
#include "ParticleTypes.h"
#include "ComputeShader.h"
#include <Egg/Cam/FirstPerson.h>
#include <Egg/ConstantBuffer.hpp>
#include <Egg/TextureCube.h>
#include <Egg/Importer.h>
#include <Egg/Mesh/Prefabs.h>
#include <imgui.h>
#include <imgui_impl_win32.h>
#include <imgui_impl_dx12.h>

using namespace Egg::Math;

// PBF implementation: based on the Macklin & Muller 2013 nvidia research article, I'll call it M&M
// SimpleApp gives us:
// command allocator and command list for recording GPU commands
// depth stencil buffer
// PSO manager
// frame synchronization (WaitForPreviousFrame)
// basic render that populates command list, executes it, presents and syncs
//
// main calls app->Run() in an infinite loop, which calculates elapsedTime and deltaTime,
// and calls Update (overridden), as well as Render (not overridden, simply PopulateCommandList,
// commandQueue->ExecuteCommandLists and WaitForPreviousFrame), in that order. In other words, when
// it comes to actual execution (not the setup), the only thing that gets exposed is Run(). And,
// for Run() to work, the only thing we need to do is correctly override Update, which updates
// the inner state of the class, and PopulateCommandList which fills the command list with the
// commands to actually draw the next frame.
class PbfApp : public Egg::SimpleApp {
protected:
	// Fixed particle and grid constants.
	const int particlesX = 50, particlesY = 100, particlesZ = 50; // number of particles along each axis of the initial grid
	const int offsetX = 0, offsetY = 0, offsetZ = 0; // world space offset of the center of the initial particle grid
	const int numParticles = particlesX * particlesY * particlesZ; // total number of particles in the simulation	
	// particleSpacing and hMultiplier are constants that define the SPH kernel width h,
	// which gives a lower bound to the spatial grid's cell width. We can use (try using...)
	// Morton codes to index the cells, which works best with a cubic simulation space that
	// has a power of two number of cells along each axis. Fixing h (= particleSpacing * hMultiplier) 
	// lets us define the box as exactly gridDim * h on each axis, giving a perfectly aligned 
	// cubic grid with a dense Morton code space.
	const float particleSpacing = 0.25f; // inter-particle distance (also determines rest density and display size)
	const float hMultiplier = 3.25f; // h = particleSpacing * hMultiplier
	const float h = particleSpacing * hMultiplier; // SPH smoothing radius = 0.875
	// if the particles are spaced "d" apart, then one d sided cube contains one particle, meaning that
	// each particle is responsible for d^3 volume of fluid, meaning that with m=1, the density is 1/d^3
	const float rho0 = 1.0f / powf(particleSpacing, 3.0f);
	const float sCorrDeltaQ = 0.2f * h; // reference distance for artificial pressure (paper: 0.1...0.3 * h)
	const float sCorrN = 3.0f; // exponent for artificial pressure (paper: 4)
	const Float3 particleColor = Float3(0.9f, 0.1f, 0.7f); // particle display color (RGB)
	const float externalAcceleration = 20.0f; // m/s^2, applied horizontally via arrow keys
	// Spatial grid: cubic, power-of-two cells per axis, box derived from grid.
	// The grid can use Morton code (Z-order curve) indexing, which requires equal power-of-two
	// dimensions on all axes for a dense index space (no wasted codes). We choose a single gridDim
	// for all three axes (cubic grid), such that the box has gridDim * h cells per axis.
	// With gridDim = 32 and h = 0.875, the box is 28 units per side, centered at the origin.
	const UINT gridDim = 32; // cells per axis (must be power of two)
	const UINT numCells = gridDim * gridDim * gridDim; // total cells in the grid
	const float boxExtent = gridDim * h; // box side length = 28.0
	const Float3 boxMin = Float3(-boxExtent / 2.0f, -boxExtent / 2.0f, -boxExtent / 2.0f);
	const Float3 boxMax = Float3(boxExtent / 2.0f, boxExtent / 2.0f, boxExtent / 2.0f);

	// parameters that are tunable via ImGui each frame
	int solverIterations = 4; // how many newton steps to take per frame
	float epsilon = 5.0f; // constraint force mixing relaxation parameter, higher value = softer constraints
	float viscosity = 0.01f; // // XSPH viscosity coefficient, higher value = "thicker" fluid, M&M: 0.01
	// artificial purely repulsive pressure term reduces clumping while leaving room for surface tension, 
	float sCorrK = 0.05f; // artificial pressure magnitude coefficient M&M: 0.1
	float vorticityEpsilon = 0.01f; // vorticity confinement strength M&M: 0.01
	bool fountainEnabled = false; // toggle for the upward jet in a corner of the box, like a fountain :)
	
	// non-constant members
	Float3 externalForce = Float3(0.0f, 0.0f, 0.0f); // current external acceleration from arrow keys
	Egg::Cam::FirstPerson::P camera; // WASD + mouse movement camera
	Egg::ConstantBuffer<PerFrameCb> perFrameCb; // constant buffer uploaded to GPU each frame -> graphics data
	Egg::Mesh::Shaded::P particleMesh; // combines material + geometry + PSO into one drawable
	Egg::Mesh::Shaded::P backgroundMesh; // fullscreen quad + cubemap shader
	Egg::TextureCube envTexture; // the cubemap texture for the skybox background

	Egg::ConstantBuffer<ComputeCb> computeCb; // constant buffer uploaded to GPU each frame -> compute data

	// Upload buffers for initial particle data (upload heap, CPU-writable).
	// Only position and velocity have meaningful initial values; other fields are
	// overwritten by the simulation before they're first read.
	com_ptr<ID3D12Resource> positionUploadBuffer;
	com_ptr<ID3D12Resource> velocityUploadBuffer;	

	com_ptr<ID3D12DescriptorHeap> imguiSrvHeap; // dedicated 1-slot SRV heap for ImGui's font texture
	com_ptr<ID3D12DescriptorHeap> descriptorHeap; // descriptor heap for shader-visible resources

	// particle field buffers (default heap, UAV-accessible by compute shaders): one per attribute
	// Index with ParticleField enum (PF_POSITION..PF_SCRATCH).
	com_ptr<ID3D12Resource> particleFields[PF_COUNT];	
	// Sorted particle field buffers (default heap, UAV). Same layout as particleFields, only
	// used as scatter target / scratch pad during sorting
	com_ptr<ID3D12Resource> sortedFields[PF_COUNT];
	com_ptr<ID3D12Resource> cellCountBuffer; // default heap: uint per cell, stores how many particles are in each cell
	com_ptr<ID3D12Resource> cellPrefixSumBuffer; // default heap: exclusive prefix sum of cellCount, used by sortCS and neighbor lookups
	
	// One ComputeShader per pass. Each holds its own PSO, root signature, descriptor
	// table bindings, and input/output resource lists for UAV barrier insertion.
	// GPU descriptor handles for the three UAV table ranges, computed once in LoadCompute.
	CD3DX12_GPU_DESCRIPTOR_HANDLE particleFieldsHandle; // UAV(u0..u6): particle field buffers
	CD3DX12_GPU_DESCRIPTOR_HANDLE gridHandle; // UAV(u7..u8): cellCount, cellPrefixSum
	CD3DX12_GPU_DESCRIPTOR_HANDLE sortedFieldsHandle; // UAV(u9..u15): sorted particle field buffers
	ComputeShader predictShader; // apply external forces like gravity, compute position prediction p*
	ComputeShader clearGridShader; // zero cellCount array
	ComputeShader countGridShader; // count particles per cell (first grid-build pass)
	ComputeShader prefixSumShader; // exclusive prefix sum of cellCount -> cellPrefixSum
	ComputeShader sortShader; // each particle writes itself to its sorted position by grid cell
	ComputeShader lambdaShader; // compute lambda per particle
	ComputeShader deltaShader; // compute delta_p, write to scratch
	ComputeShader positionFromScratchShader; // copy scratch -> predictedPosition (Jacobi commit during solver loop)
	ComputeShader collisionShader; // handle collision checking and response, such as clamping to bounding box
	ComputeShader updateVelocityShader;// update velocity from displacement: v = (p* - x) / dt
	ComputeShader vorticityShader; // estimate per-particle vorticity (curl of velocity), store in omega
	ComputeShader confinementShader; // apply vorticity confinement force to velocity
	ComputeShader viscosityShader;  // XSPH velocity smoothing, writes to scratch
	ComputeShader velocityFromScratchShader; // copy scratch -> velocity (Jacobi commit after viscosity)
	ComputeShader updatePositionShader; // update position from predictedPosition (final step per paper)
	

	// Readback buffer for density (readback heap, CPU-readable after CopyBufferRegion).
	// Serves as an example for reading particle data back from the GPU
	com_ptr<ID3D12Resource> densityReadbackBuffer;
	std::vector<float> densityReadbackData;
	float avgDensity = 0.0f; // average particle density from previous frame's readback

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
		envTexture.CreateSRV(device.Get(), descriptorHeap.Get(), 0);
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
		bgMaterial->SetSrvHeap(1, descriptorHeap, 0);

		// The fullscreen quad from Egg's prefab library - 2 triangles covering the entire screen
		// the geometry of a mesh is what handles raw vertex data
		Egg::Mesh::Geometry::P bgGeometry = Egg::Mesh::Prefabs::FullScreenQuad(device.Get());

		// Mesh = material + geometry
		backgroundMesh = Egg::Mesh::Shaded::Create(psoManager, bgMaterial, bgGeometry);
	}

	// CPU-side staging struct for ease of particle initialization.
	struct ParticleInitData {
		std::vector<Float3> positions;
		std::vector<Float3> velocities;
	};

	ParticleInitData GenerateParticles() {
		// create and return an evenly spaced grid of particles so we can see something on screen
		ParticleInitData data;
		Float3 grid = Float3(particlesX, particlesY, particlesZ);
		Float3 offset = -(grid * particleSpacing) / 2.0f; // shift so the cube is centered at the origin
		offset += Float3(offsetX, offsetY, offsetZ); // apply user-defined world space offset
		for (int x = 0; x < grid.x; x++) {
			for (int y = 0; y < grid.y; y++) {
				for (int z = 0; z < grid.z; z++) {
					data.positions.push_back(offset + Float3(x, y, z) * particleSpacing);
					data.velocities.push_back(Float3(0.0f, 0.0f, 0.0f)); // start at rest
				}
			}
		}
		return data;
	}

	void UploadParticles(const ParticleInitData& initData) {
		void* posData; // will point to the mapped CPU memory of the upload buffer
		CD3DX12_RANGE readRange(0, 0); // empty range: CPU won't read anything from this buffer
		// make the upload buffer's memory CPU accessible, i.e. positionUploadBuffer - posData association
		// 0 is subresource index, on success, posData points to the buffer 
		DX_API("Failed to map position upload buffer") 
			positionUploadBuffer->Map(0, &readRange, &posData);
		memcpy(posData, initData.positions.data(), initData.positions.size() * sizeof(Float3)); // actual copy call
		positionUploadBuffer->Unmap(0, nullptr); // release the CPU mapping -> posData is invalidated
			
		// same flow as above
		void* velData;
		DX_API("Failed to map velocity upload buffer")
			velocityUploadBuffer->Map(0, &readRange, &velData);
		memcpy(velData, initData.velocities.data(), initData.velocities.size() * sizeof(Float3));
		velocityUploadBuffer->Unmap(0, nullptr);

		// The particle data now lives in the Upload buffer, we must
		// copy it to the default heap. This is done by executing commands on the
		// command list, which requires us to reset it, as well as its associated allocator (the backing memory).
		DX_API("Failed to reset command allocator (UploadParticles)")
			commandAllocator->Reset();
		DX_API("Failed to reset command list (UploadParticles)")
			commandList->Reset(commandAllocator.Get(), nullptr);

		// In order to copy to them, we must transition position and velocity buffers to COPY_DEST
		// That's done by inserting transition type resource barriers to the command list.
		// Here we initialize the barriers.
		D3D12_RESOURCE_BARRIER toCopyDest[2] = {
			CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_POSITION].Get(),
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST),
			CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_VELOCITY].Get(),
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST),
		};
		commandList->ResourceBarrier(2, toCopyDest); // then insert them
		// then insert the copy commands to the command list: dst, offset, src, offset
		commandList->CopyBufferRegion(particleFields[PF_POSITION].Get(), 0,
			positionUploadBuffer.Get(), 0, numParticles * sizeof(Float3));
		commandList->CopyBufferRegion(particleFields[PF_VELOCITY].Get(), 0,
			velocityUploadBuffer.Get(), 0, numParticles * sizeof(Float3));

		// then transition back to UAV by, again initializing then inserting the proper barriers
		D3D12_RESOURCE_BARRIER toUav[2] = {
			CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_POSITION].Get(),
				D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
			CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_VELOCITY].Get(),
				D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
		};
		commandList->ResourceBarrier(2, toUav);

		DX_API("Failed to close command list (UploadParticles)")
			commandList->Close();

		// then we make the command queue execute the command list, or rather, an array of command lists,
		// which is only 1 command list in our case
		ID3D12CommandList* commandLists[] = { commandList.Get() };
		commandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);

		WaitForPreviousFrame(); // sync before moving on
	}

	void LoadCompute() {
		UINT descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

		// GPU descriptor handles for the three UAV table ranges; stored as members so ComputePass
		// can reach them after LoadCompute returns.
		particleFieldsHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(
			descriptorHeap->GetGPUDescriptorHandleForHeapStart(), 3, descriptorSize);  // slot 3  = start of particle field UAVs (u0..u6)
		gridHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(
			descriptorHeap->GetGPUDescriptorHandleForHeapStart(), 10, descriptorSize); // slot 10 = start of grid UAVs (u7..u8)
		sortedFieldsHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(
			descriptorHeap->GetGPUDescriptorHandleForHeapStart(), 12, descriptorSize); // slot 12 = start of sorted field UAVs (u9..u15)

		D3D12_GPU_VIRTUAL_ADDRESS cbv = computeCb.GetGPUVirtualAddress();

		// --- particle-only category: CBV(b0), DescriptorTable(UAV(u0..u6)) ---

		predictShader.createResources(device.Get(), "Shaders/predictCS.cso");
		predictShader.cbvAddress    = cbv;
		predictShader.tableBindings = { {1, particleFieldsHandle} };
		predictShader.inputs        = { particleFields[PF_POSITION].Get(), particleFields[PF_VELOCITY].Get() };
		predictShader.outputs       = { particleFields[PF_VELOCITY].Get(), particleFields[PF_PREDICTED_POSITION].Get() };

		collisionShader.createResources(device.Get(), "Shaders/collisionCS.cso");
		collisionShader.cbvAddress    = cbv;
		collisionShader.tableBindings = { {1, particleFieldsHandle} };
		collisionShader.inputs        = { particleFields[PF_PREDICTED_POSITION].Get(), particleFields[PF_VELOCITY].Get() };
		collisionShader.outputs       = { particleFields[PF_PREDICTED_POSITION].Get(), particleFields[PF_VELOCITY].Get() };

		positionFromScratchShader.createResources(device.Get(), "Shaders/positionFromScratchCS.cso");
		positionFromScratchShader.cbvAddress    = cbv;
		positionFromScratchShader.tableBindings = { {1, particleFieldsHandle} };
		positionFromScratchShader.inputs        = { particleFields[PF_SCRATCH].Get() };
		positionFromScratchShader.outputs       = { particleFields[PF_PREDICTED_POSITION].Get() };

		updateVelocityShader.createResources(device.Get(), "Shaders/updateVelocityCS.cso");
		updateVelocityShader.cbvAddress    = cbv;
		updateVelocityShader.tableBindings = { {1, particleFieldsHandle} };
		updateVelocityShader.inputs        = { particleFields[PF_POSITION].Get(), particleFields[PF_PREDICTED_POSITION].Get() };
		updateVelocityShader.outputs       = { particleFields[PF_VELOCITY].Get() };

		velocityFromScratchShader.createResources(device.Get(), "Shaders/velocityFromScratchCS.cso");
		velocityFromScratchShader.cbvAddress    = cbv;
		velocityFromScratchShader.tableBindings = { {1, particleFieldsHandle} };
		velocityFromScratchShader.inputs        = { particleFields[PF_SCRATCH].Get() };
		velocityFromScratchShader.outputs       = { particleFields[PF_VELOCITY].Get() };

		updatePositionShader.createResources(device.Get(), "Shaders/updatePositionCS.cso");
		updatePositionShader.cbvAddress    = cbv;
		updatePositionShader.tableBindings = { {1, particleFieldsHandle} };
		updatePositionShader.inputs        = { particleFields[PF_PREDICTED_POSITION].Get() };
		updatePositionShader.outputs       = { particleFields[PF_POSITION].Get() };

		// --- grid-only category: CBV(b0), DescriptorTable(UAV(u7..u8)) ---

		clearGridShader.createResources(device.Get(), "Shaders/clearGridCS.cso");
		clearGridShader.cbvAddress    = cbv;
		clearGridShader.tableBindings = { {1, gridHandle} };
		clearGridShader.inputs        = {};
		clearGridShader.outputs       = { cellCountBuffer.Get() };

		prefixSumShader.createResources(device.Get(), "Shaders/prefixSumCS.cso");
		prefixSumShader.cbvAddress    = cbv;
		prefixSumShader.tableBindings = { {1, gridHandle} };
		prefixSumShader.inputs        = { cellCountBuffer.Get() };
		prefixSumShader.outputs       = { cellPrefixSumBuffer.Get() };

		// --- particle + grid category: CBV(b0), DescriptorTable(UAV(u0..u6)), DescriptorTable(UAV(u7..u8)) ---

		countGridShader.createResources(device.Get(), "Shaders/countGridCS.cso");
		countGridShader.cbvAddress    = cbv;
		countGridShader.tableBindings = { {1, particleFieldsHandle}, {2, gridHandle} };
		countGridShader.inputs        = { particleFields[PF_PREDICTED_POSITION].Get() };
		countGridShader.outputs       = { cellCountBuffer.Get() };

		lambdaShader.createResources(device.Get(), "Shaders/lambdaCS.cso");
		lambdaShader.cbvAddress    = cbv;
		lambdaShader.tableBindings = { {1, particleFieldsHandle}, {2, gridHandle} };
		lambdaShader.inputs        = { particleFields[PF_PREDICTED_POSITION].Get(), cellCountBuffer.Get(), cellPrefixSumBuffer.Get() };
		lambdaShader.outputs       = { particleFields[PF_LAMBDA].Get(), particleFields[PF_DENSITY].Get() };

		deltaShader.createResources(device.Get(), "Shaders/deltaCS.cso");
		deltaShader.cbvAddress    = cbv;
		deltaShader.tableBindings = { {1, particleFieldsHandle}, {2, gridHandle} };
		deltaShader.inputs        = { particleFields[PF_PREDICTED_POSITION].Get(), particleFields[PF_LAMBDA].Get(), cellCountBuffer.Get(), cellPrefixSumBuffer.Get() };
		deltaShader.outputs       = { particleFields[PF_SCRATCH].Get() };

		vorticityShader.createResources(device.Get(), "Shaders/vorticityCS.cso");
		vorticityShader.cbvAddress    = cbv;
		vorticityShader.tableBindings = { {1, particleFieldsHandle}, {2, gridHandle} };
		vorticityShader.inputs        = { particleFields[PF_POSITION].Get(), particleFields[PF_VELOCITY].Get(), cellCountBuffer.Get(), cellPrefixSumBuffer.Get() };
		vorticityShader.outputs       = { particleFields[PF_OMEGA].Get() };

		confinementShader.createResources(device.Get(), "Shaders/confinementCS.cso");
		confinementShader.cbvAddress    = cbv;
		confinementShader.tableBindings = { {1, particleFieldsHandle}, {2, gridHandle} };
		confinementShader.inputs        = { particleFields[PF_POSITION].Get(), particleFields[PF_OMEGA].Get(), particleFields[PF_VELOCITY].Get(), cellCountBuffer.Get(), cellPrefixSumBuffer.Get() };
		confinementShader.outputs       = { particleFields[PF_VELOCITY].Get() };

		viscosityShader.createResources(device.Get(), "Shaders/viscosityCS.cso");
		viscosityShader.cbvAddress    = cbv;
		viscosityShader.tableBindings = { {1, particleFieldsHandle}, {2, gridHandle} };
		viscosityShader.inputs        = { particleFields[PF_POSITION].Get(), particleFields[PF_VELOCITY].Get(), cellCountBuffer.Get(), cellPrefixSumBuffer.Get() };
		viscosityShader.outputs       = { particleFields[PF_SCRATCH].Get() };

		// --- all-three category: CBV(b0), DescriptorTable(UAV(u0..u6)), DescriptorTable(UAV(u7..u8)), DescriptorTable(UAV(u9..u15)) ---

		sortShader.createResources(device.Get(), "Shaders/sortCS.cso");
		sortShader.cbvAddress    = cbv;
		sortShader.tableBindings = { {1, particleFieldsHandle}, {2, gridHandle}, {3, sortedFieldsHandle} };
		sortShader.inputs = {
			particleFields[PF_POSITION].Get(), particleFields[PF_VELOCITY].Get(),
			particleFields[PF_PREDICTED_POSITION].Get(), particleFields[PF_LAMBDA].Get(),
			particleFields[PF_DENSITY].Get(), particleFields[PF_OMEGA].Get(),
			particleFields[PF_SCRATCH].Get(), cellPrefixSumBuffer.Get(), cellCountBuffer.Get()
		};
		sortShader.outputs = {
			sortedFields[PF_POSITION].Get(), sortedFields[PF_VELOCITY].Get(),
			sortedFields[PF_PREDICTED_POSITION].Get(), sortedFields[PF_LAMBDA].Get(),
			sortedFields[PF_DENSITY].Get(), sortedFields[PF_OMEGA].Get(),
			sortedFields[PF_SCRATCH].Get(), cellCountBuffer.Get()
		};
	}

	void LoadParticles() {
		ParticleInitData initData = GenerateParticles(); // generate initial particle positions and zero velocities
		UploadParticles(initData); // copy particle data from CPU to the GPU default heap buffers

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
		// bind the particle SRV table (slots 1-2 in srvHeap) to root parameter 1 so the VS
		// can read position (t0) and density (t1).
		// SetSrvHeap's third argument is a raw byte offset into the heap, not a descriptor slot index,
		// so we must multiply the slot index by the descriptor increment size to get the correct byte offset
		UINT descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		material->SetSrvHeap(1, descriptorHeap, 1 * descriptorSize); // slot 1 = start of position+density SRV table

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
		commandList->SetDescriptorHeaps(1, descriptorHeap.GetAddressOf());
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
		// ceil(numParticles / 256) groups cover all particles; the shader discards extra threads
		UINT numGroups = (numParticles + 255) / 256;

		// prediction: apply gravity, compute p* = x + v*dt
		predictShader.dispatch_then_barrier(commandList.Get(), numGroups);

		// pre-stabilization (Koster & Kruger 2016): clamp predicted positions to the
		// simulation box and zero wall-normal velocity before building the grid.
		collisionShader.dispatch_then_barrier(commandList.Get(), numGroups);

		// build the spatial grid from predicted positions, then sort particles into grid order
		UINT numCellGroups = (numCells + 255) / 256;

		// step 1: zero the cell counts
		clearGridShader.dispatch_then_barrier(commandList.Get(), numCellGroups);

		// step 2: count particles per cell (each particle does InterlockedAdd on its cell)
		countGridShader.dispatch_then_barrier(commandList.Get(), numGroups);

		// step 3: exclusive prefix sum of cellCount -> cellPrefixSum
		// tells us where each cell's particle run starts in the sorted buffer
		prefixSumShader.dispatch_then_barrier(commandList.Get(), 1);

		// step 4: zero cell counts again so sortCS can use them as per-cell atomic counters
		clearGridShader.dispatch_then_barrier(commandList.Get(), numCellGroups);

		// step 5: each particle scatters all its field data into sorted order
		sortShader.dispatch_then_barrier(commandList.Get(), numGroups);

		// step 6: copy sorted data back into the main particle field buffers
		{
			D3D12_RESOURCE_BARRIER barriers[PF_COUNT * 2];
			for (UINT f = 0; f < PF_COUNT; f++) {
				barriers[f * 2] = CD3DX12_RESOURCE_BARRIER::Transition(
					sortedFields[f].Get(),
					D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
				barriers[f * 2 + 1] = CD3DX12_RESOURCE_BARRIER::Transition(
					particleFields[f].Get(),
					D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
			}
			commandList->ResourceBarrier(PF_COUNT * 2, barriers);

			for (UINT f = 0; f < PF_COUNT; f++)
				commandList->CopyBufferRegion(particleFields[f].Get(), 0,
					sortedFields[f].Get(), 0, (UINT64)numParticles * fieldStrides[f]);

			for (UINT f = 0; f < PF_COUNT; f++) {
				barriers[f * 2] = CD3DX12_RESOURCE_BARRIER::Transition(
					sortedFields[f].Get(),
					D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
				barriers[f * 2 + 1] = CD3DX12_RESOURCE_BARRIER::Transition(
					particleFields[f].Get(),
					D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
			}
			commandList->ResourceBarrier(PF_COUNT * 2, barriers);
		}

		// constraint solver loop
		// particles are now in grid-sorted order, and cellCount + cellPrefixSum describe
		// exactly where each cell's particles live in the buffer, so neighbor lookups
		// use simple arithmetic: particles[cellPrefixSum[ci] + s] for s in [0, cellCount[ci])
		for (int iter = 0; iter < solverIterations; iter++) {
			lambdaShader.dispatch_then_barrier(commandList.Get(), numGroups);              // compute lambda and density
			deltaShader.dispatch_then_barrier(commandList.Get(), numGroups);               // delta_p -> scratch
			positionFromScratchShader.dispatch_then_barrier(commandList.Get(), numGroups); // scratch -> predictedPosition
			collisionShader.dispatch_then_barrier(commandList.Get(), numGroups);           // clamp to box
		}

		updateVelocityShader.dispatch_then_barrier(commandList.Get(), numGroups);    // v = (p* - x) / dt
		vorticityShader.dispatch_then_barrier(commandList.Get(), numGroups);         // estimate curl(v) -> omega
		confinementShader.dispatch_then_barrier(commandList.Get(), numGroups);       // vorticity confinement -> velocity
		viscosityShader.dispatch_then_barrier(commandList.Get(), numGroups);         // XSPH viscosity -> scratch
		velocityFromScratchShader.dispatch_then_barrier(commandList.Get(), numGroups); // scratch -> velocity
		updatePositionShader.dispatch_then_barrier(commandList.Get(), numGroups);    // position = predictedPosition

		// Copy density data to readback buffer for CPU access next frame
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
			particleFields[PF_DENSITY].Get(),
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE));
		commandList->CopyBufferRegion(densityReadbackBuffer.Get(), 0,
			particleFields[PF_DENSITY].Get(), 0, (UINT64)numParticles * sizeof(float));
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
			particleFields[PF_DENSITY].Get(),
			D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
	}

	void GraphicsPass() {
		backgroundMesh->Draw(commandList.Get()); // draw skybox at the back first

		// The particle field buffers' home state is UNORDERED_ACCESS, because that's what the compute
		// shaders need. The vertex shader reads position and density as SRVs, which requires the
		// resources to be in a shader-resource state. We transition in before the draw and back out
		// to UAV afterward, so the rest of the pipeline always sees the home state.
		D3D12_RESOURCE_BARRIER toSrv[2] = {
			CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_POSITION].Get(),
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
				D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
			CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_DENSITY].Get(),
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
				D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
		};
		commandList->ResourceBarrier(2, toSrv);

		particleMesh->Draw(commandList.Get()); // draw particles on top

		D3D12_RESOURCE_BARRIER toUav[2] = {
			CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_POSITION].Get(),
				D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
			CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_DENSITY].Get(),
				D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
		};
		commandList->ResourceBarrier(2, toUav);
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
		ImGui::InputFloat("Epsilon (relaxation) [5.0]", &epsilon, 0.5f, 1.0f, "%.2f");
		ImGui::InputFloat("Viscosity (XSPH) [0.01]", &viscosity, 0.001f, 0.01f, "%.4f");
		ImGui::InputFloat("Artificial pressure [0.05]", &sCorrK, 0.005f, 0.05f, "%.4f");
		ImGui::InputFloat("Vorticity epsilon [0.01]", &vorticityEpsilon, 0.001f, 0.01f, "%.4f");
		ImGui::Checkbox("Fountain", &fountainEnabled);
		ImGui::PopItemWidth(); // restore default width for any subsequent widgets
		// show derived values as read-only text for reference
		ImGui::Separator(); // horizontal line to separate tunable parameters from derived values
		ImGui::Text("%d particles, %u cells, rho0 = %.2f", numParticles, gridDim*gridDim*gridDim, rho0);
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
		computeCb->fountainEnabled = fountainEnabled ? 1 : 0;
		computeCb.Upload();
	}

	virtual void Update(float dt, float T) override {
		dt = std::min(dt, 1.0f / 30.0f); // cap at 33ms: prevents energy spikes on window drag or stutter
		camera->Animate(dt); // update camera position and orientation based on user input

		UpdateExternalForce();

		UpdatePerFrameCb();

		UpdateComputeCb(dt);


		// Read back previous frame's particle data (GPU is done after WaitForPreviousFrame)
		if (physicsRunning) { CalculateAvgDensity(); }
	}

	void CalculateAvgDensity() {
		// map the readback buffer to CPU memory and copy the density data into a vector
		const UINT64 bufferSize = numParticles * sizeof(float);
		void* pData;
		CD3DX12_RANGE readRange(0, bufferSize);
		if (SUCCEEDED(densityReadbackBuffer->Map(0, &readRange, &pData))) {
			memcpy(densityReadbackData.data(), pData, bufferSize);
			CD3DX12_RANGE writeRange(0, 0);
			densityReadbackBuffer->Unmap(0, &writeRange);
		}

		// Compute average density from readback data
		double densitySum = 0.0;
		for (int i = 0; i < numParticles; i++)
			densitySum += densityReadbackData[i];
		avgDensity = static_cast<float>(densitySum / numParticles);
	}

	void CreateParticleBuffers() {
		// create one buffer per particle field on the default heap (GPU-local, writable by compute shaders via UAV)
		const CD3DX12_HEAP_PROPERTIES defaultHeapProps(D3D12_HEAP_TYPE_DEFAULT);
		for (UINT f = 0; f < PF_COUNT; f++) {
			const UINT64 bufferSize = (UINT64)numParticles * fieldStrides[f];
			const CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(
				bufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
			DX_API("Failed to create particle field buffer")
				device->CreateCommittedResource(
					&defaultHeapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc,
					D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
					IID_PPV_ARGS(particleFields[f].ReleaseAndGetAddressOf()));
			std::wstring name = std::wstring(fieldNames[f]) + L" Buffer";
			particleFields[f]->SetName(name.c_str());
		}
	}

	void CreateUploadBuffers() {
		const CD3DX12_HEAP_PROPERTIES uploadHeapProps(D3D12_HEAP_TYPE_UPLOAD);

		// position upload buffer
		const UINT64 posSize = (UINT64)numParticles * sizeof(Float3);
		const CD3DX12_RESOURCE_DESC posDesc = CD3DX12_RESOURCE_DESC::Buffer(posSize);
		DX_API("Failed to create position upload buffer")
			device->CreateCommittedResource(
				&uploadHeapProps, D3D12_HEAP_FLAG_NONE, &posDesc,
				D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
				IID_PPV_ARGS(positionUploadBuffer.ReleaseAndGetAddressOf()));
		positionUploadBuffer->SetName(L"Position Upload Buffer");

		// velocity upload buffer
		const UINT64 velSize = (UINT64)numParticles * sizeof(Float3);
		const CD3DX12_RESOURCE_DESC velDesc = CD3DX12_RESOURCE_DESC::Buffer(velSize);
		DX_API("Failed to create velocity upload buffer")
			device->CreateCommittedResource(
				&uploadHeapProps, D3D12_HEAP_FLAG_NONE, &velDesc,
				D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
				IID_PPV_ARGS(velocityUploadBuffer.ReleaseAndGetAddressOf()));
		velocityUploadBuffer->SetName(L"Velocity Upload Buffer");
	}

	void CreateParticleSrvs() {
		UINT descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

		// position SRV (slot 1, t0): read-only view for the vertex shader
		D3D12_SHADER_RESOURCE_VIEW_DESC posSrvDesc = {};
		posSrvDesc.Format = DXGI_FORMAT_UNKNOWN;
		posSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		posSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		posSrvDesc.Buffer.FirstElement = 0;
		posSrvDesc.Buffer.NumElements = numParticles;
		posSrvDesc.Buffer.StructureByteStride = sizeof(Float3);
		posSrvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
		CD3DX12_CPU_DESCRIPTOR_HANDLE posSrvHandle(
			descriptorHeap->GetCPUDescriptorHandleForHeapStart(), 1, descriptorSize);
		device->CreateShaderResourceView(particleFields[PF_POSITION].Get(), &posSrvDesc, posSrvHandle);

		// density SRV (slot 2, t1): read-only view for the vertex shader
		D3D12_SHADER_RESOURCE_VIEW_DESC denSrvDesc = {};
		denSrvDesc.Format = DXGI_FORMAT_UNKNOWN;
		denSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		denSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		denSrvDesc.Buffer.FirstElement = 0;
		denSrvDesc.Buffer.NumElements = numParticles;
		denSrvDesc.Buffer.StructureByteStride = sizeof(float);
		denSrvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
		CD3DX12_CPU_DESCRIPTOR_HANDLE denSrvHandle(
			descriptorHeap->GetCPUDescriptorHandleForHeapStart(), 2, descriptorSize);
		device->CreateShaderResourceView(particleFields[PF_DENSITY].Get(), &denSrvDesc, denSrvHandle);
	}

	void CreateParticleUavs() {
		// create one UAV per particle field at slots 3..9 (u0..u6)
		UINT descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		for (UINT f = 0; f < PF_COUNT; f++) {
			D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
			uavDesc.Format = DXGI_FORMAT_UNKNOWN;
			uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
			uavDesc.Buffer.FirstElement = 0;
			uavDesc.Buffer.NumElements = numParticles;
			uavDesc.Buffer.StructureByteStride = fieldStrides[f];
			uavDesc.Buffer.CounterOffsetInBytes = 0;
			uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
			CD3DX12_CPU_DESCRIPTOR_HANDLE handle(
				descriptorHeap->GetCPUDescriptorHandleForHeapStart(), 3 + f, descriptorSize);
			device->CreateUnorderedAccessView(particleFields[f].Get(), nullptr, &uavDesc, handle);
		}
	}

	void CreateGridBuffers() {
		const CD3DX12_HEAP_PROPERTIES defaultHeapProps(D3D12_HEAP_TYPE_DEFAULT);

		// cellCount buffer: one uint per cell, indicating how many particles are in each
		// cell, zeroed each frame by clearGridCS, then incremented atomically by countGridCS
		const UINT64 cellCountSize = numCells * sizeof(UINT);
		const CD3DX12_RESOURCE_DESC cellCountDesc = CD3DX12_RESOURCE_DESC::Buffer(
			cellCountSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		DX_API("Failed to create cell count buffer")
			device->CreateCommittedResource(
				&defaultHeapProps, D3D12_HEAP_FLAG_NONE, &cellCountDesc,
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
				IID_PPV_ARGS(cellCountBuffer.ReleaseAndGetAddressOf()));
		cellCountBuffer->SetName(L"Cell Count Buffer");

		// cellPrefixSumBuffer: one uint per cell, stores the exclusive prefix sum
		// of cellCount — i.e. where each cell's particles start in the sorted buffer
		const UINT64 prefixSumSize = numCells * sizeof(UINT);
		const CD3DX12_RESOURCE_DESC prefixSumDesc = CD3DX12_RESOURCE_DESC::Buffer(
			prefixSumSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		DX_API("Failed to create cell prefix sum buffer")
			device->CreateCommittedResource(
				&defaultHeapProps, D3D12_HEAP_FLAG_NONE, &prefixSumDesc,
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
				IID_PPV_ARGS(cellPrefixSumBuffer.ReleaseAndGetAddressOf()));
		cellPrefixSumBuffer->SetName(L"Cell Prefix Sum Buffer");
	}

	void CreateGridUavs() {
		UINT descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

		// cellCount UAV (slot 10, u7)
		D3D12_UNORDERED_ACCESS_VIEW_DESC cellCountUavDesc = {};
		cellCountUavDesc.Format = DXGI_FORMAT_UNKNOWN;
		cellCountUavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		cellCountUavDesc.Buffer.FirstElement = 0;
		cellCountUavDesc.Buffer.NumElements = numCells;
		cellCountUavDesc.Buffer.StructureByteStride = sizeof(UINT);
		cellCountUavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
		CD3DX12_CPU_DESCRIPTOR_HANDLE cellCountHandle(
			descriptorHeap->GetCPUDescriptorHandleForHeapStart(), 10, descriptorSize);
		device->CreateUnorderedAccessView(cellCountBuffer.Get(), nullptr, &cellCountUavDesc, cellCountHandle);

		// cellPrefixSum UAV (slot 11, u8)
		D3D12_UNORDERED_ACCESS_VIEW_DESC prefixSumUavDesc = {};
		prefixSumUavDesc.Format = DXGI_FORMAT_UNKNOWN;
		prefixSumUavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		prefixSumUavDesc.Buffer.FirstElement = 0;
		prefixSumUavDesc.Buffer.NumElements = numCells;
		prefixSumUavDesc.Buffer.StructureByteStride = sizeof(UINT);
		prefixSumUavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
		CD3DX12_CPU_DESCRIPTOR_HANDLE prefixSumHandle(
			descriptorHeap->GetCPUDescriptorHandleForHeapStart(), 11, descriptorSize);
		device->CreateUnorderedAccessView(cellPrefixSumBuffer.Get(), nullptr, &prefixSumUavDesc, prefixSumHandle);
	}

	void CreateSortBuffers() {
		// create one sorted buffer per particle field, same size and flags as the main field buffers
		const CD3DX12_HEAP_PROPERTIES defaultHeapProps(D3D12_HEAP_TYPE_DEFAULT);
		for (UINT f = 0; f < PF_COUNT; f++) {
			const UINT64 bufferSize = (UINT64)numParticles * fieldStrides[f];
			const CD3DX12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(
				bufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
			DX_API("Failed to create sorted field buffer")
				device->CreateCommittedResource(
					&defaultHeapProps, D3D12_HEAP_FLAG_NONE, &desc,
					D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
					IID_PPV_ARGS(sortedFields[f].ReleaseAndGetAddressOf()));
			std::wstring name = std::wstring(L"Sorted ") + fieldNames[f] + L" Buffer";
			sortedFields[f]->SetName(name.c_str());
		}
	}

	void CreateSortUavs() {
		// create one UAV per sorted field at slots 12..18 (u9..u15)
		UINT descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		for (UINT f = 0; f < PF_COUNT; f++) {
			D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
			uavDesc.Format = DXGI_FORMAT_UNKNOWN;
			uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
			uavDesc.Buffer.FirstElement = 0;
			uavDesc.Buffer.NumElements = numParticles;
			uavDesc.Buffer.StructureByteStride = fieldStrides[f];
			uavDesc.Buffer.CounterOffsetInBytes = 0;
			uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
			CD3DX12_CPU_DESCRIPTOR_HANDLE handle(
				descriptorHeap->GetCPUDescriptorHandleForHeapStart(), 12 + f, descriptorSize);
			device->CreateUnorderedAccessView(sortedFields[f].Get(), nullptr, &uavDesc, handle);
		}
	}

	void CreateDensityReadbackBuffer() {
		// create a readback buffer for density only (4 bytes per particle instead of 68)
		const UINT64 bufferSize = (UINT64)numParticles * sizeof(float);
		const CD3DX12_HEAP_PROPERTIES readbackHeapProps(D3D12_HEAP_TYPE_READBACK);
		const CD3DX12_RESOURCE_DESC readbackDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);
		DX_API("Failed to create density readback buffer")
			device->CreateCommittedResource(
				&readbackHeapProps, D3D12_HEAP_FLAG_NONE, &readbackDesc,
				D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
				IID_PPV_ARGS(densityReadbackBuffer.ReleaseAndGetAddressOf()));
		densityReadbackBuffer->SetName(L"Density Readback Buffer");
		densityReadbackData.resize(numParticles);
	}

	void CreateDescriptorHeap() {
		// descriptor heap layout (SoA):
		//   slot 0:     cubemap SRV (t0)           — sampled by the background pixel shader
		//   slot 1:     position SRV (t0)          — read by the particle vertex shader
		//   slot 2:     density SRV (t1)           — read by the particle vertex shader
		//   slots 3-9:  particle field UAVs (u0..u6) — compute shader read/write
		//   slot 10:    cellCount UAV (u7)          — per-cell particle count for the spatial grid
		//   slot 11:    cellPrefixSum UAV (u8)      — exclusive prefix sum for sort offsets and neighbor lookups
		//   slots 12-18: sorted field UAVs (u9..u15) — scatter targets for spatial sorting
		D3D12_DESCRIPTOR_HEAP_DESC dhd;
		dhd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE; // GPU can see these descriptors
		dhd.NodeMask = 0; // single GPU setup, so no node masking needed
		dhd.NumDescriptors = 19;
		dhd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; // heap type that holds CBVs, SRVs, and UAVs

		DX_API("Failed to create descriptor heap")
			device->CreateDescriptorHeap(&dhd, IID_PPV_ARGS(descriptorHeap.GetAddressOf()));
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
		camera->SetView(Float3(0.0f, 5.0f, -20.0f), Float3(0.0f, 0.0f, 1.0f)); // start further back to see the full box
		camera->SetSpeed(10.0f); // movement speed of the camera

		CreateImGuiDescriptorHeap();

		CreateDescriptorHeap();

		CreateParticleBuffers();

		CreateUploadBuffers();

		CreateGridBuffers();

		CreateParticleSrvs();

		CreateParticleUavs();

		CreateGridUavs();

		CreateSortBuffers();

		CreateSortUavs();

		CreateDensityReadbackBuffer();
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
