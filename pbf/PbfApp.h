#pragma once

#include <algorithm>
#include "AsyncComputeApp.h"
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
#include "SolidObstacle.h"
#include <immintrin.h>
#include <thread>

using namespace Egg::Math;

// PBF implementation: based on the Macklin & Muller 2013 nvidia research article (M&M)
// AsyncComputeApp gives us:
// graphics command allocator and command list
// compute command queue, allocator, and command list
// depth stencil buffer
// PSO manager
// per-queue frame counters (graphicsFrame, computeFrame) and fences (graphicsFence, computeFence)
// sync helpers: cpuWaitForGraphics, cpuWaitForCompute, graphicsWaitForCompute
// note: we're not using graphicsFrame and computeFrame, instead, we use a single frame counter,
// because the two queues are currently in lockstep (this might change later if we want a different
// display and physics frequency)
//
// main calls app->Run() in an infinite loop, which calculates elapsedTime and deltaTime,
// and calls Update (overridden) and Render (overridden), in that order.
//
// Initialization flow:
// main.cpp first creates the window handle, command queue, d3d12 device and swap chain,
// then initializes the pbf App by calling these methods: 
//	(which in turn call the methods indented under them like so)
//
// SetDevice / SetCommandQueue / SetSwapChain
// CreateSwapChainResources
// CreateResources                     -- allocate ALL GPU memory and ALL descriptors
//   AsyncComputeApp::CreateResources  -- cmdAllocator, cmdList, computeQueue/Allocator/List, fences
//   constant buffers + camera
//   CreateParticleBuffers             -- 7 default-heap UAV buffers
//   CreateUploadBuffers               -- 2 upload-heap staging buffers (position, velocity)
//   CreateGridBuffers                 -- cellCount + cellPrefixSum
//   CreateSortBuffers                 -- 7 sorted field buffers
//   CreateDensityReadbackBuffer       -- readback buffer
//   CreateTextureResources            -- envTexture + SolidObstacle (mesh, SDF allocation)
//   CreateImGuiDescriptorHeap         -- 1-slot SRV heap for ImGui font
//   CreateDescriptorHeap              -- 25-slot main descriptor heap
//   CreateAllDescriptors              -- ALL 25 descriptor slots filled here
//     slot 0: cubemap SRV, slots 1-2: particle SRVs, slots 3-9: particle UAVs,
//     slots 10-11: grid UAVs, slots 12-18: sort UAVs, slot 19: SDF SRV,
//     slot 20: perm UAV, slots 21-24: snapshot SRVs
//   CacheDescriptorHandles            -- GPU handles for compute shader binding
// LoadAssets                          -- upload data to GPU and build pipelines
//   UploadAll                         -- single batched command list: cubemap + particles + SDF
//   BuildGraphicsPipelines            -- background mesh, particle mesh, solid transform
//   BuildComputePipelines             -- all compute shader PSOs
// InitImGui
class PbfApp : public AsyncComputeApp {
protected:
	// Fixed particle and grid constants.
	const int particlesX = 80, particlesY = 30, particlesZ = 80; // number of particles along each axis of the initial grid
	const int offsetX = 0, offsetY = 9, offsetZ = 0; // world space offset of the center of the initial particle grid
	const int numParticles = particlesX * particlesY * particlesZ; // total number of particles in the simulation	
	// particleSpacing and hMultiplier are constants that define the SPH kernel width h,
	// which gives a lower bound to the spatial grid's cell width. We can use (try using...)
	// Morton codes to index the cells, which works best with a cubic simulation space that
	// has a power of two number of cells along each axis. Fixing h (= particleSpacing * hMultiplier) 
	// lets us define the box as exactly gridDim * h on each axis, giving a perfectly aligned 
	// cubic grid with a dense Morton code space.
	const float particleSpacing = 0.25f; // inter-particle distance (also determines rest density and display size)
	const float particleRadius = particleSpacing * 0.4f;
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
	const Float3 gridMin = Float3(-boxExtent / 2.0f, -boxExtent / 2.0f, -boxExtent / 2.0f); // most negative point of the grid
	const Float3 gridMax = Float3( boxExtent / 2.0f,  boxExtent / 2.0f,  boxExtent / 2.0f); // most positive point of the grid
	Float3 boxMin = Float3(-10.0f, gridMin.y, -10.0f); // adjustable collision boundary
	Float3 boxMax = Float3(10.0f, gridMax.y, 10.0f); // adjustable collision boundary

	// parameters that are tunable via ImGui each frame
	int solverIterations = 4; // how many newton steps to take per frame
	float epsilon = 5.0f; // constraint force mixing relaxation parameter, higher value = softer constraints
	float viscosity = 0.01f; // // XSPH viscosity coefficient, higher value = "thicker" fluid, M&M: 0.01
	// artificial purely repulsive pressure term reduces clumping while leaving room for surface tension, 
	float sCorrK = 0.02f; // artificial pressure magnitude coefficient M&M: 0.1
	float vorticityEpsilon = 0.01f; // vorticity confinement strength M&M: 0.01
	float adhesion = 0.05f; // tangential velocity damping on wall contact (0 = frictionless, 1 = full stop)
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
	com_ptr<ID3D12Resource> permBuffer; // default heap: uint per particle, maps old index -> sorted index (computed by sortCS, applied by permutateCS)

	// One ComputeShader per pass. Each holds its own PSO, root signature, descriptor
	// table bindings, and input/output resource lists for UAV barrier insertion.
	// GPU descriptor handles for the descriptor table ranges, computed once in CacheDescriptorHandles.
	CD3DX12_GPU_DESCRIPTOR_HANDLE particleFieldsHandle; // UAV(u0..u6): particle field buffers
	CD3DX12_GPU_DESCRIPTOR_HANDLE gridHandle; // UAV(u7..u8): cellCount, cellPrefixSum
	CD3DX12_GPU_DESCRIPTOR_HANDLE sortedFieldsHandle; // UAV(u9..u15): sorted particle field buffers
	CD3DX12_GPU_DESCRIPTOR_HANDLE permHandle; // UAV(u16): permutation buffer (slot 20)
	ComputeShader::P applyForcesShader;  // apply gravity and external forces to velocity
	ComputeShader::P collisionVelocityShader;  // zero wall-directed velocity, apply adhesion
	ComputeShader::P predictPositionShader; // p* = position + velocity * dt
	ComputeShader::P collisionPredictedPositionShader; // clamp p* to box; runs pre-sort and every solver iteration
	ComputeShader::P clearGridShader; // zero cellCount array
	ComputeShader::P countGridShader; // count particles per cell (first grid-build pass)
	ComputeShader::P prefixSumShader; // exclusive prefix sum of cellCount -> cellPrefixSum
	ComputeShader::P sortShader; // each particle computes its sorted destination index -> perm[]
	ComputeShader::P permutateShader; // applies perm[] to scatter all particle fields to sorted positions
	ComputeShader::P lambdaShader; // compute lambda per particle
	ComputeShader::P deltaShader; // compute delta_p, write to scratch
	ComputeShader::P positionFromScratchShader; // copy scratch -> predictedPosition (Jacobi commit during solver loop)
	ComputeShader::P updateVelocityShader;// update velocity from displacement: v = (p* - x) / dt
	ComputeShader::P vorticityShader; // estimate per-particle vorticity (curl of velocity), store in omega
	ComputeShader::P confinementShader; // apply vorticity confinement force to velocity
	ComputeShader::P viscosityShader;  // XSPH velocity smoothing, writes to scratch
	ComputeShader::P velocityFromScratchShader; // copy scratch -> velocity (Jacobi commit after viscosity)
	ComputeShader::P updatePositionShader; // update position from predictedPosition (final step per paper)
	// Solid obstacle: owns the renderable mesh and the SDF volume texture
	SolidObstacle::P solidObstacle;
	CD3DX12_GPU_DESCRIPTOR_HANDLE sdfHandle; // GPU handle for descriptor heap slot 19: SDF Texture3D SRV
	Float3 solidPosition = Float3(0.0f, -13.0f, 0.0f); // world-space translation, driven by ImGui
	Float3 solidEulerDeg = Float3(0.0f, 30.0f, 0.0f); // XYZ Euler rotation in degrees, driven by ImGui
	float  solidScale = 2.0f; // uniform scale, driven by ImGui
	

	// Readback buffer for density (readback heap, CPU-readable after CopyBufferRegion).
	// Serves as an example for reading particle data back from the GPU
	com_ptr<ID3D12Resource> densityReadbackBuffer;
	std::vector<float> densityReadbackData;
	float avgDensity = 0.0f; // average particle density from previous frame's readback
	using clock = std::chrono::high_resolution_clock;
	clock::time_point lastFrame; // tracks accumulated time toward next physics step
	std::chrono::duration<double> targetPeriod{ 1.0 / 30.0 };
	clock::time_point t0, t1; // debug timer variabeles
	float debugTimer = 0.0f;
	float lastDt = 0.0f; // dt from last Update(), consumed by Render() for compute CB upload after GPU sync

	uint64_t frameCount = 0;

	bool fpsCapped = false;
	bool physicsRunning = false; // toggled by spacebar: when false, compute passes are skipped each frame
	
	bool arrowLeft = false, arrowRight = false, arrowUp = false, arrowDown = false; // arrow key held state for box translation

	// Async compute: physics runs on the compute queue (from AsyncComputeApp), decoupled from vsync.
	// Double-buffered snapshot buffers hold position and density for the graphics queue.
	// snapshotWriteIdx is the slot currently being written by the compute queue;
	// the graphics queue always reads from the OTHER slot (1 - snapshotWriteIdx).
	com_ptr<ID3D12Resource> snapshotPosition[2]; // position snapshot double-buffer (COMMON state, written by compute, read by direct)
	com_ptr<ID3D12Resource> snapshotDensity[2];  // density snapshot double-buffer (COMMON state, written by compute, read by direct)
	int snapshotWriteIdx = 0; // snapshot slot being written by compute, 0 vs 1: graphics reads (1 - snapshotWriteIdx)

	// Allocate GPU resources for the cubemap texture and the solid obstacle (mesh + SDF).
	// No GPU commands are recorded here; uploads happen later in UploadAll().
	void CreateTextureResources() {
		// ImportTextureCube reads the dds file, and creates the two GPU resources for the texture cube:
		// 1. the default heap resource, which has 6 array slices for the cube map faces and is GPU-local (fast)
		// 2. the upload heap resource, which the CPU can write to, and is used for staging
		// the texture data is also copied to the upload heap by this function call, but not yet transferred to
		// the default heap until we call UploadAll(), as this step requires the command queue
		envTexture = Egg::Importer::ImportTextureCube(device.Get(), "../Media/cloudyNoon.dds");

		// Create the solid obstacle: loads mesh geometry, shaders, material, constant buffer,
		// reads the SDF file and allocates the SDF Texture3D + upload buffer. No GPU commands.
		solidObstacle = SolidObstacle::Create();
		solidObstacle->Load(device.Get(), psoManager, "dragonite.obj", "dragonite.sdf", perFrameCb);
	}

	// Build the background skybox rendering pipeline (shaders, material, mesh).
	// Called after all resources and descriptors are ready.
	void BuildBackgroundPipeline() {
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

	// Map the upload buffers and copy initial particle data (positions + velocities) from CPU memory.
	// This is a CPU-side operation; the actual GPU transfer is recorded by RecordParticleUpload().
	void FillUploadBuffers(const ParticleInitData& initData) {
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
	}

	// Record copy commands for particle data into the already-open command list.
	// The command list must have been Reset() before calling this.
	void RecordParticleUpload() {
		// In order to copy to them, we must transition position and velocity buffers to COPY_DEST
		// That's done by inserting transition type resource barriers to the command list.
		D3D12_RESOURCE_BARRIER toCopyDest[2] = {
			CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_POSITION].Get(),
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST),
			CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_VELOCITY].Get(),
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST),
		};
		commandList->ResourceBarrier(2, toCopyDest);
		// copy commands: dst, offset, src, offset
		commandList->CopyBufferRegion(particleFields[PF_POSITION].Get(), 0,
			positionUploadBuffer.Get(), 0, numParticles * sizeof(Float3));
		commandList->CopyBufferRegion(particleFields[PF_VELOCITY].Get(), 0,
			velocityUploadBuffer.Get(), 0, numParticles * sizeof(Float3));

		// transition back to UAV
		D3D12_RESOURCE_BARRIER toUav[2] = {
			CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_POSITION].Get(),
				D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
			CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_VELOCITY].Get(),
				D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
		};
		commandList->ResourceBarrier(2, toUav);
	}


	// Create all 16 compute shader PSOs and their descriptor table bindings.
	// Requires CacheDescriptorHandles() to have been called first.
	void BuildComputePipelines() {
		D3D12_GPU_VIRTUAL_ADDRESS cbv = computeCb.GetGPUVirtualAddress();

		// aliases for a tiny bit of verbosity
		using std::vector;
		using TableBinding = ComputeShader::TableBinding;

		applyForcesShader = ComputeShader::Create(device.Get(), "Shaders/applyForcesCS.cso", cbv,
			vector<TableBinding>{ {1, particleFieldsHandle} },
			vector<ID3D12Resource*>{ particleFields[PF_POSITION].Get(), particleFields[PF_VELOCITY].Get() },
			vector<ID3D12Resource*>{ particleFields[PF_VELOCITY].Get() });

		collisionVelocityShader = ComputeShader::Create(device.Get(), "Shaders/collisionVelocityCS.cso", cbv,
			vector<TableBinding>{ {1, particleFieldsHandle}, {2, sdfHandle} },
			vector<ID3D12Resource*>{ particleFields[PF_POSITION].Get(), particleFields[PF_VELOCITY].Get() },
			vector<ID3D12Resource*>{ particleFields[PF_VELOCITY].Get() });

		predictPositionShader = ComputeShader::Create(device.Get(), "Shaders/predictPositionCS.cso", cbv,
			vector<TableBinding>{ {1, particleFieldsHandle} },
			vector<ID3D12Resource*>{ particleFields[PF_POSITION].Get(), particleFields[PF_VELOCITY].Get() },
			vector<ID3D12Resource*>{ particleFields[PF_PREDICTED_POSITION].Get() });

		collisionPredictedPositionShader = ComputeShader::Create(device.Get(), "Shaders/collisionPredictedPositionCS.cso", cbv,
			vector<TableBinding>{ {1, particleFieldsHandle}, {2, sdfHandle} },
			vector<ID3D12Resource*>{ particleFields[PF_PREDICTED_POSITION].Get() },
			vector<ID3D12Resource*>{ particleFields[PF_PREDICTED_POSITION].Get() });

		positionFromScratchShader = ComputeShader::Create(device.Get(), "Shaders/positionFromScratchCS.cso", cbv,
			vector<TableBinding>{ {1, particleFieldsHandle} },
			vector<ID3D12Resource*>{ particleFields[PF_SCRATCH].Get() },
			vector<ID3D12Resource*>{ particleFields[PF_PREDICTED_POSITION].Get() });

		updateVelocityShader = ComputeShader::Create(device.Get(), "Shaders/updateVelocityCS.cso", cbv,
			vector<TableBinding>{ {1, particleFieldsHandle} },
			vector<ID3D12Resource*>{ particleFields[PF_POSITION].Get(), particleFields[PF_PREDICTED_POSITION].Get() },
			vector<ID3D12Resource*>{ particleFields[PF_VELOCITY].Get() });

		velocityFromScratchShader = ComputeShader::Create(device.Get(), "Shaders/velocityFromScratchCS.cso", cbv,
			vector<TableBinding>{ {1, particleFieldsHandle} },
			vector<ID3D12Resource*>{ particleFields[PF_SCRATCH].Get() },
			vector<ID3D12Resource*>{ particleFields[PF_VELOCITY].Get() });

		updatePositionShader = ComputeShader::Create(device.Get(), "Shaders/updatePositionCS.cso", cbv,
			vector<TableBinding>{ {1, particleFieldsHandle} },
			vector<ID3D12Resource*>{ particleFields[PF_PREDICTED_POSITION].Get() },
			vector<ID3D12Resource*>{ particleFields[PF_POSITION].Get() });

		clearGridShader = ComputeShader::Create(device.Get(), "Shaders/clearGridCS.cso", cbv,
			vector<TableBinding>{ {1, gridHandle} },
			vector<ID3D12Resource*>{ cellCountBuffer.Get() },
			vector<ID3D12Resource*>{ cellCountBuffer.Get() });

		prefixSumShader = ComputeShader::Create(device.Get(), "Shaders/prefixSumCS.cso", cbv,
			vector<TableBinding>{ {1, gridHandle} },
			vector<ID3D12Resource*>{ cellCountBuffer.Get() },
			vector<ID3D12Resource*>{ cellPrefixSumBuffer.Get() });

		countGridShader = ComputeShader::Create(device.Get(), "Shaders/countGridCS.cso", cbv,
			vector<TableBinding>{ {1, particleFieldsHandle}, {2, gridHandle} },
			vector<ID3D12Resource*>{ particleFields[PF_PREDICTED_POSITION].Get(), cellCountBuffer.Get() },
			vector<ID3D12Resource*>{ cellCountBuffer.Get() });

		lambdaShader = ComputeShader::Create(device.Get(), "Shaders/lambdaCS.cso", cbv,
			vector<TableBinding>{ {1, particleFieldsHandle}, {2, gridHandle} },
			vector<ID3D12Resource*>{ particleFields[PF_PREDICTED_POSITION].Get(), cellCountBuffer.Get(), cellPrefixSumBuffer.Get() },
			vector<ID3D12Resource*>{ particleFields[PF_LAMBDA].Get(), particleFields[PF_DENSITY].Get() });

		deltaShader = ComputeShader::Create(device.Get(), "Shaders/deltaCS.cso", cbv,
			vector<TableBinding>{ {1, particleFieldsHandle}, {2, gridHandle} },
			vector<ID3D12Resource*>{ particleFields[PF_PREDICTED_POSITION].Get(), particleFields[PF_LAMBDA].Get(), cellCountBuffer.Get(), cellPrefixSumBuffer.Get() },
			vector<ID3D12Resource*>{ particleFields[PF_SCRATCH].Get() });

		vorticityShader = ComputeShader::Create(device.Get(), "Shaders/vorticityCS.cso", cbv,
			vector<TableBinding>{ {1, particleFieldsHandle}, {2, gridHandle} },
			vector<ID3D12Resource*>{ particleFields[PF_POSITION].Get(), particleFields[PF_VELOCITY].Get(), cellCountBuffer.Get(), cellPrefixSumBuffer.Get() },
			vector<ID3D12Resource*>{ particleFields[PF_OMEGA].Get() });

		confinementShader = ComputeShader::Create(device.Get(), "Shaders/confinementCS.cso", cbv,
			vector<TableBinding>{ {1, particleFieldsHandle}, {2, gridHandle} },
			vector<ID3D12Resource*>{ particleFields[PF_POSITION].Get(), particleFields[PF_OMEGA].Get(), particleFields[PF_VELOCITY].Get(), cellCountBuffer.Get(), cellPrefixSumBuffer.Get() },
			vector<ID3D12Resource*>{ particleFields[PF_VELOCITY].Get() });

		viscosityShader = ComputeShader::Create(device.Get(), "Shaders/viscosityCS.cso", cbv,
			vector<TableBinding>{ {1, particleFieldsHandle}, {2, gridHandle} },
			vector<ID3D12Resource*>{ particleFields[PF_POSITION].Get(), particleFields[PF_VELOCITY].Get(), cellCountBuffer.Get(), cellPrefixSumBuffer.Get() },
			vector<ID3D12Resource*>{ particleFields[PF_SCRATCH].Get() });

		sortShader = ComputeShader::Create(device.Get(), "Shaders/sortCS.cso", cbv,
			vector<TableBinding>{ {1, particleFieldsHandle}, {2, gridHandle}, {3, permHandle} },
			vector<ID3D12Resource*>{ particleFields[PF_PREDICTED_POSITION].Get(),
			  cellPrefixSumBuffer.Get(), cellCountBuffer.Get() },
			vector<ID3D12Resource*>{ permBuffer.Get(), cellCountBuffer.Get() });

		permutateShader = ComputeShader::Create(device.Get(), "Shaders/permutateCS.cso", cbv,
			vector<TableBinding>{ {1, particleFieldsHandle}, {2, sortedFieldsHandle}, {3, permHandle} },
			vector<ID3D12Resource*>{ particleFields[PF_POSITION].Get(), particleFields[PF_VELOCITY].Get(),
			  particleFields[PF_PREDICTED_POSITION].Get(), particleFields[PF_LAMBDA].Get(),
			  particleFields[PF_DENSITY].Get(), particleFields[PF_OMEGA].Get(),
			  particleFields[PF_SCRATCH].Get(), permBuffer.Get() },
			vector<ID3D12Resource*>{ sortedFields[PF_POSITION].Get(), sortedFields[PF_VELOCITY].Get(),
			  sortedFields[PF_PREDICTED_POSITION].Get(), sortedFields[PF_LAMBDA].Get(),
			  sortedFields[PF_DENSITY].Get(), sortedFields[PF_OMEGA].Get(),
			  sortedFields[PF_SCRATCH].Get() });
	}

	// Rebuild the solid's world transform from solidPosition and solidEulerDeg (XYZ Euler, degrees).
	void SetSolidTransform() {
		const float deg2rad = 3.14159265358979323846f / 180.0f;
		float rx = solidEulerDeg.x * deg2rad;
		float ry = solidEulerDeg.y * deg2rad;
		float rz = solidEulerDeg.z * deg2rad;
		Float4x4 rot =
			Float4x4::Rotation(Float3(1.0f, 0.0f, 0.0f), rx) *
			Float4x4::Rotation(Float3(0.0f, 1.0f, 0.0f), ry) *
			Float4x4::Rotation(Float3(0.0f, 0.0f, 1.0f), rz);
		solidObstacle->SetTransform(Float4x4::Scaling(Float3(solidScale, solidScale, solidScale)) * rot * Float4x4::Translation(solidPosition));
	}

	// Build the particle rendering pipeline (shaders, material, mesh).
	// Called after all resources and descriptors are ready.
	void BuildParticlePipeline() {
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

	void PrepareComputeCommandList() {
		// Reset compute allocator and command list for the next compute frame
		DX_API("Failed to reset compute allocator")
			computeAllocator->Reset();
		DX_API("Failed to reset compute list")
			computeList->Reset(computeAllocator.Get(), nullptr);

		// Bind the shared descriptor heap: must be done each time the compute list is reset
		computeList->SetDescriptorHeaps(1, descriptorHeap.GetAddressOf());
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

	void ExecuteGraphics() {
		//transition the back buffer back to "present" state so the swap chain can display it
		commandList->ResourceBarrier(1, // number of barriers
			&CD3DX12_RESOURCE_BARRIER::Transition( // helper function to create a transition barrier
				renderTargets[swapChainBackBufferIndex].Get(), // resource: the current back buffer, identified by the swap chain's current back buffer index
				D3D12_RESOURCE_STATE_RENDER_TARGET, // before: we just rendered into the back buffer
				D3D12_RESOURCE_STATE_PRESENT)); // after: we want to present the back buffer

		// close the command list, no more commands can be recorded until the next Reset()
		DX_API("Failed to close command list")
			commandList->Close();
		ID3D12CommandList* graphicsCls[] = { commandList.Get() };
		commandQueue->ExecuteCommandLists(_countof(graphicsCls), graphicsCls);

		DX_API("Failed to present swap chain")
			swapChain->Present(0, 0);
	}

	void ExecuteCompute() {
		DX_API("Failed to close compute list")
			computeList->Close();
		ID3D12CommandList* computeCls[] = { computeList.Get() };
		computeCommandQueue->ExecuteCommandLists(_countof(computeCls), computeCls);
	}

	void SortParticles() {
		// ceil(numParticles / 256) groups cover all particles; the shader discards extra threads
		UINT numGroups = (numParticles + 255) / 256;
		// ceil(numCells / 256) groups cover all cells; the shader discards extra threads
		UINT numCellGroups = (numCells + 255) / 256;

		// zero the cell count
		clearGridShader->dispatch_then_barrier(computeList.Get(), numCellGroups);

		// count particles per cell (each particle does InterlockedAdd on its cell)
		// after this call, the ith element in cellCount indicates how many particles
		// are in that cell
		countGridShader->dispatch_then_barrier(computeList.Get(), numGroups);

		// exclusive prefix sum of cellCount -> cellPrefixSum
		// tells us where each cell's particle run starts in the sorted buffer,
		// meaning that after this call, the ith element in cellPrefixSum tells
		// us where the ith cell's range begins in the particle buffer
		prefixSumShader->dispatch_then_barrier(computeList.Get(), 1);

		// zero cell counts again so sortCS can use them as per-cell atomic counters
		clearGridShader->dispatch_then_barrier(computeList.Get(), numCellGroups);

		// compute perm[i] = sorted destination index for each particle i
		sortShader->dispatch_then_barrier(computeList.Get(), numGroups);

		// scatter all particle fields to their sorted positions using perm[]
		permutateShader->dispatch_then_barrier(computeList.Get(), numGroups);

		// copy sorted data back into the main particle field buffers
		{
			// one barrier for each buffer both sorted and unsorted
			// sorted fields become copy source, unsorted fields become copy destination
			D3D12_RESOURCE_BARRIER barriers[PF_COUNT * 2];
			for (UINT f = 0; f < PF_COUNT; f++) {
				barriers[f * 2] = CD3DX12_RESOURCE_BARRIER::Transition(
					sortedFields[f].Get(),
					D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
				barriers[f * 2 + 1] = CD3DX12_RESOURCE_BARRIER::Transition(
					particleFields[f].Get(),
					D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
			}
			computeList->ResourceBarrier(PF_COUNT * 2, barriers); // insert the created barriers

			// actual copy calls here
			for (UINT f = 0; f < PF_COUNT; f++)
				computeList->CopyBufferRegion(particleFields[f].Get(), 0,
					sortedFields[f].Get(), 0, (UINT64)numParticles * fieldStrides[f]);

			// copy is done, transition back to the "default" state, which is UAV, for the next pass
			for (UINT f = 0; f < PF_COUNT; f++) {
				barriers[f * 2] = CD3DX12_RESOURCE_BARRIER::Transition(
					sortedFields[f].Get(),
					D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
				barriers[f * 2 + 1] = CD3DX12_RESOURCE_BARRIER::Transition(
					particleFields[f].Get(),
					D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
			}
			computeList->ResourceBarrier(PF_COUNT * 2, barriers); // insert the created barriers
		}
	}

	// writeIdx: which snapshot slot to write to this step (caller sets and flips).
	void RecordComputeCommands(int writeIdx) {
		// ceil(numParticles / 256) groups cover all particles; the shader discards extra threads
		UINT numGroups = (numParticles + 255) / 256;

		// apply gravity + external forces to velocity
		applyForcesShader->dispatch_then_barrier(computeList.Get(), numGroups);

		// Zero wall-directed velocity components and apply adhesion damping.
		// Must run before predictPosition so the correction survives into p*;
		// updateVelocityCS would overwrite any velocity edits made after prediction.
		collisionVelocityShader->dispatch_then_barrier(computeList.Get(), numGroups);

		// p* = position + velocity * dt  (velocity is now wall-corrected)
		predictPositionShader->dispatch_then_barrier(computeList.Get(), numGroups);

		// Clamp p* to the simulation box before building the spatial grid.
		collisionPredictedPositionShader->dispatch_then_barrier(computeList.Get(), numGroups);

		SortParticles(); // sort particle data for improved cache coherence -> fewer cache misses

		// constraint solver loop
		// particles are now in grid-sorted order, and cellCount + cellPrefixSum describe
		// exactly where each cell's particles live in the buffer, so neighbor lookups
		// use simple arithmetic: particles[cellPrefixSum[ci] + s] for s in [0, cellCount[ci])
		for (int iter = 0; iter < solverIterations; iter++) {
			lambdaShader->dispatch_then_barrier(computeList.Get(), numGroups); // compute lambda and density
			deltaShader->dispatch_then_barrier(computeList.Get(), numGroups); // delta_p -> scratch
			positionFromScratchShader->dispatch_then_barrier(computeList.Get(), numGroups); // scratch -> predictedPosition
			collisionPredictedPositionShader->dispatch_then_barrier(computeList.Get(), numGroups); // clamp to box
		}

		updateVelocityShader->dispatch_then_barrier(computeList.Get(), numGroups);    // v = (p* - x) / dt
		vorticityShader->dispatch_then_barrier(computeList.Get(), numGroups);         // estimate curl(v) -> omega
		confinementShader->dispatch_then_barrier(computeList.Get(), numGroups);       // vorticity confinement -> velocity
		viscosityShader->dispatch_then_barrier(computeList.Get(), numGroups);         // XSPH viscosity -> scratch
		velocityFromScratchShader->dispatch_then_barrier(computeList.Get(), numGroups); // scratch -> velocity
		updatePositionShader->dispatch_then_barrier(computeList.Get(), numGroups);    // position = predictedPosition

		// Write snapshot: copy position and density into snapshot slot [writeIdx].
		// Transition particle buffers to COPY_SOURCE, snapshot buffers to COPY_DEST.
		D3D12_RESOURCE_BARRIER toCopySrc[2] = { // position, density particle buffer becomes source
			CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_POSITION].Get(),
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
			CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_DENSITY].Get(),
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
		};
		D3D12_RESOURCE_BARRIER snapshotToDest[2] = { // position, density snapshot buffer becomes destination
			CD3DX12_RESOURCE_BARRIER::Transition(snapshotPosition[writeIdx].Get(),
				D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
			CD3DX12_RESOURCE_BARRIER::Transition(snapshotDensity[writeIdx].Get(),
				D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
		};
		// insert them into the command list so the GPU knows about the state changes before the copy calls
		computeList->ResourceBarrier(2, toCopySrc);
		computeList->ResourceBarrier(2, snapshotToDest);

		computeList->CopyBufferRegion(snapshotPosition[writeIdx].Get(), 0,
			particleFields[PF_POSITION].Get(), 0, (UINT64)numParticles * sizeof(Float3));
		computeList->CopyBufferRegion(snapshotDensity[writeIdx].Get(), 0,
			particleFields[PF_DENSITY].Get(), 0, (UINT64)numParticles * sizeof(float));

		// Copy density to readback buffer (CPU reads it after the next cpuWaitForCompute).
		computeList->CopyBufferRegion(densityReadbackBuffer.Get(), 0,
			particleFields[PF_DENSITY].Get(), 0, (UINT64)numParticles * sizeof(float));

		// Transition everything back to its home state.
		D3D12_RESOURCE_BARRIER toUav[2] = {
			CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_POSITION].Get(),
				D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
			CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_DENSITY].Get(),
				D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
		};
		D3D12_RESOURCE_BARRIER snapshotToCommon[2] = {
			CD3DX12_RESOURCE_BARRIER::Transition(snapshotPosition[writeIdx].Get(),
				D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON),
			CD3DX12_RESOURCE_BARRIER::Transition(snapshotDensity[writeIdx].Get(),
				D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON),
		};
		computeList->ResourceBarrier(2, toUav);
		computeList->ResourceBarrier(2, snapshotToCommon);
	}

	// readIdx: which snapshot slot the graphics queue reads from (always 1 - snapshotWriteIdx).
	void RecordGraphicsCommands(int readIdx) {
		backgroundMesh->Draw(commandList.Get()); // draw skybox at the back first
		solidObstacle->Draw(commandList.Get());  // draw solid with depth test before particles

		// Before the particle draw, redirect descriptor heap slots 1-2 to the active snapshot.
		// The particle VS fetches position (t0) and density (t1) from slots 1-2 in the heap.
		// We copy the snapshot SRVs (slots 21+readIdx and 23+readIdx) into slots 1-2 so the
		// shader reads the latest complete snapshot without any root signature change.
		UINT sz = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		device->CopyDescriptorsSimple(1,
			CD3DX12_CPU_DESCRIPTOR_HANDLE(descriptorHeap->GetCPUDescriptorHandleForHeapStart(), 1,  sz),
			CD3DX12_CPU_DESCRIPTOR_HANDLE(descriptorHeap->GetCPUDescriptorHandleForHeapStart(), 21 + readIdx, sz),
			D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		device->CopyDescriptorsSimple(1,
			CD3DX12_CPU_DESCRIPTOR_HANDLE(descriptorHeap->GetCPUDescriptorHandleForHeapStart(), 2,  sz),
			CD3DX12_CPU_DESCRIPTOR_HANDLE(descriptorHeap->GetCPUDescriptorHandleForHeapStart(), 23 + readIdx, sz),
			D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

		// Snapshot buffers live in COMMON. Transition them to SRV state for the draw, then back.
		D3D12_RESOURCE_BARRIER toSrv[2] = {
			CD3DX12_RESOURCE_BARRIER::Transition(snapshotPosition[readIdx].Get(),
				D3D12_RESOURCE_STATE_COMMON,
				D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
			CD3DX12_RESOURCE_BARRIER::Transition(snapshotDensity[readIdx].Get(),
				D3D12_RESOURCE_STATE_COMMON,
				D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
		};
		commandList->ResourceBarrier(2, toSrv);

		particleMesh->Draw(commandList.Get()); // draw particles on top

		D3D12_RESOURCE_BARRIER toCommon[2] = {
			CD3DX12_RESOURCE_BARRIER::Transition(snapshotPosition[readIdx].Get(),
				D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
				D3D12_RESOURCE_STATE_COMMON),
			CD3DX12_RESOURCE_BARRIER::Transition(snapshotDensity[readIdx].Get(),
				D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
				D3D12_RESOURCE_STATE_COMMON),
		};
		commandList->ResourceBarrier(2, toCommon);
	}

	void BuildImGui() {
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
		ImGui::InputFloat("Artificial pressure [0.02]", &sCorrK, 0.005f, 0.05f, "%.4f");
		ImGui::InputFloat("Vorticity epsilon [0.01]", &vorticityEpsilon, 0.001f, 0.01f, "%.4f");
		ImGui::InputFloat("Adhesion [0.05]", &adhesion, 0.01f, 0.1f, "%.3f");
		ImGui::Checkbox("Fountain", &fountainEnabled);
		ImGui::SameLine();
		ImGui::Checkbox("FPS cap", &fpsCapped);
		ImGui::PopItemWidth(); // restore default width for any subsequent widgets
		// show derived values as read-only text for reference
		ImGui::Separator(); // horizontal line to separate tunable parameters from derived values
		ImGui::Text("%d particles, %u cells, rho0 = %.2f", numParticles, gridDim*gridDim*gridDim, rho0);
		ImGui::Text("%.1f FPS, render: %.2f ms", ImGui::GetIO().Framerate, debugTimer);
		ImGui::Text("avg density: %.2f (rho0: %.2f)", avgDensity, rho0);
		ImGui::Separator();
		ImGui::Text("Dragonite");
		ImGui::PushItemWidth(200);
		ImGui::DragFloat3("Position",       &solidPosition.x, 0.1f);
		ImGui::DragFloat3("Rotation (deg)", &solidEulerDeg.x, 1.0f);
		ImGui::DragFloat ("Scale",          &solidScale,       0.01f, 0.01f, 100.0f);
		ImGui::PopItemWidth(); // restore default width for any subsequent widgets
		ImGui::Separator();
		ImGui::Text("Simulation box");
		ImGui::PushItemWidth(200);
		ImGui::DragFloat3("Box min", &boxMin.x, 0.1f, gridMin.x, 0.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::DragFloat3("Box max", &boxMax.x, 0.1f, 0.0f, gridMax.x, "%.2f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::PopItemWidth();
		ImGui::End();

		// Finalizes the frame.ImGui takes all the widgets you defined since NewFrame(), performs layout
		// (positions, sizes, clipping), and produces an ImDrawData structure : a list of vertex buffers, index
		// buffers, and draw commands that describe exactly what triangles to draw and with what textures.No
		// GPU calls happen here - it's pure CPU-side geometry generation.
		ImGui::Render();
		// ImGui needs its own SRV heap bound (for the font texture), so we switch heaps here.
		// The scene's srvHeap was used during RecordGraphicsCommands; that's done, so this is safe.
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
		perFrameCb->particleParams = Float4(rho0, 0.0f, 0.0f, particleRadius); // x = rho0 (for density coloring in PS), w = particle display radius (for billboard sizing in GS)
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
		computeCb->adhesion = adhesion;
		computeCb->pushRadius = particleRadius; // push the particle out one radius' width to not clip visually
		computeCb->solidInvTransform = solidObstacle->GetInvTransform();
		Float3 smin = solidObstacle->GetSdfMin();
		Float3 smax = solidObstacle->GetSdfMax();
		computeCb->sdfMin = Float4(smin, 0.0f);
		computeCb->sdfMax = Float4(smax, 0.0f);
		computeCb->gridMin = gridMin;
		computeCb->gridMax = gridMax;
		computeCb.Upload();
	}

	virtual void Update(float dt, float T) override {
		camera->Animate(dt); // real dt for responsive camera
		lastDt = std::min(dt, 1.0f / 25.0f); // cap at 25Hz: prevents energy spikes on window drag or stutter
		UpdateExternalForce();
		UpdatePerFrameCb();
		SetSolidTransform();
	}

	void CalculateAvgDensity() {
		// map the readback buffer to CPU memory and copy the density data into a vector
		const UINT64 bufferSize = numParticles * sizeof(float);
		void* pData; // ptr will be set by Map to point at the readback buffer's CPU visible memory
		CD3DX12_RANGE readRange(0, bufferSize);
		// in the Map call, we map with the range we intend to read
		if (SUCCEEDED(densityReadbackBuffer->Map(0, &readRange, &pData))) { // prepare pData for reading
			memcpy(densityReadbackData.data(), pData, bufferSize); // actual data movement call
			// during the unmap, we unmap while indicating which bytes we dirtied
			CD3DX12_RANGE writeRange(0, 0); // in this case, we wrote nothing
			densityReadbackBuffer->Unmap(0, &writeRange); // release mapping: invalidate pData
		}

		// Compute average density from readback data
		double densitySum = 0.0;
		for (int i = 0; i < numParticles; i++)
			densitySum += densityReadbackData[i];
		avgDensity = static_cast<float>(densitySum / numParticles);
	}

	// Override Render() to decouple physics (compute queue) from graphics (direct queue).
	//
	// Physics step: CPU waits for compute frame N-1 to finish (so the allocator can be reused),
	// records compute frame N onto computeList, submits it, signals computeFence to N.
	//
	// Graphics step: graphics queue GPU-waits on frame N-1
	// to ensure the snapshot is ready, records and submits the scene draw, presents,
	// signals graphicsFence, and CPU-waits on it.
	virtual void Render() override {
		frameCount++; // increment N for this next render
		Throttle(); // apply fps cap if necessary
		t0 = std::chrono::high_resolution_clock::now(); // debug time measurement start

		if (physicsRunning) {
			// swap which snapshot buffer we write to, which will also swap which snapshot 
			// buffer the graphics reads from
			snapshotWriteIdx ^= 1;

			// We're about to compute data for frame N: wait for the computations
			// of frame N-1 to finish before reusing the allocator and readback buffer.			
			cpuWaitForCompute(frameCount - 1);


			// CalculateAvgDensity reads from the particle readback buffers, which
			// are not double buffered, so techically the GPU can write to them at any point.
			// This means that we should only read from them when we know they can't be
			// mid-write. This is exactly that point: the CPU has waited for the compute to
			// finish, but has not yet dispatched any new GPU commands.
			// BUG: placing this at certain points causes fps degradation
			// but putting it at the very end of Render() caused the issue to go away
			// specifically, it was the cpuWaitForCompute call above that took a long time
			//
			// an interesting debug finding: putting Sleep(2); here *also* causes the same issue!
			// I think this is a case of the GPU going idle cause there's no commands going to it...
			// nvidia-smi --query-gpu=clocks.gr,clocks.mem,power.draw --format=csv -l 1
			// yep :) fix: nvidia control panel -> manage 3d settings -> power management mode -> prefer maximum performance
			CalculateAvgDensity();


			// Safe to write computeCb now: the previous step has finished reading it.
			UpdateComputeCb(lastDt); // update CB to reflect frame N

			PrepareComputeCommandList();
			// Record commands for compute frame N: physics step + snapshot copy
			RecordComputeCommands(snapshotWriteIdx);
			ExecuteCompute(); // dispatch the calculations for frame N			
		}
		// signal: when the compute queue reaches this point, the data for frame N is ready
		// this signal dispatches even if there was no physics loop, since that means that the
		// data for frame N is ready to begin with
		computeFence.signal(computeCommandQueue, frameCount);

		// When this call happens, the compute queue should be working on producing data for frame N.
		// That means that right now the compute  queue is writing the data of frame N to the snapshot 
		// buffer with index snapshotWriteIdx, we can read the data of frame N-1 from 1-snapshotWriteIdx
		int readIdx = 1 - snapshotWriteIdx;

		// GPU-stall the graphics queue until the compute queue has finished writing the snapshot
		// we're about to read: frame N-1. Since the compute pass for frame N is not dispatched until frame
		// N-1 is done computing, the wait under here is usually no-op, just a sanity check.
		graphicsWaitForCompute(frameCount - 1);

		// Record and submit graphics commands for displaying the snapshot at readIdx
		PrepareCommandList();
		RecordGraphicsCommands(readIdx);
		BuildImGui();
		ExecuteGraphics();

		// Signal graphicsFence and CPU-wait: blocks until the graphics queue (including the
		// GPU-side wait above and all subsequent draws) finishes, meaning that the graphics
		// command queue has processed the commands that render frame N-1. After this the graphics
		// allocator is safe to reset next frame, and render frame N.
		graphicsFence.signal(commandQueue, frameCount - 1);
		cpuWaitForGraphics(frameCount - 1);

		// save debug timer value for display in ImGui
		t1 = std::chrono::high_resolution_clock::now();
		debugTimer = std::chrono::duration<float, std::milli>(t1 - t0).count(); 
	}

	// This function cannot be called more than once every targetPeriod time: rate limit
	void Throttle() {
		if (fpsCapped) {
			auto elapsed = clock::now() - lastFrame;
			if (elapsed < targetPeriod) std::this_thread::sleep_for(targetPeriod - elapsed);
		}
		lastFrame = clock::now();
	}

	void CreateParticleBuffers() {
		// create one buffer per particle field on the default heap (GPU-local, writable by compute shaders via UAV)
		const CD3DX12_HEAP_PROPERTIES defaultHeapProps(D3D12_HEAP_TYPE_DEFAULT);
		for (UINT f = 0; f < PF_COUNT; f++) { // PF_COUNT number of particle fields, using enum trick
			// fieldStrides is where the size of a single particle's worth of data for each field is defined
			const UINT64 bufferSize = (UINT64)numParticles * fieldStrides[f]; // how many * how big each
			const CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(
				bufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS); // make it usable as UAV
			DX_API("Failed to create particle field buffer")
				device->CreateCommittedResource(
					&defaultHeapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc,
					D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
					IID_PPV_ARGS(particleFields[f].ReleaseAndGetAddressOf()));
			std::wstring name = std::wstring(fieldNames[f]) + L" Buffer";
			particleFields[f]->SetName(name.c_str()); // for debugging
		}
	}

	void CreateUploadBuffers() {
		// here we create the upload buffers that will serve as a staging area
		// for uploading UAVs' starting data: we can only CPU-copy data into
		// the GPU via an upload buffer, then we use the GPU to copy from the upload buffer
		// to the actual (default heap) buffer where we will use the data in the shader
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
		D3D12_SHADER_RESOURCE_VIEW_DESC posSrvDesc = {}; // zero out the struct to start with default values, then fill in the rest
		posSrvDesc.Format = DXGI_FORMAT_UNKNOWN; // structured buffers use unknown format
		posSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER; // this is a view of a buffer, not a texture, so we use the BUFFER option
		posSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; // required for buffer views
		posSrvDesc.Buffer.FirstElement = 0; // start of the buffer
		posSrvDesc.Buffer.NumElements = numParticles; // how many elements (particles) are in the buffer
		posSrvDesc.Buffer.StructureByteStride = sizeof(Float3); // how big is each element (particle) in bytes
		posSrvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE; // no special options, just a plain structured buffer view
		CD3DX12_CPU_DESCRIPTOR_HANDLE posSrvHandle( // calculate the CPU handle for the SRV descriptor: start of heap + slot index * descriptor size
			descriptorHeap->GetCPUDescriptorHandleForHeapStart(), 1, descriptorSize);
		device->CreateShaderResourceView(particleFields[PF_POSITION].Get(), &posSrvDesc, posSrvHandle); // create the SRV using the description we just filled out

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
		// cell, also used as a running counter by the sorting shader
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
		// these are the buffers into which we gather the particle data in sorted order, then 
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

	void CreatePermBuffer() {
		// one uint per particle: perm[i] = sorted destination index for particle i
		const CD3DX12_HEAP_PROPERTIES defaultHeapProps(D3D12_HEAP_TYPE_DEFAULT);
		const UINT64 bufferSize = (UINT64)numParticles * sizeof(UINT);
		const CD3DX12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(
			bufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		DX_API("Failed to create permutation buffer")
			device->CreateCommittedResource(
				&defaultHeapProps, D3D12_HEAP_FLAG_NONE, &desc,
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
				IID_PPV_ARGS(permBuffer.ReleaseAndGetAddressOf()));
		permBuffer->SetName(L"Permutation Buffer");
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

	void CreatePermUav() {
		// create perm UAV at slot 20 (u16)
		UINT descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
		uavDesc.Format = DXGI_FORMAT_UNKNOWN;
		uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		uavDesc.Buffer.FirstElement = 0;
		uavDesc.Buffer.NumElements = numParticles;
		uavDesc.Buffer.StructureByteStride = sizeof(UINT);
		uavDesc.Buffer.CounterOffsetInBytes = 0;
		uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
		CD3DX12_CPU_DESCRIPTOR_HANDLE handle(
			descriptorHeap->GetCPUDescriptorHandleForHeapStart(), 20, descriptorSize);
		device->CreateUnorderedAccessView(permBuffer.Get(), nullptr, &uavDesc, handle);
	}

	void CreateDensityReadbackBuffer() {
		// create a readback buffer for density only (4 bytes per particle instead of 68)
		// this is here mainly to serve as an example of how to read data back to the CPU
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

	// Create the double-buffered snapshot buffers for position and density.
	// These live in COMMON state: compute writes via CopyBufferRegion (COPY_DEST),
	// then transitions back to COMMON; direct queue reads as SRV (promoted from COMMON).
	// There's 4 of these: 2 per vertex attribute relevant to the graphics: position 
	// and density (for coloring). The compute step writes to one of the two buffers each frame,
	// alternating between them, while the graphics step reads from the other buffer, so there's 
	// no read/write hazard.
	void CreateSnapshotBuffers() {
		const CD3DX12_HEAP_PROPERTIES defaultHeapProps(D3D12_HEAP_TYPE_DEFAULT);
		const UINT64 posSize = (UINT64)numParticles * sizeof(Float3);
		const UINT64 denSize = (UINT64)numParticles * sizeof(float);
		const CD3DX12_RESOURCE_DESC posDesc = CD3DX12_RESOURCE_DESC::Buffer(posSize);
		const CD3DX12_RESOURCE_DESC denDesc = CD3DX12_RESOURCE_DESC::Buffer(denSize);
		for (int i = 0; i < 2; i++) {
			DX_API("Failed to create snapshot position buffer")
				device->CreateCommittedResource(&defaultHeapProps, D3D12_HEAP_FLAG_NONE, &posDesc,
					D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(snapshotPosition[i].ReleaseAndGetAddressOf()));
			snapshotPosition[i]->SetName(i == 0 ? L"Snapshot Position [0]" : L"Snapshot Position [1]");
			DX_API("Failed to create snapshot density buffer")
				device->CreateCommittedResource(&defaultHeapProps, D3D12_HEAP_FLAG_NONE, &denDesc,
					D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(snapshotDensity[i].ReleaseAndGetAddressOf()));
			snapshotDensity[i]->SetName(i == 0 ? L"Snapshot Density [0]" : L"Snapshot Density [1]");
		}
	}

	// Create SRV descriptors for all four snapshot buffers at heap slots 21-24.
	void CreateSnapshotSrvs() {
		UINT sz = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		D3D12_SHADER_RESOURCE_VIEW_DESC posSrvDesc = {};
		posSrvDesc.Format = DXGI_FORMAT_UNKNOWN;
		posSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		posSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		posSrvDesc.Buffer.FirstElement = 0;
		posSrvDesc.Buffer.NumElements = numParticles;
		posSrvDesc.Buffer.StructureByteStride = sizeof(Float3);
		posSrvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
		D3D12_SHADER_RESOURCE_VIEW_DESC denSrvDesc = {};
		denSrvDesc.Format = DXGI_FORMAT_UNKNOWN;
		denSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		denSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		denSrvDesc.Buffer.FirstElement = 0;
		denSrvDesc.Buffer.NumElements = numParticles;
		denSrvDesc.Buffer.StructureByteStride = sizeof(float);
		denSrvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
		for (int i = 0; i < 2; i++) {
			CD3DX12_CPU_DESCRIPTOR_HANDLE posHandle(
				descriptorHeap->GetCPUDescriptorHandleForHeapStart(), 21 + i, sz);
			device->CreateShaderResourceView(snapshotPosition[i].Get(), &posSrvDesc, posHandle);
			CD3DX12_CPU_DESCRIPTOR_HANDLE denHandle(
				descriptorHeap->GetCPUDescriptorHandleForHeapStart(), 23 + i, sz);
			device->CreateShaderResourceView(snapshotDensity[i].Get(), &denSrvDesc, denHandle);
		}
	}

	void CreateDescriptorHeap() {
		// descriptor heap layout (SoA):
		//   slot 0:     cubemap SRV (t0)           — sampled by the background pixel shader
		//   slot 1:     position SRV (t0)          — read by particle VS; CopyDescriptorsSimple redirects this to the active snapshot each frame
		//   slot 2:     density SRV (t1)           — read by particle VS; same
		//   slots 3-9:  particle field UAVs (u0..u6) — compute shader read/write
		//   slot 10:    cellCount UAV (u7)          — per-cell particle count for the spatial grid
		//   slot 11:    cellPrefixSum UAV (u8)      — exclusive prefix sum for sort offsets and neighbor lookups
		//   slots 12-18: sorted field UAVs (u9..u15) — scatter targets for spatial sorting
		//   slot 19:    SDF Texture3D SRV (t0) — sampled by the solid-obstacle collision compute shaders
		//   slot 20:    permutation UAV (u16)  — old index -> sorted index, written by sortCS, read by permutateCS
		//   slot 21:    snapshotPosition[0] SRV — source for CopyDescriptorsSimple into slot 1
		//   slot 22:    snapshotPosition[1] SRV — source for CopyDescriptorsSimple into slot 1
		//   slot 23:    snapshotDensity[0] SRV  — source for CopyDescriptorsSimple into slot 2
		//   slot 24:    snapshotDensity[1] SRV  — source for CopyDescriptorsSimple into slot 2
		D3D12_DESCRIPTOR_HEAP_DESC dhd;
		dhd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE; // GPU can see these descriptors
		dhd.NodeMask = 0; // single GPU setup, so no node masking needed
		dhd.NumDescriptors = 25;
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

	// Fill slots of the main descriptor heap. Every SRV and UAV is created here,
	// so the heap layout is visible in one place.
	void CreateAllDescriptors() {
		// slot 0: cubemap SRV (t0) -- sampled by the background pixel shader
		envTexture.CreateSRV(device.Get(), descriptorHeap.Get(), 0);
		// slots 1-2: particle position + density SRVs -- redirected to active snapshot each frame via CopyDescriptorsSimple
		CreateParticleSrvs();
		// slots 3-9: particle field UAVs (u0..u6) -- compute shader read/write
		CreateParticleUavs();
		// slots 10-11: grid UAVs (u7..u8) -- per-cell particle count and prefix sum
		CreateGridUavs();
		// slots 12-18: sorted field UAVs (u9..u15) -- scatter targets for spatial sorting
		CreateSortUavs();
		// slot 19: SDF Texture3D SRV (t0) -- sampled by solid-obstacle collision compute shaders
		solidObstacle->CreateSdfSrv(device.Get(), descriptorHeap.Get(), 19);
		// slot 20: permutation UAV (u16) -- old index -> sorted index
		CreatePermUav();
		// slots 21-24: snapshot position[0/1] and density[0/1] SRVs -- sources for CopyDescriptorsSimple
		CreateSnapshotSrvs();
	}

	// Compute GPU descriptor handles for the descriptor table ranges used by compute shaders.
	// These are byte offsets into the descriptor heap, stored as members so RecordComputeCommands can use them.
	void CacheDescriptorHandles() {
		UINT sz = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		particleFieldsHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(
			descriptorHeap->GetGPUDescriptorHandleForHeapStart(), 3, sz);   // slot 3  = particle field UAVs (u0..u6)
		gridHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(
			descriptorHeap->GetGPUDescriptorHandleForHeapStart(), 10, sz);  // slot 10 = grid UAVs (u7..u8)
		sortedFieldsHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(
			descriptorHeap->GetGPUDescriptorHandleForHeapStart(), 12, sz);  // slot 12 = sorted field UAVs (u9..u15)
		sdfHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(
			descriptorHeap->GetGPUDescriptorHandleForHeapStart(), 19, sz);  // slot 19 = SDF Texture3D SRV
		permHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(
			descriptorHeap->GetGPUDescriptorHandleForHeapStart(), 20, sz);  // slot 20 = permutation UAV (u16)
	}

	// Batch all initial data uploads (cubemap, particles, SDF) into a single command list execution.
	// All three operate on independent resources so there are no state conflicts.
	void UploadAll() {
		ParticleInitData initData = GenerateParticles(); // initData holds particle data on CPU
		FillUploadBuffers(initData); // memcpy particle data into the upload buffers on the GPU

		DX_API("Failed to reset command allocator (UploadAll)")
			commandAllocator->Reset();
		DX_API("Failed to reset command list (UploadAll)")
			commandList->Reset(commandAllocator.Get(), nullptr);

		envTexture.UploadResource(commandList.Get()); // record cubemap copy + barrier
		RecordParticleUpload(); // record particle copy + barriers
		solidObstacle->UploadSdf(commandList.Get()); // record SDF texture copy + barrier
		UploadSnapshotData(); // record initial state of snapshot buffers for frame 1

		DX_API("Failed to close command list (UploadAll)")
			commandList->Close();
		ID3D12CommandList* cls[] = { commandList.Get() };
		commandQueue->ExecuteCommandLists(_countof(cls), cls);

		WaitFirstFrame(); // GPU-wait until the above uploads are finished and the snapshot buffers are ready for use in frame 1

		// the upload heap copies are done - free the temporary upload resources
		envTexture.ReleaseUploadResources();
		solidObstacle->ReleaseUploadResources();
	}

	// Copy initial particle positions into both snapshot slots so particles are visible
	// before physics starts. Expects command list to be recording.
	void UploadSnapshotData() {
		D3D12_RESOURCE_BARRIER barriers[3];
		barriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(
			particleFields[PF_POSITION].Get(),
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
		barriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(
			snapshotPosition[0].Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
		barriers[2] = CD3DX12_RESOURCE_BARRIER::Transition(
			snapshotPosition[1].Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
		commandList->ResourceBarrier(3, barriers);

		const UINT64 posBytes = (UINT64)numParticles * sizeof(Float3);
		commandList->CopyBufferRegion(snapshotPosition[0].Get(), 0, particleFields[PF_POSITION].Get(), 0, posBytes);
		commandList->CopyBufferRegion(snapshotPosition[1].Get(), 0, particleFields[PF_POSITION].Get(), 0, posBytes);

		barriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(
			particleFields[PF_POSITION].Get(),
			D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		barriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(
			snapshotPosition[0].Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
		barriers[2] = CD3DX12_RESOURCE_BARRIER::Transition(
			snapshotPosition[1].Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
		commandList->ResourceBarrier(3, barriers);
	}

	// Sets frameCount = 1 and signals computeFence to 1 so the
	// graphics queue's GPU-side wait is immediately satisfied on the first frame.
	void WaitFirstFrame() {
		frameCount = 1;
		computeFence.signal(commandQueue, frameCount); // we're done calculating frame 1
		cpuWaitForCompute(frameCount);
		graphicsFence.signal(commandQueue, frameCount); // we're done rendering frame 1		
		cpuWaitForGraphics(frameCount);

		lastFrame = clock::now();
	}

	// Build all graphics rendering pipelines (background, particles, solid obstacle transform).
	void BuildGraphicsPipelines() {
		BuildBackgroundPipeline();
		BuildParticlePipeline();
		SetSolidTransform();
	}

public:
	// Allocate all GPU memory and descriptors.
	// After this returns, every ID3D12Resource and every descriptor heap slot exists,
	// but texture/buffer contents have not been uploaded yet.
	virtual void CreateResources() override {
		AsyncComputeApp::CreateResources(); // command allocators, command lists, PSO manager, fences for both queues

		perFrameCb.CreateResources(device.Get()); // create the constant buffer on the GPU (upload heap, so CPU can write to it every frame)
		computeCb.CreateResources(device.Get()); // create the compute constant buffer (upload heap: dt and numParticles written each frame)
		camera = Egg::Cam::FirstPerson::Create(); // create the camera, which will handle user input and calculate view/projection matrices
		camera->SetView(Float3(0.0f, 5.0f, -20.0f), Float3(0.0f, 0.0f, 1.0f)); // start further back to see the full box
		camera->SetSpeed(10.0f); // movement speed of the camera
		camera->SetAspect(aspectRatio);

		CreateParticleBuffers();
		CreateUploadBuffers();
		CreateGridBuffers();
		CreateSortBuffers();
		CreatePermBuffer();
		CreateDensityReadbackBuffer();
		CreateSnapshotBuffers();
		CreateTextureResources();

		CreateImGuiDescriptorHeap();
		CreateDescriptorHeap();
		CreateAllDescriptors();
		CacheDescriptorHandles();
	}

	// upload initial data to the GPU and build rendering/compute pipelines.
	virtual void LoadAssets() override {
		UploadAll();
		BuildGraphicsPipelines();
		BuildComputePipelines();
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
