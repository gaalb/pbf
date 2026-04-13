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
#include "DescriptorLayout.h"
#include <immintrin.h>
#include <thread>
#include "SharedConfig.hlsli"

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
//   AsyncComputeApp::CreateSwapChainResources
//   InitParticleDepthTextures  - depth texture resources + DSV heap + DSVs
// CreateResources
//   AsyncComputeApp::CreateResources
//   InitDescriptorHeaps  - main, imgui, and snapshot staging heaps
//   InitConstantBuffers  - upload-heap CBs
//   InitCamera
//   InitParticleFields - buffers, UAVs, position/density SRVs, GPU handles
//   InitSortedFields - sorted-field buffers, UAVs, GPU handle
//   InitPermBuffer - permutation buffer, UAV, GPU handle
//   InitGridBuffers - cellCount, cellPrefixSum, groupSum buffers+UAVs+handles
//   InitLodBuffers - LOD + reduction buffers, UAVs, LOD SRV, GPU handles
//   InitReadbackBuffers - density + LOD readback buffers
//   InitSnapshotBuffers - snapshot buffers + staging SRVs
//   InitTexture - cubemap SRV
//   InitObstacle - solid obstacle SDF SRV + sdfHandle
//   InitParticleDepthSrvs - depth SRVs into main heap, GPU handle cache
// LoadAssets
//   UploadAll
//   BuildGraphicsPipelines
//   BuildComputePipelines
// InitImGui
class PbfApp : public AsyncComputeApp {
protected:
	// Fixed particle and grid constants.
	const int particlesX = 100, particlesY = 50, particlesZ = 100; // number of particles along each axis of the initial grid
	const int offsetX = 0, offsetY = 10, offsetZ = 0; // world space offset of the center of the initial particle grid
	const int numParticles = particlesX * particlesY * particlesZ; // total number of particles in the simulation	
	// particleSpacing and hMultiplier are constants that define the SPH kernel width h,
	// which gives a lower bound to the spatial grid's cell width. We can use (try using...)
	// Morton codes to index the cells, which works best with a cubic simulation space that
	// has a power of two number of cells along each axis. Fixing h (= particleSpacing * hMultiplier) 
	// lets us define the box as exactly gridDim * h on each axis, giving a perfectly aligned 
	// cubic grid with a dense Morton code space.
	const float particleSpacing = PARTICLE_SPACING; // inter-particle distance (also determines rest density and display size)
	const float particleRadius = PUSH_RADIUS;
	const float hMultiplier = H_MULTIPLIER; // h = particleSpacing * hMultiplier
	const float h = H; // SPH smoothing radius
	// if the particles are spaced "d" apart, then one d sided cube contains one particle, meaning that
	// each particle is responsible for d^3 volume of fluid, meaning that with m=1, the density is 1/d^3
	const float rho0 = RHO0;
	const Float3 particleColor = Float3(0.9f, 0.1f, 0.7f); // particle display color (RGB)
	const float externalAcceleration = 20.0f; // m/s^2, applied horizontally via arrow keys
	// Spatial grid: cubic, power-of-two cells per axis, box derived from grid.
	// The grid can use Morton code (Z-order curve) indexing, which requires equal power-of-two
	// dimensions on all axes for a dense index space (no wasted codes). We choose a single gridDim
	// for all three axes (cubic grid), such that the box has gridDim * h cells per axis.
	// With gridDim = 32 and h = 0.875, the box is 28 units per side, centered at the origin.
	const UINT gridDim = GRID_DIM; // cells per axis (must be power of two)
	const UINT numCells = gridDim * gridDim * gridDim; // total cells in the grid
	const float boxExtent = gridDim * h / CELL_PER_H; // box side length: gridDim cells of width h/CELL_PER_H
	const Float3 gridMin = Float3(-boxExtent / 2.0f, -boxExtent / 2.0f, -boxExtent / 2.0f); // most negative point of the grid
	const Float3 gridMax = Float3( boxExtent / 2.0f,  boxExtent / 2.0f,  boxExtent / 2.0f); // most positive point of the grid
	Float3 boxMin = Float3(-17.0f, -13.0f, -17.0f); // adjustable collision boundary
	Float3 boxMax = Float3(17.0f, 20.0f, 17.0f); // adjustable collision boundary

	// parameters that are tunable via ImGui each frame
	int solverIterations = 5; // how many newton steps to take per frame
	int minLOD = 2;  // minimum solver iterations for the farthest particles
	float epsilon = 4.0f; // constraint force mixing relaxation parameter, higher value = softer constraints
	float viscosity = 0.01f; // // XSPH viscosity coefficient, higher value = "thicker" fluid, M&M: 0.01
	// artificial purely repulsive pressure term reduces clumping while leaving room for surface tension, 
	float sCorrK = 0.01f; // artificial pressure magnitude coefficient M&M: 0.1
	float vorticityEpsilon = 0.01f; // vorticity confinement strength M&M: 0.01
	float adhesion = 0.02f; // tangential velocity damping on wall contact (0 = frictionless, 1 = full stop)
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
	com_ptr<ID3D12DescriptorHeap> snapshotStagingHeap; // CPU-only staging heap: snapshot SRVs used as CopyDescriptors source

	// particle field buffers (default heap, UAV-accessible by compute shaders): one per attribute
	// Index with ParticleField enum (PF_POSITION..PF_SCRATCH).
	com_ptr<ID3D12Resource> particleFields[PF_COUNT];	
	// Sorted particle field buffers (default heap, UAV). Same layout as particleFields, only
	// used as scatter target / scratch pad during sorting
	com_ptr<ID3D12Resource> sortedFields[PF_COUNT];
	com_ptr<ID3D12Resource> cellCountBuffer; // default heap: uint per cell, stores how many particles are in each cell
	com_ptr<ID3D12Resource> cellPrefixSumBuffer; // default heap: exclusive prefix sum of cellCount, used by sortCS and neighbor lookups
	com_ptr<ID3D12Resource> permBuffer; // default heap: uint per particle, maps old index -> sorted index (computed by sortCS, applied by permutateCS)
	com_ptr<ID3D12Resource> groupSumBuffer; // default heap: per-group totals scratch for the Blelloch 3-pass prefix sum (numCells / (2*THREAD_GROUP_SIZE) uints)
	com_ptr<ID3D12Resource> lodBuffer;  // default heap: per-particle LOD countdown (uint per particle)
	com_ptr<ID3D12Resource> lodReductionBuffer; // default heap: DTC reduction scratch [minDTC bits, maxDTC bits] (2 uints)

	// One ComputeShader per pass. Each holds its own PSO, root signature, descriptor
	// table bindings, and input/output resource lists for UAV barrier insertion.
	// GPU descriptor handles for the descriptor table ranges, computed once in CacheDescriptorHandles.
	// TODO: comments, updated after new sort
	CD3DX12_GPU_DESCRIPTOR_HANDLE particleFieldsHandle; // particle field buffers
	CD3DX12_GPU_DESCRIPTOR_HANDLE gridHandle; // cellCount, cellPrefixSum
	CD3DX12_GPU_DESCRIPTOR_HANDLE sortedFieldsHandle; // sorted particle field buffers
	CD3DX12_GPU_DESCRIPTOR_HANDLE permHandle; // permutation buffer 
	CD3DX12_GPU_DESCRIPTOR_HANDLE cellPrefixSumHandle; // cellPrefixSum alone, used by prefix sum pass 3
	CD3DX12_GPU_DESCRIPTOR_HANDLE groupSumHandle; // group totals scratch, used by all three prefix sum passes
	CD3DX12_GPU_DESCRIPTOR_HANDLE lodHandle; // per-particle LOD UAV
	CD3DX12_GPU_DESCRIPTOR_HANDLE lodReductionHandle; // DTC min/max reduction UAV

	ComputeShader::P predictShader; // apply forces + velocity collision response + predict p* in one per-particle pass
	ComputeShader::P collisionPredictedPositionShader; // clamp p* to box; runs pre-sort and every solver iteration
	ComputeShader::P clearGridShader; // zero cellCount array
	ComputeShader::P countGridShader; // count particles per cell (first grid-build pass)
	ComputeShader::P prefixSumPass1Shader; // Blelloch pass 1: intra-group exclusive scan + save group totals
	ComputeShader::P prefixSumPass2Shader; // Blelloch pass 2: exclusive scan of the group totals
	ComputeShader::P prefixSumPass3Shader; // Blelloch pass 3: add global group offsets to intra-group sums
	ComputeShader::P prefixSumShader; // exclusive prefix sum of cellCount -> cellPrefixSum
	ComputeShader::P sortShader; // each particle computes its sorted destination index -> perm[]
	ComputeShader::P permutateShader; // applies perm[] to scatter all particle fields to sorted positions
	ComputeShader::P lambdaShader; // compute lambda per particle
	ComputeShader::P deltaShader; // compute delta_p, write to scratch
	ComputeShader::P positionFromScratchShader; // copy scratch -> predictedPosition (Jacobi commit during solver loop)
	ComputeShader::P updateVelocityShader;// update velocity from displacement: v = (p* - x) / dt
	ComputeShader::P vorticityShader; // estimate per-particle vorticity (curl of velocity), store in omega
	ComputeShader::P confinementViscosityShader; // vorticity confinement + XSPH viscosity in one neighbor pass
	ComputeShader::P velocityFromScratchShader; // copy scratch -> velocity (Jacobi commit after viscosity)
	ComputeShader::P updatePositionShader; // update position from predictedPosition (final step per paper)
	ComputeShader::P clearDtcReductionShader; // zero lodReduction accumulator before dtcReductionShader
	ComputeShader::P lodReductionShader;  // compute per-frame DTC min/max via GPU atomics
	ComputeShader::P lodShader;           // assign per-particle LOD countdown from DTC range
	ComputeShader::P setLodMaxShader;     // fill all lod[i] = maxLOD (used when adaptivity is off)

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

	// Readback buffer for LOD (readback heap, CPU-readable after CopyBufferRegion).
	// Copied in the same window as the LOD snapshot, before the solver decrements lodBuffer.
	com_ptr<ID3D12Resource> lodReadbackBuffer;
	std::vector<uint32_t> lodReadbackData;
	float avgLod = 0.0f; // average per-particle LOD from previous frame's readback

	using clock = std::chrono::high_resolution_clock;
	clock::time_point lastFrame; // tracks accumulated time toward next physics step
	const float targetFps = 60.0f;
	std::chrono::duration<double> targetPeriod{ 1.0 / targetFps }; // 60 fps cap
	clock::time_point t0, t1; // debug timer variabeles
	float debugTimer = 0.0f;
	float lastDt = 0.0f; // dt from last Update(), consumed by Render() for compute CB upload after GPU sync

	uint64_t frameCount = 0;

	bool fpsCapped = false;
	bool physicsRunning = false;  // toggled by spacebar: when false, compute passes are skipped each frame

	// LOD mode: which per-particle LOD assignment method to use each frame
	enum class LodMode { NONE = 0, DTC = 1, DTVS = 2 };
	LodMode lodMode = LodMode::DTVS; // default to DTVS

	bool arrowLeft = false, arrowRight = false, arrowUp = false, arrowDown = false; // arrow key held state for box translation

	// Async compute: physics runs on the compute queue (from AsyncComputeApp), decoupled from vsync.
	// Double-buffered snapshot buffers hold position, density, and LOD for the graphics queue.
	// snapshotWriteIdx is the slot currently being written by the compute queue;
	// the graphics queue always reads from the OTHER slot (1 - snapshotWriteIdx).
	com_ptr<ID3D12Resource> snapshotPosition[2]; // position snapshot double-buffer (COMMON state, written by compute, read by direct)
	com_ptr<ID3D12Resource> snapshotDensity[2];  // density snapshot double-buffer (COMMON state, written by compute, read by direct)
	com_ptr<ID3D12Resource> snapshotLod[2];      // LOD snapshot double-buffer (COMMON state, written by compute, read by direct)
	int snapshotWriteIdx = 0; // snapshot slot being written by compute, 0 vs 1: graphics reads (1 - snapshotWriteIdx)

	// DTVS: double-buffered window-resolution depth textures. Graphics (cpu frame N) writes
	// slot readIdx; Compute (cpu frame N) reads slot writeIdx — always different resources,
	// so no cross-queue serialization is needed.
	// Created as R32_TYPELESS to allow both D32_FLOAT (DSV write) and R32_FLOAT (SRV read).
	com_ptr<ID3D12Resource> particleDepthTexture[2];    // default heap; recreated on resize
	com_ptr<ID3D12DescriptorHeap> particleDsvHeap;      // 2-slot DSV heap for both textures
	CD3DX12_GPU_DESCRIPTOR_HANDLE particleDepthHandle[2]; // GPU handles for SRV slots 25/26
	CD3DX12_GPU_DESCRIPTOR_HANDLE particleDepthActiveHandle; // set to [writeIdx] each frame before dispatch

	// DTVS graphics pipeline: reuses particleVS + particleGS with a depth-only PS
	com_ptr<ID3D12RootSignature> depthOnlyRootSig;
	com_ptr<ID3D12PipelineState> depthOnlyPso;

	// DTVS compute shaders
	ComputeShader::P clearDtvsReductionShader; // reset lodReduction[0] to 0u before DTVS reduction
	ComputeShader::P dtvsReductionShader;      // accumulate max DTVS into lodReduction[0]
	ComputeShader::P dtvsLodShader;            // assign per-particle LOD from DTVS / maxDTVS

	int shadingMode = SHADING_DENSITY; // current particle shading mode, driven by ImGui

	// Create the two window-resolution depth textures and their 2-slot DSV heap.
	// Called from CreateSwapChainResources (and again on resize). Both textures start in COMMON state.
	// R32_TYPELESS allows both D32_FLOAT DSV writes (graphics) and R32_FLOAT SRV reads (compute DTVS).
	void InitParticleDepthTextures() {
		UINT width  = (UINT)scissorRect.right;
		UINT height = (UINT)scissorRect.bottom;

		// DSV heap: 2 slots, one per double-buffer slot
		D3D12_DESCRIPTOR_HEAP_DESC desc = {};
		desc.NumDescriptors = 2;
		desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
		desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE; // not shader visible: only used for depth writes on the graphics queue, never read by shaders
		DX_API("Failed to create particle DSV descriptor heap")
			device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(particleDsvHeap.ReleaseAndGetAddressOf()));

		D3D12_CLEAR_VALUE clearValue = {};
		clearValue.Format = DXGI_FORMAT_D32_FLOAT;
		clearValue.DepthStencil.Depth = 1.0f; // far plane depth is 1

		const wchar_t* names[2] = { L"Particle Depth Texture [0] (DTVS)", L"Particle Depth Texture [1] (DTVS)" };
		for (int i = 0; i < 2; i++) {
			// Texture (R32_TYPELESS: writable as D32_FLOAT DSV, readable as R32_FLOAT SRV)
			DX_API("Failed to create particle depth texture")
				device->CreateCommittedResource(
					&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), // GPU local memory, not CPU-writable
					D3D12_HEAP_FLAG_NONE, // no special flags
					&CD3DX12_RESOURCE_DESC::Tex2D( // 2D texture
						DXGI_FORMAT_R32_TYPELESS, // typeless allows both DSV (D32_FLOAT) and SRV (R32_FLOAT) views
						width, height, 1, 1, 1, 0, // array size 1, mip levels 1, no multisampling
						D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL), // allow depth stencil usage (required for DSV)
					D3D12_RESOURCE_STATE_COMMON, // start in COMMON state since both queues will transition it before use
					&clearValue, // optimized clear value for depth
					IID_PPV_ARGS(particleDepthTexture[i].ReleaseAndGetAddressOf())); // create the texture and get the resource pointer
			particleDepthTexture[i]->SetName(names[i]); // debug name

			UINT dsvSz = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
			D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc = {};
			dsvDesc.Format = DXGI_FORMAT_D32_FLOAT;
			dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
			dsvDesc.Flags = D3D12_DSV_FLAG_NONE;

			// DSV
			CD3DX12_CPU_DESCRIPTOR_HANDLE dsvHandle(
				particleDsvHeap->GetCPUDescriptorHandleForHeapStart(), i, dsvSz);
			device->CreateDepthStencilView(particleDepthTexture[i].Get(), &dsvDesc, dsvHandle);
		}
	}

	// Create (or recreate on resize) the R32_FLOAT SRVs for both depth textures in the main
	// descriptor heap, and update the GPU handle cache. Requires descriptorHeap to exist.
	void InitParticleDepthSrvs() {
		UINT sz = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

		D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {}; // zero initialize, then set fields we care about
		srvDesc.Format = DXGI_FORMAT_R32_FLOAT; // only red channel is used, as 32 bit float
		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; // 2D texture (as opposed to cubemap, 3D, etc)
		srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; // default shader swizzling
		srvDesc.Texture2D.MostDetailedMip = 0; 
		srvDesc.Texture2D.MipLevels = 1;

		constexpr UINT slots[2] = { HeapSlot::PARTICLE_DEPTH_SRV_0, HeapSlot::PARTICLE_DEPTH_SRV_1 };
		for (int i = 0; i < 2; i++) {
			CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle( // CPU side write handle 
				descriptorHeap->GetCPUDescriptorHandleForHeapStart(), slots[i], sz); // offset into the heap by slot*size
			device->CreateShaderResourceView(
				particleDepthTexture[i].Get(), // what resource to create the view for
				&srvDesc, // describes the srv
				srvHandle); // points to the slot on the heap where the srv will live
			// GPU handle cache (used by dtvsReductionShader / dtvsLodShader each frame)
			particleDepthHandle[i] = CD3DX12_GPU_DESCRIPTOR_HANDLE(
				descriptorHeap->GetGPUDescriptorHandleForHeapStart(), slots[i], sz);
		}
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
		// bind the SRV heap containing the cubemap (root parameter 1, starting at the cubemap slot)
		UINT bgDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		bgMaterial->SetSrvHeap(1, descriptorHeap, HeapSlot::CUBEMAP_SRV * bgDescriptorSize);

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
				D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
			CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_VELOCITY].Get(),
				D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
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

		using std::vector;
		using TableBinding = ComputeShader::TableBinding;
		using P = com_ptr<ID3D12Resource>*;
		
		// regarding std::addressof (reminder for myself because I had a lot of headache due to this:
		// since we're swapping which ID3D12Resource certain com_ptrs point to:
		// for (UINT f = 0; f < PF_COUNT; f++)	std::swap(particleFields[f], sortedFields[f]);
		// we're passing com_ptr<ID3D12Resource>* pointers (pointer to a com_ptr to a ID3DResource)
		// because that allows the outer indirection to correctly re-assess the given
		// com_ptr under a certain label each time we run a shader
		// however, com_ptr<ID3D12Resource>* p = &comptr wouldn't work, only addressof(comptr)!
		// this is because the operator&() overload actually releases the internal pointer
		// in comptr, and so it would leave the com_ptrs here in PbfApp.h as null!
		predictShader = ComputeShader::Create(device.Get(), "Shaders/predictCS.cso", cbv,
			vector<TableBinding>{ {1, &particleFieldsHandle}, {2, &sdfHandle} },
			vector<P>{ std::addressof(particleFields[PF_POSITION]), std::addressof(particleFields[PF_VELOCITY]) },
			vector<P>{ std::addressof(particleFields[PF_VELOCITY]), std::addressof(particleFields[PF_PREDICTED_POSITION]) });

		collisionPredictedPositionShader = ComputeShader::Create(device.Get(), "Shaders/collisionPredictedPositionCS.cso", cbv,
			vector<TableBinding>{ {1, & particleFieldsHandle}, { 2, &lodHandle }, { 3, &sdfHandle } },
			vector<P>{ std::addressof(particleFields[PF_PREDICTED_POSITION]), std::addressof(lodBuffer) },
			vector<P>{ std::addressof(particleFields[PF_PREDICTED_POSITION]) });

		positionFromScratchShader = ComputeShader::Create(device.Get(), "Shaders/positionFromScratchCS.cso", cbv,
			vector<TableBinding>{ {1, & particleFieldsHandle}, { 2, &lodHandle } },
			vector<P>{ std::addressof(particleFields[PF_SCRATCH]), std::addressof(lodBuffer) },
			vector<P>{ std::addressof(particleFields[PF_PREDICTED_POSITION]), std::addressof(lodBuffer) });

		updateVelocityShader = ComputeShader::Create(device.Get(), "Shaders/updateVelocityCS.cso", cbv,
			vector<TableBinding>{ {1, & particleFieldsHandle} },
			vector<P>{ std::addressof(particleFields[PF_POSITION]), std::addressof(particleFields[PF_PREDICTED_POSITION]) },
			vector<P>{ std::addressof(particleFields[PF_VELOCITY]) });

		velocityFromScratchShader = ComputeShader::Create(device.Get(), "Shaders/velocityFromScratchCS.cso", cbv,
			vector<TableBinding>{ {1, & particleFieldsHandle} },
			vector<P>{ std::addressof(particleFields[PF_SCRATCH]) },
			vector<P>{ std::addressof(particleFields[PF_VELOCITY]) });

		updatePositionShader = ComputeShader::Create(device.Get(), "Shaders/updatePositionCS.cso", cbv,
			vector<TableBinding>{ {1, & particleFieldsHandle} },
			vector<P>{ std::addressof(particleFields[PF_PREDICTED_POSITION]) },
			vector<P>{ std::addressof(particleFields[PF_POSITION]) });

		clearGridShader = ComputeShader::Create(device.Get(), "Shaders/clearGridCS.cso", cbv,
			vector<TableBinding>{ {1, & gridHandle} },
			vector<P>{ std::addressof(cellCountBuffer) },
			vector<P>{ std::addressof(cellCountBuffer) });

		// Three-pass Blelloch parallel prefix sum
		// pass 1: intra-group exclusive scan + per-group totals -> cellPrefixSum (local) + groupSums
		// pass 2: exclusive scan of groupSums in-place -> global offsets per group
		// pass 3: add global offsets to cellPrefixSum -> final global exclusive prefix sum
		prefixSumPass1Shader = ComputeShader::Create(device.Get(), "Shaders/prefixSumPass1CS.cso", cbv,
			vector<TableBinding>{ {1, & gridHandle}, { 2, &groupSumHandle } },
			vector<P>{ std::addressof(cellCountBuffer) },
			vector<P>{ std::addressof(cellPrefixSumBuffer), std::addressof(groupSumBuffer) });

		prefixSumPass2Shader = ComputeShader::Create(device.Get(), "Shaders/prefixSumPass2CS.cso", cbv,
			vector<TableBinding>{ {1, & groupSumHandle} },
			vector<P>{ std::addressof(groupSumBuffer) },
			vector<P>{ std::addressof(groupSumBuffer) });

		prefixSumPass3Shader = ComputeShader::Create(device.Get(), "Shaders/prefixSumPass3CS.cso", cbv,
			vector<TableBinding>{ {1, & cellPrefixSumHandle}, { 2, &groupSumHandle } },
			vector<P>{ std::addressof(groupSumBuffer), std::addressof(cellPrefixSumBuffer) },
			vector<P>{ std::addressof(cellPrefixSumBuffer) });

		countGridShader = ComputeShader::Create(device.Get(), "Shaders/countGridCS.cso", cbv,
			vector<TableBinding>{ {1, & particleFieldsHandle}, { 2, &gridHandle } },
			vector<P>{ std::addressof(particleFields[PF_PREDICTED_POSITION]), std::addressof(cellCountBuffer) },
			vector<P>{ std::addressof(cellCountBuffer) });

		lambdaShader = ComputeShader::Create(device.Get(), "Shaders/lambdaCS.cso", cbv,
			vector<TableBinding>{ {1, & particleFieldsHandle}, { 2, &gridHandle }, { 3, &lodHandle } },
			vector<P>{ std::addressof(particleFields[PF_PREDICTED_POSITION]), std::addressof(cellCountBuffer), std::addressof(cellPrefixSumBuffer), std::addressof(lodBuffer) },
			vector<P>{ std::addressof(particleFields[PF_LAMBDA]), std::addressof(particleFields[PF_DENSITY]) });

		deltaShader = ComputeShader::Create(device.Get(), "Shaders/deltaCS.cso", cbv,
			vector<TableBinding>{ {1, & particleFieldsHandle}, { 2, &gridHandle }, { 3, &lodHandle } },
			vector<P>{ std::addressof(particleFields[PF_PREDICTED_POSITION]), std::addressof(particleFields[PF_LAMBDA]), std::addressof(cellCountBuffer), std::addressof(cellPrefixSumBuffer), std::addressof(lodBuffer) },
			vector<P>{ std::addressof(particleFields[PF_SCRATCH]) });

		vorticityShader = ComputeShader::Create(device.Get(), "Shaders/vorticityCS.cso", cbv,
			vector<TableBinding>{ {1, & particleFieldsHandle}, { 2, &gridHandle } },
			vector<P>{ std::addressof(particleFields[PF_POSITION]), std::addressof(particleFields[PF_VELOCITY]), std::addressof(cellCountBuffer), std::addressof(cellPrefixSumBuffer) },
			vector<P>{ std::addressof(particleFields[PF_OMEGA]) });

		confinementViscosityShader = ComputeShader::Create(device.Get(), "Shaders/confinementViscosityCS.cso", cbv,
			vector<TableBinding>{ {1, &particleFieldsHandle}, {2, &gridHandle} },
			vector<P>{ std::addressof(particleFields[PF_POSITION]), std::addressof(particleFields[PF_VELOCITY]),
			std::addressof(particleFields[PF_OMEGA]), std::addressof(cellCountBuffer), std::addressof(cellPrefixSumBuffer) },
			vector<P>{ std::addressof(particleFields[PF_SCRATCH]) });

		sortShader = ComputeShader::Create(device.Get(), "Shaders/sortCS.cso", cbv,
			vector<TableBinding>{ {1, & particleFieldsHandle}, { 2, &gridHandle }, { 3, &permHandle } },
			vector<P>{ std::addressof(particleFields[PF_PREDICTED_POSITION]), std::addressof(cellPrefixSumBuffer), std::addressof(cellCountBuffer) },
			vector<P>{ std::addressof(permBuffer), std::addressof(cellCountBuffer) });

		permutateShader = ComputeShader::Create(device.Get(), "Shaders/permutateCS.cso", cbv,
			vector<TableBinding>{ {1, & particleFieldsHandle}, { 2, &sortedFieldsHandle }, { 3, &permHandle } },
			vector<P>{ std::addressof(particleFields[PF_POSITION]), std::addressof(particleFields[PF_VELOCITY]),
			std::addressof(particleFields[PF_PREDICTED_POSITION]), std::addressof(particleFields[PF_LAMBDA]),
			std::addressof(particleFields[PF_DENSITY]), std::addressof(particleFields[PF_OMEGA]),
			std::addressof(particleFields[PF_SCRATCH]), std::addressof(permBuffer) },
			vector<P>{ std::addressof(sortedFields[PF_POSITION]), std::addressof(sortedFields[PF_VELOCITY]),
			std::addressof(sortedFields[PF_PREDICTED_POSITION]), std::addressof(sortedFields[PF_LAMBDA]),
			std::addressof(sortedFields[PF_DENSITY]), std::addressof(sortedFields[PF_OMEGA]),
			std::addressof(sortedFields[PF_SCRATCH]) });

		clearDtcReductionShader = ComputeShader::Create(device.Get(), "Shaders/clearDtcReductionCS.cso", cbv,
			vector<TableBinding>{ {1, & lodReductionHandle} },
			vector<P>{},
			vector<P>{ std::addressof(lodReductionBuffer) });

		lodReductionShader = ComputeShader::Create(device.Get(), "Shaders/dtcReductionCS.cso", cbv,
			vector<TableBinding>{ {1, & particleFieldsHandle}, { 2, &lodReductionHandle } },
			vector<P>{ std::addressof(particleFields[PF_PREDICTED_POSITION]), std::addressof(lodReductionBuffer) },
			vector<P>{ std::addressof(lodReductionBuffer) });

		lodShader = ComputeShader::Create(device.Get(), "Shaders/dtcLodCS.cso", cbv,
			vector<TableBinding>{ {1, & particleFieldsHandle}, { 2, &lodHandle }, { 3, &lodReductionHandle }},
			vector<P>{ std::addressof(particleFields[PF_PREDICTED_POSITION]), std::addressof(lodReductionBuffer) },
			vector<P>{ std::addressof(lodBuffer)});

		setLodMaxShader = ComputeShader::Create(device.Get(), "Shaders/setLodMaxCS.cso", cbv,
			vector<TableBinding>{ {1, &lodHandle} },
			vector<P>{},
			vector<P>{ std::addressof(lodBuffer) });

		// DTVS shaders: clear reduction buffer, reduce max DTVS, assign LOD from burial depth
		clearDtvsReductionShader = ComputeShader::Create(device.Get(), "Shaders/clearDtvsReductionCS.cso", cbv,
			vector<TableBinding>{ {1, &lodReductionHandle} },
			vector<P>{},
			vector<P>{ std::addressof(lodReductionBuffer) });

		dtvsReductionShader = ComputeShader::Create(device.Get(), "Shaders/dtvsReductionCS.cso", cbv,
			vector<TableBinding>{ {1, &particleFieldsHandle}, {2, &lodReductionHandle}, {3, &particleDepthActiveHandle} },
			vector<P>{ std::addressof(particleFields[PF_PREDICTED_POSITION]), std::addressof(lodReductionBuffer) },
			vector<P>{ std::addressof(lodReductionBuffer) });

		dtvsLodShader = ComputeShader::Create(device.Get(), "Shaders/dtvsLodCS.cso", cbv,
			vector<TableBinding>{ {1, &particleFieldsHandle}, {2, &lodHandle}, {3, &lodReductionHandle}, {4, &particleDepthActiveHandle} },
			vector<P>{ std::addressof(particleFields[PF_PREDICTED_POSITION]), std::addressof(lodReductionBuffer) },
			vector<P>{ std::addressof(lodBuffer) });
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
		// bind the particle SRV table (slots 1-3 in srvHeap) to root parameter 1 so the VS
		// can read position (t0), density (t1), and LOD (t2).
		// SetSrvHeap's third argument is a raw byte offset into the heap, not a descriptor slot index,
		// so we must multiply the slot index by the descriptor increment size to get the correct byte offset
		UINT descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		material->SetSrvHeap(1, descriptorHeap, HeapSlot::PARTICLE_POS_SRV * descriptorSize);

		// NullGeometry: no vertex buffer - the VS fetches positions from the structured buffer using SV_VertexID
		// numParticles tells DrawInstanced how many vertices (and therefore SV_VertexID values) to generate
		Egg::Mesh::Geometry::P geometry = Egg::Mesh::NullGeometry::Create(numParticles);
		geometry->SetTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST); // each SV_VertexID maps to one point, expanded to a quad by the GS

		// mesh = material + geometry + PSO (created by PSO manager based on the material's root signature, shaders, and states)
		particleMesh = Egg::Mesh::Shaded::Create(psoManager, material, geometry);
	}

	// Build the depth-only PSO for the DTVS particle depth pass.
	// Reuses particleVS + particleGS for correct billboard coverage;
	// dtvsDepthOnlyPS discards outside the sphere and writes no color.
	void BuildParticleDepthOnlyPipeline() {
		com_ptr<ID3DBlob> vertexShader   = Egg::Shader::LoadCso("Shaders/particleVS.cso");
		com_ptr<ID3DBlob> geometryShader = Egg::Shader::LoadCso("Shaders/particleGS.cso");
		com_ptr<ID3DBlob> pixelShader    = Egg::Shader::LoadCso("Shaders/dtvsDepthOnlyPS.cso");
		depthOnlyRootSig = Egg::Shader::LoadRootSignature(device.Get(), vertexShader.Get());

		D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
		psoDesc.pRootSignature          = depthOnlyRootSig.Get();
		psoDesc.VS                      = { vertexShader->GetBufferPointer(),   vertexShader->GetBufferSize() };
		psoDesc.GS                      = { geometryShader->GetBufferPointer(), geometryShader->GetBufferSize() };
		psoDesc.PS                      = { pixelShader->GetBufferPointer(),    pixelShader->GetBufferSize() };
		psoDesc.BlendState              = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
		psoDesc.SampleMask              = UINT_MAX;
		psoDesc.RasterizerState         = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
		psoDesc.DepthStencilState       = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT); // depth test + write on
		psoDesc.InputLayout             = { nullptr, 0 }; // no vertex buffer: positions read from SRV
		psoDesc.PrimitiveTopologyType   = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
		psoDesc.NumRenderTargets        = 0; // depth-only: no color output
		psoDesc.DSVFormat               = DXGI_FORMAT_D32_FLOAT;
		psoDesc.SampleDesc.Count        = 1;
		psoDesc.SampleDesc.Quality      = 0;

		DX_API("Failed to create particle depth-only PSO")
			device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(depthOnlyPso.ReleaseAndGetAddressOf()));
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
		// ceil(numParticles / THREAD_GROUP_SIZE) groups cover all particles; the shader discards extra threads
		UINT numGroups = (numParticles + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
		// ceil(numCells / THREAD_GROUP_SIZE) groups cover all cells; the shader discards extra threads
		UINT numCellGroups = (numCells + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;

		// zero the cell count
		clearGridShader->dispatch_then_barrier(computeList.Get(), numCellGroups);

		// count particles per cell (each particle does InterlockedAdd on its cell)
		// after this call, the ith element in cellCount indicates how many particles
		// are in that cell
		countGridShader->dispatch_then_barrier(computeList.Get(), numGroups);

		// Parallel exclusive prefix sum of cellCount -> cellPrefixSum via the Blelloch algorithm.
		// Three passes are required because the full array (32768 cells) doesn't fit in one
		// thread group's shared memory; each group processes 512 cells independently, then
		// a second pass scans the 64 group totals, and a third pass propagates them back.
		// After the three passes, cellPrefixSum[i] = sum of cellCount[0..i-1] for all i.
		UINT numPass1Groups = numCells / (2 * THREAD_GROUP_SIZE); // = 64 for gridDim=32
		prefixSumPass1Shader->dispatch_then_barrier(computeList.Get(), numPass1Groups); // local Blelloch + group totals
		prefixSumPass2Shader->dispatch_then_barrier(computeList.Get(), 1);              // scan group totals into global offsets
		prefixSumPass3Shader->dispatch_then_barrier(computeList.Get(), numPass1Groups); // add global offsets to local sums

		// zero cell counts again so sortCS can use them as per-cell atomic counters
		clearGridShader->dispatch_then_barrier(computeList.Get(), numCellGroups);

		// compute perm[i] = sorted destination index for each particle i
		sortShader->dispatch_then_barrier(computeList.Get(), numGroups);

		// scatter all particle fields to their sorted positions using perm[]
		permutateShader->dispatch_then_barrier(computeList.Get(), numGroups);

		// Double-buffer swap: no copy needed. Instead, swap the GPU descriptor handle values so
		// that SetComputeRootDescriptorTable calls recorded after this point route to the sorted
		// buffers (handle values are captured into the command list at record time, while descriptor
		// heap contents at those addresses are read by the GPU at execution time -- so swapping
		// the handle values here correctly splits the command list into pre-sort and post-sort halves).
		// The com_ptr swap keeps the inputs/outputs barrier resource pointers consistent with
		// whichever physical buffer is now playing the "particle fields" role.
		std::swap(particleFieldsHandle, sortedFieldsHandle);
		for (UINT f = 0; f < PF_COUNT; f++)	std::swap(particleFields[f], sortedFields[f]);
	}

	// Record the DTVS depth-only particle draw into the already-open graphics command list.
	// Writes into particleDepthTexture[readIdx] — the slot NOT being read by compute this frame.
	// Must be called after the snapshot SRVs are already in SRV state (after the toSrv barrier).
	// Leaves the texture in COMMON state so compute can read it next frame.
	void DrawParticleDepth(int readIdx) {
		UINT dsvSz = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);

		// Transition this slot from COMMON to DEPTH_WRITE for the clear + draw
		D3D12_RESOURCE_BARRIER toDepthWrite = CD3DX12_RESOURCE_BARRIER::Transition(
			particleDepthTexture[readIdx].Get(),
			D3D12_RESOURCE_STATE_COMMON,
			D3D12_RESOURCE_STATE_DEPTH_WRITE);
		commandList->ResourceBarrier(1, &toDepthWrite);

		CD3DX12_CPU_DESCRIPTOR_HANDLE particleDepthDsv(
			particleDsvHeap->GetCPUDescriptorHandleForHeapStart(), readIdx, dsvSz);

		// Depth-only render target: no RTV, just the DSV
		commandList->OMSetRenderTargets(0, nullptr, FALSE, &particleDepthDsv);
		commandList->ClearDepthStencilView(particleDepthDsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

		// Bind the depth-only PSO and root signature
		commandList->SetGraphicsRootSignature(depthOnlyRootSig.Get());
		commandList->SetPipelineState(depthOnlyPso.Get());
		commandList->SetGraphicsRootConstantBufferView(0, perFrameCb.GetGPUVirtualAddress());

		// Root param 1: SRV table (pos/density/lod) — pos at t0 drives the VS; same heap slot as main draw
		UINT sz = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		CD3DX12_GPU_DESCRIPTOR_HANDLE posSrvGpu(
			descriptorHeap->GetGPUDescriptorHandleForHeapStart(), HeapSlot::PARTICLE_POS_SRV, sz);
		commandList->SetGraphicsRootDescriptorTable(1, posSrvGpu);

		commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
		commandList->DrawInstanced(numParticles, 1, 0, 0);

		// Restore the original render targets (backbuffer RTV + scene DSV) for subsequent draws
		CD3DX12_CPU_DESCRIPTOR_HANDLE mainDsv(dsvHeap->GetCPUDescriptorHandleForHeapStart());
		CD3DX12_CPU_DESCRIPTOR_HANDLE rtv(
			rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart(),
			swapChainBackBufferIndex,
			rtvDescriptorHandleIncrementSize);
		commandList->OMSetRenderTargets(1, &rtv, FALSE, &mainDsv);

		// Transition back to COMMON so compute can read it next frame
		D3D12_RESOURCE_BARRIER toCommon = CD3DX12_RESOURCE_BARRIER::Transition(
			particleDepthTexture[readIdx].Get(),
			D3D12_RESOURCE_STATE_DEPTH_WRITE,
			D3D12_RESOURCE_STATE_COMMON);
		commandList->ResourceBarrier(1, &toCommon);
	}

	// writeIdx: which snapshot slot to write to this step (caller sets and flips).
	void RecordComputeCommands(int writeIdx) {
		// ceil(numParticles / THREAD_GROUP_SIZE) groups cover all particles; the shader discards extra threads
		UINT numGroups = (numParticles + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;

		// apply forces, wall-correct velocity, predict p* = position + v*dt
		predictShader->dispatch_then_barrier(computeList.Get(), numGroups);

		// Clamp p* to the simulation box before building the spatial grid.
		collisionPredictedPositionShader->dispatch_then_barrier(computeList.Get(), numGroups);

		SortParticles(); // sort particle data for improved cache coherence -> fewer cache misses

		// APBF LOD assignment
		if (lodMode == LodMode::DTC) {
			// DTC: reduce per-frame min/max camera distance, interpolate LOD
			clearDtcReductionShader->dispatch_then_barrier(computeList.Get(), 1);
			lodReductionShader->dispatch_then_barrier(computeList.Get(), numGroups);
			lodShader->dispatch_then_barrier(computeList.Get(), numGroups);
		} else if (lodMode == LodMode::DTVS) {
			// DTVS: read particleDepthTexture[writeIdx] — the slot graphics is NOT writing this frame.
			// ALLOW_DEPTH_STENCIL textures require explicit barriers; no implicit promotion from COMMON.
			particleDepthActiveHandle = particleDepthHandle[writeIdx]; // select the correct slot for dispatch
			D3D12_RESOURCE_BARRIER toSrv = CD3DX12_RESOURCE_BARRIER::Transition(
				particleDepthTexture[writeIdx].Get(),
				D3D12_RESOURCE_STATE_COMMON,
				D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
			computeList->ResourceBarrier(1, &toSrv);

			clearDtvsReductionShader->dispatch_then_barrier(computeList.Get(), 1);
			dtvsReductionShader->dispatch_then_barrier(computeList.Get(), numGroups);
			dtvsLodShader->dispatch_then_barrier(computeList.Get(), numGroups);

			D3D12_RESOURCE_BARRIER backToCommon = CD3DX12_RESOURCE_BARRIER::Transition(
				particleDepthTexture[writeIdx].Get(),
				D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
				D3D12_RESOURCE_STATE_COMMON);
			computeList->ResourceBarrier(1, &backToCommon);
		} else {
			// NONE: every particle runs the full solver (maxLOD iterations)
			setLodMaxShader->dispatch_then_barrier(computeList.Get(), numGroups);
		}

		// Snapshot LOD immediately after assignment — the solver loop decrements lodBuffer
		// each iteration (positionFromScratchCS), so by the end of the loop all values
		// would be 0. We capture the initial per-particle LOD here, before any decrement.
		{
			D3D12_RESOURCE_BARRIER toLodSrc = CD3DX12_RESOURCE_BARRIER::Transition(
				lodBuffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
			D3D12_RESOURCE_BARRIER toLodDest = CD3DX12_RESOURCE_BARRIER::Transition(
				snapshotLod[writeIdx].Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
			computeList->ResourceBarrier(1, &toLodSrc);
			computeList->ResourceBarrier(1, &toLodDest);
			computeList->CopyBufferRegion(snapshotLod[writeIdx].Get(), 0,
				lodBuffer.Get(), 0, (UINT64)numParticles * sizeof(UINT));
			// Copy LOD to readback buffer in the same window (CPU reads it after the next cpuWaitForCompute).
			computeList->CopyBufferRegion(lodReadbackBuffer.Get(), 0,
				lodBuffer.Get(), 0, (UINT64)numParticles * sizeof(UINT));
			D3D12_RESOURCE_BARRIER fromLodSrc = CD3DX12_RESOURCE_BARRIER::Transition(
				lodBuffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
			D3D12_RESOURCE_BARRIER fromLodDest = CD3DX12_RESOURCE_BARRIER::Transition(
				snapshotLod[writeIdx].Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
			computeList->ResourceBarrier(1, &fromLodSrc);
			computeList->ResourceBarrier(1, &fromLodDest);
		}

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
		confinementViscosityShader->dispatch_then_barrier(computeList.Get(), numGroups); // vorticity confinement + XSPH viscosity -> scratch
		velocityFromScratchShader->dispatch_then_barrier(computeList.Get(), numGroups); // scratch -> velocity
		updatePositionShader->dispatch_then_barrier(computeList.Get(), numGroups);    // position = predictedPosition

		// Write snapshot: copy position and density into snapshot slot [writeIdx].
		// (LOD was already snapshotted above, right after lodShader, before the solver decrements it.)
		// Transition particle buffers to COPY_SOURCE, snapshot buffers to COPY_DEST.
		D3D12_RESOURCE_BARRIER toCopySrc[2] = {
			CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_POSITION].Get(),
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
			CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_DENSITY].Get(),
				D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
		};
		D3D12_RESOURCE_BARRIER snapshotToDest[2] = {
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

		// Before the particle draw, redirect descriptor heap slots 1-3 to the active snapshot.
		// The particle VS fetches position (t0), density (t1), and LOD (t2) from slots 1-3 in the heap.
		// We copy the snapshot SRVs from snapshotStagingHeap into slots 1-3 so the
		// shader reads the latest complete snapshot without any root signature change.
		UINT sz = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		device->CopyDescriptorsSimple(1,
			CD3DX12_CPU_DESCRIPTOR_HANDLE(descriptorHeap->GetCPUDescriptorHandleForHeapStart(), HeapSlot::PARTICLE_POS_SRV, sz),
			CD3DX12_CPU_DESCRIPTOR_HANDLE(snapshotStagingHeap->GetCPUDescriptorHandleForHeapStart(), StagingSlot::SNAPSHOT_POS_0 + readIdx, sz),
			D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		device->CopyDescriptorsSimple(1,
			CD3DX12_CPU_DESCRIPTOR_HANDLE(descriptorHeap->GetCPUDescriptorHandleForHeapStart(), HeapSlot::PARTICLE_DEN_SRV, sz),
			CD3DX12_CPU_DESCRIPTOR_HANDLE(snapshotStagingHeap->GetCPUDescriptorHandleForHeapStart(), StagingSlot::SNAPSHOT_DEN_0 + readIdx, sz),
			D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		device->CopyDescriptorsSimple(1,
			CD3DX12_CPU_DESCRIPTOR_HANDLE(descriptorHeap->GetCPUDescriptorHandleForHeapStart(), HeapSlot::PARTICLE_LOD_SRV, sz),
			CD3DX12_CPU_DESCRIPTOR_HANDLE(snapshotStagingHeap->GetCPUDescriptorHandleForHeapStart(), StagingSlot::SNAPSHOT_LOD_0 + readIdx, sz),
			D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

		// Snapshot buffers live in COMMON. Transition them to SRV state for the draw, then back.
		D3D12_RESOURCE_BARRIER toSrv[3] = {
			CD3DX12_RESOURCE_BARRIER::Transition(snapshotPosition[readIdx].Get(),
				D3D12_RESOURCE_STATE_COMMON,
				D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
			CD3DX12_RESOURCE_BARRIER::Transition(snapshotDensity[readIdx].Get(),
				D3D12_RESOURCE_STATE_COMMON,
				D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
			CD3DX12_RESOURCE_BARRIER::Transition(snapshotLod[readIdx].Get(),
				D3D12_RESOURCE_STATE_COMMON,
				D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
		};
		commandList->ResourceBarrier(3, toSrv);

		// DTVS depth-only pass: render billboard particles into particleDepthTexture[readIdx]
		// (the slot compute is NOT reading this frame) so compute can sample it next frame.
		if (lodMode == LodMode::DTVS) DrawParticleDepth(readIdx);

		particleMesh->Draw(commandList.Get()); // draw particles on top

		D3D12_RESOURCE_BARRIER toCommon[3] = {
			CD3DX12_RESOURCE_BARRIER::Transition(snapshotPosition[readIdx].Get(),
				D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
				D3D12_RESOURCE_STATE_COMMON),
			CD3DX12_RESOURCE_BARRIER::Transition(snapshotDensity[readIdx].Get(),
				D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
				D3D12_RESOURCE_STATE_COMMON),
			CD3DX12_RESOURCE_BARRIER::Transition(snapshotLod[readIdx].Get(),
				D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
				D3D12_RESOURCE_STATE_COMMON),
		};
		commandList->ResourceBarrier(3, toCommon);
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

		// Shading mode combo. The order of items must match the ShadingMode:: constants.
		static const char* shadingModeItems[] = { "Unicolor", "Density", "LOD" };
		ImGui::Combo("Shading", &shadingMode, shadingModeItems, IM_ARRAYSIZE(shadingModeItems));
		static const char* lodModeItems[] = { "Non-adaptive", "DTC", "DTVS" };
		int lodModeInt = (int)lodMode;
		if (ImGui::Combo("LOD mode", &lodModeInt, lodModeItems, IM_ARRAYSIZE(lodModeItems)))
			lodMode = (LodMode)lodModeInt;
		ImGui::InputInt("Solver iterations", &solverIterations, 1); // step 1 per click
		ImGui::InputInt("Min LOD", &minLOD, 1);
		ImGui::InputFloat("Epsilon (relaxation)", &epsilon, 0.5f, 1.0f, "%.2f");
		ImGui::InputFloat("Viscosity (XSPH)", &viscosity, 0.001f, 0.01f, "%.4f");
		ImGui::InputFloat("Artificial pressure", &sCorrK, 0.005f, 0.05f, "%.4f");
		ImGui::InputFloat("Vorticity epsilon", &vorticityEpsilon, 0.001f, 0.01f, "%.4f");
		ImGui::InputFloat("Adhesion", &adhesion, 0.01f, 0.1f, "%.3f");
		ImGui::Checkbox("Fountain", &fountainEnabled);		
		ImGui::SameLine();
		ImGui::Checkbox("FPS cap", &fpsCapped);
		ImGui::PopItemWidth(); // restore default width for any subsequent widgets
		// show derived values as read-only text for reference
		ImGui::Separator(); // horizontal line to separate tunable parameters from derived values
		ImGui::Text("%d particles, %u cells", numParticles, gridDim*gridDim*gridDim);
		ImGui::Text("%.1f FPS, render: %.2f ms", ImGui::GetIO().Framerate, debugTimer);
		ImGui::Text("avg density: %.2f (rho0: %.2f)", avgDensity, rho0);
		ImGui::Text("avg LOD: %.2f", avgLod);
		ImGui::Separator();
		ImGui::Text("Dragonite");
		ImGui::PushItemWidth(200);
		ImGui::DragFloat3("Pos",       &solidPosition.x, 0.1f);
		ImGui::DragFloat3("Rot (deg)", &solidEulerDeg.x, 1.0f);
		ImGui::DragFloat ("Scale",          &solidScale,       0.01f, 0.01f, 100.0f);
		ImGui::PopItemWidth(); // restore default width for any subsequent widgets
		ImGui::Separator();
		ImGui::Text("Bounding Box");
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
		perFrameCb->shadingMode = (UINT)shadingMode;
		perFrameCb->minLOD = (UINT)minLOD;
		perFrameCb->maxLOD = (UINT)solverIterations;
		perFrameCb.Upload(); // memcpy the data to the GPU-visible constant buffer
	}

	void UpdateComputeCb(float dt) {
		computeCb->dt = dt;
		computeCb->numParticles = numParticles;
		computeCb->sCorrK = sCorrK;
		computeCb->vorticityEpsilon = vorticityEpsilon;
		computeCb->boxMin = boxMin;
		computeCb->epsilon = epsilon;
		computeCb->boxMax = boxMax;
		computeCb->viscosity = viscosity;
		computeCb->externalForce = externalForce;
		computeCb->fountainEnabled = fountainEnabled ? 1 : 0;
		computeCb->adhesion = adhesion;
		computeCb->solidInvTransform = solidObstacle->GetInvTransform();
		Float3 smin = solidObstacle->GetSdfMin();
		Float3 smax = solidObstacle->GetSdfMax();
		computeCb->sdfMin = Float4(smin, 0.0f);
		computeCb->sdfMax = Float4(smax, 0.0f);
		computeCb->cameraPos = camera->GetEyePosition();
		computeCb->minLOD = (UINT)minLOD;
		computeCb->maxLOD = (UINT)solverIterations;
		computeCb->viewProjTransform = camera->GetViewMatrix() * camera->GetProjMatrix();
		computeCb->viewportWidth  = (float)scissorRect.right;
		computeCb->viewportHeight = (float)scissorRect.bottom;
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

	void CalculateAvgLod() {
		const UINT64 bufferSize = numParticles * sizeof(uint32_t);
		void* pData;
		CD3DX12_RANGE readRange(0, bufferSize);
		if (SUCCEEDED(lodReadbackBuffer->Map(0, &readRange, &pData))) {
			memcpy(lodReadbackData.data(), pData, bufferSize);
			CD3DX12_RANGE writeRange(0, 0);
			lodReadbackBuffer->Unmap(0, &writeRange);
		}

		double lodSum = 0.0;
		for (int i = 0; i < numParticles; i++)
			lodSum += lodReadbackData[i];
		avgLod = static_cast<float>(lodSum / numParticles);
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
			CalculateAvgLod();


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
	// a better way of doing this would be a fixed timestep accumulation, where we decouple
	// physics dt from render dt entirely, accumulate wall-clock time, and step physics at a 
	// fixed interval
	void Throttle() {
		if (fpsCapped) {
			auto deadline = lastFrame + targetPeriod;
			auto remaining = deadline - clock::now();

			// Sleep for all but the last ~1ms to avoid overshooting
			if (remaining > std::chrono::milliseconds(1))
				std::this_thread::sleep_for(remaining - std::chrono::milliseconds(1));

			// Spin-wait the remainder for precision
			while (clock::now() < deadline) {}
		}
		lastFrame = clock::now();
	}

	// Create all three descriptor heaps. Must be called before any Init function
	// that populates descriptors.
	void InitDescriptorHeaps() {
		// ImGui SRV heap: 1 slot, shader-visible, exclusively for ImGui's font texture atlas.
		// Kept separate so we don't disturb the main heap layout.
		{
			D3D12_DESCRIPTOR_HEAP_DESC desc = {};
			desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
			desc.NumDescriptors = 1;
			desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
			DX_API("Failed to create ImGui SRV descriptor heap")
				device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(imguiSrvHeap.GetAddressOf()));
		}

		// Main shader-visible heap: see DescriptorLayout.h for the full slot map.
		{
			D3D12_DESCRIPTOR_HEAP_DESC desc = {};
			desc.Type  = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
			desc.NumDescriptors = HeapSlot::TOTAL;
			desc.Flags  = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
			DX_API("Failed to create descriptor heap")
				device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(descriptorHeap.GetAddressOf()));
		}

		// Each frame the particle VS needs to read from the snapshot buffer that the compute
		// queue isn't currently writing to. The active snapshot changes every frame, so the SRVs
		// at heap slots 1-3 need to be repointed every frame. The approach is to use 
		// device->CopyDescriptorsSimple, which we can use to copy an already created descriptor
		// to the main shader-visible heap. This is cheaper than re-calling CreateShaderResourceView
		// every frame. However, in D3D12, shader-visible heaps (which, of course, the main heap has 
		// to be) are CPU-writeable but not CPU-readable. Since CopyDescriptorsSimple needs to read
		// the source descriptor to copy it, so the source heap can't be shader-visible. Hence,
		// all 6 snapshot SRVs are created once at startup here, in the staging heap. Each frame, 
		// CopyDescriptorsSimple picks the correct ones and overwrites slots 1-3 in the main heap.
		{
			D3D12_DESCRIPTOR_HEAP_DESC desc = {};
			desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
			desc.NumDescriptors = StagingSlot::TOTAL;
			desc.Flags  = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
			DX_API("Failed to create snapshot staging descriptor heap")
				device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(snapshotStagingHeap.GetAddressOf()));
		}
	}

	void InitConstantBuffers() {
		perFrameCb.CreateResources(device.Get());
		computeCb.CreateResources(device.Get());
	}

	void InitCamera() {
		camera = Egg::Cam::FirstPerson::Create();
		camera->SetView(Float3(0.0f, 5.0f, -20.0f), Float3(0.0f, 0.0f, 1.0f));
		camera->SetSpeed(10.0f);
		camera->SetAspect(aspectRatio);
	}

	// particleFields[]: one default-heap UAV buffer per particle attribute.
	// positionUploadBuffer / velocityUploadBuffer: CPU-writable staging for initial data.
	void InitParticleFields() {
		const CD3DX12_HEAP_PROPERTIES defaultHeap(D3D12_HEAP_TYPE_DEFAULT);
		const CD3DX12_HEAP_PROPERTIES uploadHeap(D3D12_HEAP_TYPE_UPLOAD);
		UINT sz = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

		// Default-heap buffers: one per particle attribute, UAV-accessible by compute shaders
		for (UINT f = 0; f < PF_COUNT; f++) {
			UINT64 bufferSize = (UINT64)numParticles * fieldStrides[f]; // fieldStrides shows attribute size
			CD3DX12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(
				bufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
			DX_API("Failed to create particle field buffer")
				device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &desc,
					D3D12_RESOURCE_STATE_COMMON, nullptr,
					IID_PPV_ARGS(particleFields[f].ReleaseAndGetAddressOf()));
			particleFields[f]->SetName((std::wstring(fieldNames[f]) + L" Buffer").c_str());
		}

		// Upload buffers: CPU-writable staging used once at startup to push initial
		// particle positions and velocities to the GPU (released after UploadAll).
		{
			UINT64 size = (UINT64)numParticles * sizeof(Float3);
			DX_API("Failed to create position upload buffer")
				device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE,
					&CD3DX12_RESOURCE_DESC::Buffer(size),
					D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, // upload heaps always start in generic read
					IID_PPV_ARGS(positionUploadBuffer.ReleaseAndGetAddressOf()));
			positionUploadBuffer->SetName(L"Position Upload Buffer");
		}
		{
			UINT64 size = (UINT64)numParticles * sizeof(Float3);
			DX_API("Failed to create velocity upload buffer")
				device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE,
					&CD3DX12_RESOURCE_DESC::Buffer(size),
					D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, // upload heaps always start in generic read
					IID_PPV_ARGS(velocityUploadBuffer.ReleaseAndGetAddressOf()));
			velocityUploadBuffer->SetName(L"Velocity Upload Buffer");
		}

		// UAVs: compute shader read/write (slots PARTICLE_FIELDS .. PARTICLE_FIELDS+PF_COUNT-1)
		for (UINT f = 0; f < PF_COUNT; f++) {
			D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
			uavDesc.Format = DXGI_FORMAT_UNKNOWN; // structured buffer view: the format is determined by the stride, not specified here
			uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; // this UAV describes a buffer resource
			uavDesc.Buffer.NumElements = numParticles;
			uavDesc.Buffer.StructureByteStride = fieldStrides[f];
			CD3DX12_CPU_DESCRIPTOR_HANDLE handle(
				descriptorHeap->GetCPUDescriptorHandleForHeapStart(), HeapSlot::PARTICLE_FIELDS + f, sz);
			device->CreateUnorderedAccessView(particleFields[f].Get(), nullptr, &uavDesc, handle);
		}

		// Position SRV (slot PARTICLE_POS_SRV, t0): particle VS reads positions for billboard placement
		{
			D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
			srvDesc.Format = DXGI_FORMAT_UNKNOWN; // structured buffer view: the format is determined by the stride, not specified here
			srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER; // this SRV describes a buffer resource
			srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; // required for buffer SRVs, allows using the .xyzw swizzle
			srvDesc.Buffer.NumElements = numParticles; 
			srvDesc.Buffer.StructureByteStride = sizeof(Float3);
			CD3DX12_CPU_DESCRIPTOR_HANDLE handle(
				descriptorHeap->GetCPUDescriptorHandleForHeapStart(), HeapSlot::PARTICLE_POS_SRV, sz);
			device->CreateShaderResourceView(particleFields[PF_POSITION].Get(), &srvDesc, handle);
		}

		// Density SRV (slot PARTICLE_DEN_SRV, t1): particle PS reads density for coloring
		{
			D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
			srvDesc.Format  = DXGI_FORMAT_UNKNOWN; // structured buffer view: the format is determined by the stride, not specified here
			srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER; // this SRV describes a buffer resource
			srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; // required for buffer SRVs, allows using the .xyzw swizzle
			srvDesc.Buffer.NumElements  = numParticles;
			srvDesc.Buffer.StructureByteStride = sizeof(float);
			CD3DX12_CPU_DESCRIPTOR_HANDLE handle(
				descriptorHeap->GetCPUDescriptorHandleForHeapStart(), HeapSlot::PARTICLE_DEN_SRV, sz);
			device->CreateShaderResourceView(particleFields[PF_DENSITY].Get(), &srvDesc, handle);
		}

		// GPU handle cache: base of the particle UAV table, swapped with sortedFieldsHandle after each sort
		particleFieldsHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(
			descriptorHeap->GetGPUDescriptorHandleForHeapStart(), HeapSlot::PARTICLE_FIELDS, sz);
	}

	// sortedFields[]: mirror buffers for spatially sorting particle data each frame.
	void InitSortedFields() {
		const CD3DX12_HEAP_PROPERTIES defaultHeap(D3D12_HEAP_TYPE_DEFAULT);
		UINT sz = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

		// Default-heap buffers: identical layout to particleFields[], used as scatter destination
		for (UINT f = 0; f < PF_COUNT; f++) {
			UINT64 bufferSize = (UINT64)numParticles * fieldStrides[f];
			CD3DX12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(
				bufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
			DX_API("Failed to create sorted field buffer")
				device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &desc,
					D3D12_RESOURCE_STATE_COMMON, nullptr,
					IID_PPV_ARGS(sortedFields[f].ReleaseAndGetAddressOf()));
			sortedFields[f]->SetName((std::wstring(L"Sorted ") + fieldNames[f] + L" Buffer").c_str());
		}

		// UAVs (slots SORTED_FIELDS .. SORTED_FIELDS+PF_COUNT-1)
		for (UINT f = 0; f < PF_COUNT; f++) {
			D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
			uavDesc.Format = DXGI_FORMAT_UNKNOWN;
			uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
			uavDesc.Buffer.NumElements = numParticles;
			uavDesc.Buffer.StructureByteStride = fieldStrides[f];
			CD3DX12_CPU_DESCRIPTOR_HANDLE handle(
				descriptorHeap->GetCPUDescriptorHandleForHeapStart(), HeapSlot::SORTED_FIELDS + f, sz);
			device->CreateUnorderedAccessView(sortedFields[f].Get(), nullptr, &uavDesc, handle);
		}

		// GPU handle cache: swapped with particleFieldsHandle in SortParticles() each frame
		sortedFieldsHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(
			descriptorHeap->GetGPUDescriptorHandleForHeapStart(), HeapSlot::SORTED_FIELDS, sz);
	}

	// permBuffer: uint per particle — sortCS writes each particle's sorted destination index here;
	// permutateCS reads it to scatter all fields into their sorted positions.
	void InitPermBuffer() {
		const CD3DX12_HEAP_PROPERTIES defaultHeap(D3D12_HEAP_TYPE_DEFAULT);
		UINT sz = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

		UINT64 bufferSize = (UINT64)numParticles * sizeof(UINT);
		DX_API("Failed to create permutation buffer")
			device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE,
				&CD3DX12_RESOURCE_DESC::Buffer(bufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
				D3D12_RESOURCE_STATE_COMMON, nullptr,
				IID_PPV_ARGS(permBuffer.ReleaseAndGetAddressOf()));
		permBuffer->SetName(L"Permutation Buffer");

		// UAV at slot PERM_UAV
		D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
		uavDesc.Format = DXGI_FORMAT_UNKNOWN;
		uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		uavDesc.Buffer.NumElements = numParticles;
		uavDesc.Buffer.StructureByteStride = sizeof(UINT);
		CD3DX12_CPU_DESCRIPTOR_HANDLE handle(
			descriptorHeap->GetCPUDescriptorHandleForHeapStart(), HeapSlot::PERM_UAV, sz);
		device->CreateUnorderedAccessView(permBuffer.Get(), nullptr, &uavDesc, handle);

		// GPU handle cache
		permHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(
			descriptorHeap->GetGPUDescriptorHandleForHeapStart(), HeapSlot::PERM_UAV, sz);
	}

	// cellCountBuffer: uint per cell, particle count / running atomic counter for sorting.
	// cellPrefixSumBuffer: exclusive prefix sum of cellCount, gives each cell's start offset.
	// groupSumBuffer: per-group totals scratch for the 3-pass Blelloch parallel prefix sum.
	void InitGridBuffers() {
		const CD3DX12_HEAP_PROPERTIES defaultHeap(D3D12_HEAP_TYPE_DEFAULT);
		UINT sz = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

		// cellCount: how many particles landed in each cell; also used as an atomic scatter counter by sortCS
		{
			UINT64 size = numCells * sizeof(UINT);
			DX_API("Failed to create cell count buffer")
				device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE,
					&CD3DX12_RESOURCE_DESC::Buffer(size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
					D3D12_RESOURCE_STATE_COMMON, nullptr,
					IID_PPV_ARGS(cellCountBuffer.ReleaseAndGetAddressOf()));
			cellCountBuffer->SetName(L"Cell Count Buffer");

			D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
			uavDesc.Format = DXGI_FORMAT_UNKNOWN;
			uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
			uavDesc.Buffer.NumElements = numCells;
			uavDesc.Buffer.StructureByteStride = sizeof(UINT);
			CD3DX12_CPU_DESCRIPTOR_HANDLE handle(
				descriptorHeap->GetCPUDescriptorHandleForHeapStart(), HeapSlot::CELL_COUNT, sz);
			device->CreateUnorderedAccessView(cellCountBuffer.Get(), nullptr, &uavDesc, handle);
		}

		// cellPrefixSum: cellPrefixSum[i] = sum(cellCount[0..i-1]), used to find a cell's particle range
		{
			UINT64 size = numCells * sizeof(UINT);
			DX_API("Failed to create cell prefix sum buffer")
				device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE,
					&CD3DX12_RESOURCE_DESC::Buffer(size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
					D3D12_RESOURCE_STATE_COMMON, nullptr,
					IID_PPV_ARGS(cellPrefixSumBuffer.ReleaseAndGetAddressOf()));
			cellPrefixSumBuffer->SetName(L"Cell Prefix Sum Buffer");

			D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
			uavDesc.Format = DXGI_FORMAT_UNKNOWN;
			uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
			uavDesc.Buffer.NumElements = numCells;
			uavDesc.Buffer.StructureByteStride = sizeof(UINT);
			CD3DX12_CPU_DESCRIPTOR_HANDLE handle(
				descriptorHeap->GetCPUDescriptorHandleForHeapStart(), HeapSlot::CELL_PREFIX_SUM, sz);
			device->CreateUnorderedAccessView(cellPrefixSumBuffer.Get(), nullptr, &uavDesc, handle);
		}

		// groupSum: numCells/(2*THREAD_GROUP_SIZE) uints — pass-1 group totals for the Blelloch scan.
		// Pass 1 writes them, pass 2 scans them into global offsets, pass 3 propagates them back.
		{
			UINT numPass1Groups = numCells / (2 * THREAD_GROUP_SIZE);
			UINT64 size = numPass1Groups * sizeof(UINT);
			DX_API("Failed to create group sum buffer")
				device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE,
					&CD3DX12_RESOURCE_DESC::Buffer(size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
					D3D12_RESOURCE_STATE_COMMON, nullptr,
					IID_PPV_ARGS(groupSumBuffer.ReleaseAndGetAddressOf()));
			groupSumBuffer->SetName(L"Prefix Sum Group Sum Buffer");

			D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
			uavDesc.Format = DXGI_FORMAT_UNKNOWN;
			uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
			uavDesc.Buffer.NumElements = numPass1Groups;
			uavDesc.Buffer.StructureByteStride = sizeof(UINT);
			CD3DX12_CPU_DESCRIPTOR_HANDLE handle(
				descriptorHeap->GetCPUDescriptorHandleForHeapStart(), HeapSlot::GROUP_SUM_UAV, sz);
			device->CreateUnorderedAccessView(groupSumBuffer.Get(), nullptr, &uavDesc, handle);
		}

		// GPU handle cache
		gridHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(
			descriptorHeap->GetGPUDescriptorHandleForHeapStart(), HeapSlot::CELL_COUNT, sz);
		cellPrefixSumHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(
			descriptorHeap->GetGPUDescriptorHandleForHeapStart(), HeapSlot::CELL_PREFIX_SUM, sz);
		groupSumHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(
			descriptorHeap->GetGPUDescriptorHandleForHeapStart(), HeapSlot::GROUP_SUM_UAV, sz);
	}

	// lodBuffer: uint per particle — LOD countdown, written by lod shader, decremented by solver.
	// lodReductionBuffer: 2 uints [minDTC bits, maxDTC bits] — DTC/DTVS reduction accumulator.
	void InitLodBuffers() {
		const CD3DX12_HEAP_PROPERTIES defaultHeap(D3D12_HEAP_TYPE_DEFAULT);
		UINT sz = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

		// lodBuffer: one uint per particle, LOD countdown; recomputed fresh each frame after sorting
		{
			UINT64 size = (UINT64)numParticles * sizeof(UINT);
			DX_API("Failed to create LOD buffer")
				device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE,
					&CD3DX12_RESOURCE_DESC::Buffer(size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
					D3D12_RESOURCE_STATE_COMMON, nullptr,
					IID_PPV_ARGS(lodBuffer.ReleaseAndGetAddressOf()));
			lodBuffer->SetName(L"LOD Buffer");

			// UAV at LOD_UAV (compute write)
			D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
			uavDesc.Format = DXGI_FORMAT_UNKNOWN;
			uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
			uavDesc.Buffer.NumElements = numParticles;
			uavDesc.Buffer.StructureByteStride = sizeof(UINT);
			CD3DX12_CPU_DESCRIPTOR_HANDLE uavHandle(
				descriptorHeap->GetCPUDescriptorHandleForHeapStart(), HeapSlot::LOD_UAV, sz);
			device->CreateUnorderedAccessView(lodBuffer.Get(), nullptr, &uavDesc, uavHandle);

			// SRV at PARTICLE_LOD_SRV (particle VS reads LOD for shading; overwritten each frame
			// via CopyDescriptorsSimple to redirect to the active snapshot LOD buffer)
			D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
			srvDesc.Format = DXGI_FORMAT_UNKNOWN;
			srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
			srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
			srvDesc.Buffer.NumElements = numParticles;
			srvDesc.Buffer.StructureByteStride  = sizeof(UINT);
			CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandle(
				descriptorHeap->GetCPUDescriptorHandleForHeapStart(), HeapSlot::PARTICLE_LOD_SRV, sz);
			device->CreateShaderResourceView(lodBuffer.Get(), &srvDesc, srvHandle);
		}

		// lodReductionBuffer: 2 uints — written atomically per frame by clearLodReductionCS + dtcReductionCS
		{
			UINT64 size = 2 * sizeof(UINT);
			DX_API("Failed to create LOD reduction buffer")
				device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE,
					&CD3DX12_RESOURCE_DESC::Buffer(size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
					D3D12_RESOURCE_STATE_COMMON, nullptr,
					IID_PPV_ARGS(lodReductionBuffer.ReleaseAndGetAddressOf()));
			lodReductionBuffer->SetName(L"LOD Reduction Buffer");

			D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
			uavDesc.Format = DXGI_FORMAT_UNKNOWN;
			uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
			uavDesc.Buffer.NumElements = 2;
			uavDesc.Buffer.StructureByteStride = sizeof(UINT);
			CD3DX12_CPU_DESCRIPTOR_HANDLE handle(
				descriptorHeap->GetCPUDescriptorHandleForHeapStart(), HeapSlot::LOD_REDUCTION_UAV, sz);
			device->CreateUnorderedAccessView(lodReductionBuffer.Get(), nullptr, &uavDesc, handle);
		}

		// GPU handle cache
		lodHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(
			descriptorHeap->GetGPUDescriptorHandleForHeapStart(), HeapSlot::LOD_UAV, sz);
		lodReductionHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(
			descriptorHeap->GetGPUDescriptorHandleForHeapStart(), HeapSlot::LOD_REDUCTION_UAV, sz);
	}

	// Readback buffers: CPU-readable copies of density and LOD, copied by the compute queue.
	// These are the way we get data back to the CPU. Currently used to display average density
	// and LOD in the UI.
	void InitReadbackBuffers() {
		const CD3DX12_HEAP_PROPERTIES readbackHeap(D3D12_HEAP_TYPE_READBACK);

		// Density readback: float per particle (4 bytes vs 68 per full particle — minimal readback cost)
		{
			UINT64 size = (UINT64)numParticles * sizeof(float);
			DX_API("Failed to create density readback buffer")
				device->CreateCommittedResource(&readbackHeap, D3D12_HEAP_FLAG_NONE,
					&CD3DX12_RESOURCE_DESC::Buffer(size),
					D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
					IID_PPV_ARGS(densityReadbackBuffer.ReleaseAndGetAddressOf()));
			densityReadbackBuffer->SetName(L"Density Readback Buffer");
			densityReadbackData.resize(numParticles);
		}

		// LOD readback: uint per particle — average displayed in ImGui
		{
			UINT64 size = (UINT64)numParticles * sizeof(uint32_t);
			DX_API("Failed to create LOD readback buffer")
				device->CreateCommittedResource(&readbackHeap, D3D12_HEAP_FLAG_NONE,
					&CD3DX12_RESOURCE_DESC::Buffer(size),
					D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
					IID_PPV_ARGS(lodReadbackBuffer.ReleaseAndGetAddressOf()));
			lodReadbackBuffer->SetName(L"LOD Readback Buffer");
			lodReadbackData.resize(numParticles);
		}
	}

	// Double-buffered snapshot buffers for position, density, and LOD.
	// These live in COMMON state: compute writes via CopyBufferRegion (transitions to COPY_DEST
	// then back to COMMON); graphics reads as SRV (promoted from COMMON automatically).
	// SRVs are placed in the CPU-only snapshotStagingHeap and copied to the main heap each frame
	// via CopyDescriptorsSimple so the graphics queue sees the active read slot.
	void InitSnapshotBuffers() {
		const CD3DX12_HEAP_PROPERTIES defaultHeap(D3D12_HEAP_TYPE_DEFAULT);
		UINT sz = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

		const UINT64 posSize = (UINT64)numParticles * sizeof(Float3);
		const UINT64 denSize = (UINT64)numParticles * sizeof(float);
		const UINT64 lodSize = (UINT64)numParticles * sizeof(UINT);

		for (int i = 0; i < 2; i++) {
			DX_API("Failed to create snapshot position buffer")
				device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE,
					&CD3DX12_RESOURCE_DESC::Buffer(posSize),
					D3D12_RESOURCE_STATE_COMMON, nullptr,
					IID_PPV_ARGS(snapshotPosition[i].ReleaseAndGetAddressOf()));
			snapshotPosition[i]->SetName(i == 0 ? L"Snapshot Position [0]" : L"Snapshot Position [1]");

			DX_API("Failed to create snapshot density buffer")
				device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE,
					&CD3DX12_RESOURCE_DESC::Buffer(denSize),
					D3D12_RESOURCE_STATE_COMMON, nullptr,
					IID_PPV_ARGS(snapshotDensity[i].ReleaseAndGetAddressOf()));
			snapshotDensity[i]->SetName(i == 0 ? L"Snapshot Density [0]" : L"Snapshot Density [1]");

			DX_API("Failed to create snapshot LOD buffer")
				device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE,
					&CD3DX12_RESOURCE_DESC::Buffer(lodSize),
					D3D12_RESOURCE_STATE_COMMON, nullptr,
					IID_PPV_ARGS(snapshotLod[i].ReleaseAndGetAddressOf()));
			snapshotLod[i]->SetName(i == 0 ? L"Snapshot LOD [0]" : L"Snapshot LOD [1]");
		}

		// SRVs in the CPU-only staging heap (used as CopyDescriptorsSimple source each frame)
		D3D12_SHADER_RESOURCE_VIEW_DESC posSrvDesc = {};
		posSrvDesc.Format = DXGI_FORMAT_UNKNOWN;
		posSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		posSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		posSrvDesc.Buffer.NumElements = numParticles;
		posSrvDesc.Buffer.StructureByteStride = sizeof(Float3);

		D3D12_SHADER_RESOURCE_VIEW_DESC denSrvDesc = {};
		denSrvDesc.Format = DXGI_FORMAT_UNKNOWN;
		denSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		denSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		denSrvDesc.Buffer.NumElements = numParticles;
		denSrvDesc.Buffer.StructureByteStride = sizeof(float);

		D3D12_SHADER_RESOURCE_VIEW_DESC lodSrvDesc = {};
		lodSrvDesc.Format = DXGI_FORMAT_UNKNOWN;
		lodSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		lodSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		lodSrvDesc.Buffer.NumElements = numParticles;
		lodSrvDesc.Buffer.StructureByteStride = sizeof(UINT);

		for (int i = 0; i < 2; i++) {
			CD3DX12_CPU_DESCRIPTOR_HANDLE posHandle(
				snapshotStagingHeap->GetCPUDescriptorHandleForHeapStart(), StagingSlot::SNAPSHOT_POS_0 + i, sz);
			device->CreateShaderResourceView(snapshotPosition[i].Get(), &posSrvDesc, posHandle);

			CD3DX12_CPU_DESCRIPTOR_HANDLE denHandle(
				snapshotStagingHeap->GetCPUDescriptorHandleForHeapStart(), StagingSlot::SNAPSHOT_DEN_0 + i, sz);
			device->CreateShaderResourceView(snapshotDensity[i].Get(), &denSrvDesc, denHandle);

			CD3DX12_CPU_DESCRIPTOR_HANDLE lodHandle(
				snapshotStagingHeap->GetCPUDescriptorHandleForHeapStart(), StagingSlot::SNAPSHOT_LOD_0 + i, sz);
			device->CreateShaderResourceView(snapshotLod[i].Get(), &lodSrvDesc, lodHandle);
		}
	}

	// Load the cubemap texture create GPU resources and descriptors.
	// No GPU commands are recorded here — uploads happen later in UploadAll().
	void InitTexture() {
		UINT sz = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

		// Cubemap: reads the DDS file, creates default + upload heap resources, copies to upload heap.
		// Actual GPU transfer deferred to UploadAll() -> envTexture.UploadResource(commandList).
		envTexture = Egg::Importer::ImportTextureCube(device.Get(), "../Media/cloudyNoon.dds");

		// SRV at CUBEMAP_SRV (slot 0, t0): sampled by the background pixel shader
		envTexture.CreateSRV(device.Get(), descriptorHeap.Get(), HeapSlot::CUBEMAP_SRV);
	}

	// Load the solid obstacle; create GPU resources and descriptors.
	// No GPU commands are recorded here — uploads happen later in UploadAll().
	void InitObstacle() {
		UINT sz = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		// Solid obstacle: loads mesh geometry, shaders, material, constant buffer,
		// reads the SDF file and allocates the SDF Texture3D + upload buffer. No GPU commands.
		solidObstacle = SolidObstacle::Create();
		solidObstacle->Load(device.Get(), psoManager, "dragonite.obj", "dragonite.sdf", perFrameCb);

		// SRV at SDF_SRV (slot 20, t0 in CS): sampled by solid-obstacle collision compute shaders
		solidObstacle->CreateSdfSrv(device.Get(), descriptorHeap.Get(), HeapSlot::SDF_SRV);

		// GPU handle cache: used by predictShader and collision shaders each frame
		sdfHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(
			descriptorHeap->GetGPUDescriptorHandleForHeapStart(), HeapSlot::SDF_SRV, sz);
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

		// Pre-clear both depth texture slots to 1.0 (far plane) so the first DTVS compute frame
		// sees valid depth data even before any graphics depth pass has run.
		{
			UINT dsvSz = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
			D3D12_RESOURCE_BARRIER toWrite[2] = {
				CD3DX12_RESOURCE_BARRIER::Transition(particleDepthTexture[0].Get(),
					D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_DEPTH_WRITE),
				CD3DX12_RESOURCE_BARRIER::Transition(particleDepthTexture[1].Get(),
					D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_DEPTH_WRITE),
			};
			commandList->ResourceBarrier(2, toWrite);
			for (int i = 0; i < 2; i++) {
				CD3DX12_CPU_DESCRIPTOR_HANDLE dsvHandle(
					particleDsvHeap->GetCPUDescriptorHandleForHeapStart(), i, dsvSz);
				commandList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
			}
			D3D12_RESOURCE_BARRIER toCommon[2] = {
				CD3DX12_RESOURCE_BARRIER::Transition(particleDepthTexture[0].Get(),
					D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COMMON),
				CD3DX12_RESOURCE_BARRIER::Transition(particleDepthTexture[1].Get(),
					D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COMMON),
			};
			commandList->ResourceBarrier(2, toCommon);
		}

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

	// Build all graphics rendering pipelines (background, particles, DTVS depth-only, solid transform).
	void BuildGraphicsPipelines() {
		BuildBackgroundPipeline();
		BuildParticlePipeline();
		BuildParticleDepthOnlyPipeline();
		SetSolidTransform();
	}

public:
	// Recreate the window-resolution depth textures whenever the swap chain is (re)created.
	virtual void CreateSwapChainResources() override {
		AsyncComputeApp::CreateSwapChainResources(); // base class: RTVs, DSV, viewport/scissorRect
		InitParticleDepthTextures();

		// InitParticleDepthSrvs must be re-ran, because the underlying depth texture resources were recreated 
		// with new sizes, so the SRV descriptors must be updated to point to the new resources. 
		// However, the heap itself might not exist yet on the first call, so guard against that with the nullptr check.
		if (descriptorHeap != nullptr)	InitParticleDepthSrvs();
	}

	virtual void ReleaseSwapChainResources() override {
		particleDepthTexture[0].Reset();
		particleDepthTexture[1].Reset();
		particleDsvHeap.Reset();
		AsyncComputeApp::ReleaseSwapChainResources();
	}


	// Allocate all GPU resources that persist across frames: descriptor heaps, 
	// buffers for particles and sorting, textures for the environment and obstacle, etc.
	// After this returns, every ID3D12Resource and descriptor heap slot exists,
    // but no data has been uploaded to the GPU yet.
	virtual void CreateResources() override {
		AsyncComputeApp::CreateResources(); // command allocators, command lists, PSO manager, fences for both queues

		// Heaps must be first: all Init functions below write descriptors into them.
		InitDescriptorHeaps();
		InitConstantBuffers();
		InitCamera();
		InitParticleFields();
		InitSortedFields();
		InitPermBuffer();
		InitGridBuffers();
		InitLodBuffers();
		InitReadbackBuffers();
		InitSnapshotBuffers();
		InitTexture();
		InitObstacle();
		// depth textures already exist (InitParticleDepthTextures was called from CreateSwapChainResources);
		// write their SRVs into the main heap now that descriptorHeap exists.
		InitParticleDepthSrvs();
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
