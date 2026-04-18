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
#include "DescriptorAllocator.h"
#include "GpuBuffer.h"
#include "GpuTexture.h"
#include "DoubleBufferGpuBuffer.h"
#include "DoubleBufferGpuTexture.h"
#include <immintrin.h>
#include <thread>
#include "SharedConfig.hlsli"

// CPU-side staging struct for ease of particle initialization.
struct ParticleInitData {
	std::vector<Float3> positions;
	std::vector<Float3> velocities;
};

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
//   InitPermBuffer - permutation buffer, UAV, GPU handle
//   InitGridBuffers - cellCount, cellPrefixSum, groupSum buffers+UAVs+handles
//   InitLodBuffers - LOD + reduction buffers, UAVs, LOD SRV, GPU handles
//   InitReadbackBuffers - density + LOD readback buffers
//   InitSnapshotBuffers - snapshot buffers + staging SRVs
//   InitBackground - cubemap SRV
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
	const int particlesX = 75, particlesY = 75, particlesZ = 75; // number of particles along each axis of the initial grid
	const int offsetX = 0, offsetY = 10, offsetZ = 0; // world space offset of the center of the initial particle grid
	const int numParticles = particlesX * particlesY * particlesZ; // total number of particles in the simulation
	// particleSpacing and hMultiplier are constants that define the SPH kernel width h,
	// which gives a lower bound to the spatial grid's cell width. We can use (try using...)
	// Morton codes to index the cells, which works best with a cubic simulation space that
	// has a power of two number of cells along each axis. Fixing h (= particleSpacing * hMultiplier)
	// lets us define the box as exactly gridDim * h on each axis, giving a perfectly aligned
	// cubic grid with a dense Morton code space.
	const float particleSpacing = PARTICLE_SPACING; // inter-particle distance (also determines rest density and display size)
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
	int solverIterations = 6; // how many newton steps to take per frame
	int minLOD = 2;  // minimum solver iterations for the farthest particles
	float epsilon = 4.0f; // constraint force mixing relaxation parameter, higher value = softer constraints
	float viscosity = 0.005f; // // XSPH viscosity coefficient, higher value = "thicker" fluid, M&M: 0.01
	// artificial purely repulsive pressure term reduces clumping while leaving room for surface tension,
	float sCorrK = 0.02f; // artificial pressure magnitude coefficient M&M: 0.1
	float vorticityEpsilon = 0.01f; // vorticity confinement strength M&M: 0.01
	float adhesion = 0.015f; // tangential velocity damping on wall contact (0 = frictionless, 1 = full stop)
	bool fountainEnabled = false; // toggle for the upward jet in a corner of the box, like a fountain :)

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
	GpuBuffer::P positionUploadBuffer;
	GpuBuffer::P velocityUploadBuffer;

	DescriptorAllocator::P imguiAllocator;   // 1-slot shader-visible heap for ImGui font texture
	DescriptorAllocator::P mainAllocator;    // shader-visible CBV/SRV/UAV heap
	DescriptorAllocator::P staticAllocator;  // CPU-only non-shader-visible heap for CopyDescriptorsSimple sources

	// Particle field double buffers: front = live data (compute reads), back = sort target (permutateCS writes).
	// flip() is called once per frame (after cpuWaitForCompute) to promote the sorted back to front.
	DoubleBufferGpuBuffer::P particleFieldDB[PF_COUNT];

	GpuBuffer::P cellCountBuffer;      // default heap: uint per cell, particle count
	GpuBuffer::P cellPrefixSumBuffer;  // default heap: exclusive prefix sum of cellCount
	GpuBuffer::P permBuffer;           // default heap: uint per particle, sort permutation
	GpuBuffer::P groupSumBuffer;       // default heap: Blelloch per-group totals scratch
	GpuBuffer::P lodBuffer;            // default heap: per-particle LOD countdown (uint)
	GpuBuffer::P lodReductionBuffer;   // default heap: DTC reduction scratch [minDTC, maxDTC]

	// One ComputeShader per pass. Each holds its own PSO, root signature, fixed descriptor
	// table handle, and input/output resource lists for UAV barrier insertion.
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
	// GSM-boosted counterparts loaded alongside the plain shaders; dispatched when gsmEnabled is true.
	ComputeShader::P gsmLambdaShader;
	ComputeShader::P gsmDeltaShader;
	ComputeShader::P gsmVorticityShader;
	ComputeShader::P gsmConfinementViscosityShader;
	ComputeShader::P velocityFromScratchShader; // copy scratch -> velocity (Jacobi commit after viscosity)
	ComputeShader::P updatePositionShader; // update position from predictedPosition (final step per paper)
	ComputeShader::P clearDtcReductionShader; // zero lodReduction accumulator before dtcReductionShader
	ComputeShader::P lodReductionShader;  // compute per-frame DTC min/max via GPU atomics
	ComputeShader::P lodShader; // assign per-particle LOD countdown from DTC range
	ComputeShader::P setLodMaxShader; // fill all lod[i] = maxLOD (used when adaptivity is off)

	// Solid obstacle: owns the renderable mesh and the SDF volume texture
	SolidObstacle::P solidObstacle;
	D3D12_CPU_DESCRIPTOR_HANDLE sdfCpuHandle{}; // CPU handle for SDF SRV in staticAllocator (source for CopyDescriptorsSimple)
	Float3 solidPosition = Float3(0.0f, -13.0f, 0.0f); // world-space translation, driven by ImGui
	Float3 solidEulerDeg = Float3(0.0f, 30.0f, 0.0f); // XYZ Euler rotation in degrees, driven by ImGui
	float  solidScale = 2.0f; // uniform scale, driven by ImGui

	// Readback buffers (readback heap, CPU-readable after CopyBufferRegion)
	GpuBuffer::P densityReadbackBuffer;
	std::vector<float> densityReadbackData;
	float avgDensity = 0.0f; // average particle density from previous frame's readback
	GpuBuffer::P lodReadbackBuffer;
	std::vector<uint32_t> lodReadbackData;
	float avgLod = 0.0f; // average per-particle LOD from previous frame's readback

	// debug/throttle timers
	using clock = std::chrono::high_resolution_clock;
	clock::time_point lastFrame; // tracks accumulated time toward next physics step
	const float targetFps = 30.0f; // adjust this for different fps cap
	std::chrono::duration<double> targetPeriod{ 1.0 / targetFps }; // 60 fps cap
	clock::time_point t0, t1; // debug timer variabeles
	float debugTimer = 0.0f;
	float lastDt = 0.0f; // dt from last Update(), consumed by Render() for compute CB upload after GPU sync

	uint64_t frameCount = 0; // indexes each frame
	bool fpsCapped = false; // toggled from ImGui
	bool gsmEnabled = false; // use GSM-boosted shaders for lambda, delta, vorticity, confinementViscosity
	bool physicsRunning = false; // toggled by spacebar: when false, compute passes are skipped each frame

	// LOD mode: which per-particle LOD assignment method to use each frame
	enum class LodMode { NONE = 0, DTC = 1, DTVS = 2 };
	LodMode lodMode = LodMode::DTVS; // default to DTVS

	// arrow key held state for external acceleration input
	bool arrowLeft = false, arrowRight = false, arrowUp = false, arrowDown = false;

	// Async compute: physics runs on the compute queue (from AsyncComputeApp), decoupled from vsync.
	// Double-buffered snapshot buffers hold position, density, and LOD for the graphics queue.
	// Compute writes to DB->getBack(); flip() (called after cpuWaitForCompute) promotes it to front for graphics.
	DoubleBufferGpuBuffer::P positionSnapshotDB; // position snapshot (NON_PIXEL_SHADER_RESOURCE home state)
	DoubleBufferGpuBuffer::P densitySnapshotDB;  // density snapshot
	DoubleBufferGpuBuffer::P lodSnapshotDB;       // LOD snapshot

	// DTVS: double-buffered window-resolution depth textures.
	// Graphics writes to DB->getBack(); flip() (after cpuWaitForGraphics) promotes it to front for compute.
	DoubleBufferGpuTexture::P particleDepthDB;    // default heap; recreated on resize
	DescriptorAllocator::P particleDsvAllocator;  // 2-slot DSV allocator; recreated on resize

	// DTVS graphics pipeline: reuses particleVS + particleGS with a depth-only PS
	com_ptr<ID3D12RootSignature> depthOnlyRootSig;
	com_ptr<ID3D12PipelineState> depthOnlyPso;

	// DTVS compute shaders
	ComputeShader::P clearDtvsReductionShader; // reset lodReduction[0] to 0u before DTVS reduction
	ComputeShader::P dtvsReductionShader; // accumulate max DTVS into lodReduction[0]
	ComputeShader::P dtvsLodShader; // assign per-particle LOD from DTVS / maxDTVS

	// Liquid surface rendering: single-buffered Texture3D density volume.
	// R32_TYPELESS resource; R32_UINT UAV for CAS float atomic add by splatDensityVolumeCS,
	// R32_FLOAT SRV for liquidPS. Home state: COMMON.
	GpuTexture::P                 densityVolume;           // VOL_DIM^3, R32_TYPELESS
	CD3DX12_GPU_DESCRIPTOR_HANDLE densityVolumeHandle;     // GPU handle for R32_UINT UAV (used for ClearUnorderedAccessViewUint)
	CD3DX12_CPU_DESCRIPTOR_HANDLE densityVolClearCpuHandle; // CPU handle in static heap for ClearUnorderedAccessViewUint
	UINT                          liquidTableStartSlot = 0; // first of the 4 contiguous liquidPS SRV slots
	UINT                          particleSrvTableStart = 0; // first of 3 contiguous SRVs: pos(t0),den(t1),lod(t2)
	UINT                          cubemapSrvSlot = 0; // slot for the environment cubemap SRV
	ComputeShader::P densityVolumeShader; // fills density+gradient volume from particle/grid snapshots (kept for reference)

	// Splat density pipeline: one per-particle CS writes Poly6 contributions via CAS float atomic add.
	ComputeShader::P splatDensityShader; // per-particle: Poly6 splat into densityVolume (R32_UINT UAV)

	// Grid snapshot double buffers: copies of cellCountBuffer and cellPrefixSumBuffer for the graphics queue.
	// Compute writes via CopyBufferRegion in WriteSnapshot(); graphics reads in densityVolumeCS.
	// Home state: NON_PIXEL_SHADER_RESOURCE.
	DoubleBufferGpuBuffer::P cellCountSnapshotDB;
	DoubleBufferGpuBuffer::P cellPrefixSumSnapshotDB;

	Egg::Mesh::Shaded::P liquidMesh;  // fullscreen quad rendered with liquidVS + liquidPS
	float liquidIsoThreshold = 0.5f * RHO0; // density cutoff for liquid/air boundary, tunable via ImGui

	int shadingMode = SHADING_UNICOLOR; // current particle shading mode, driven by ImGui

	// Create the two window-resolution depth textures and their 2-slot DSV heap.
	// Called from CreateSwapChainResources (and again on resize). Both textures start in COMMON state.
	// R32_TYPELESS allows both D32_FLOAT DSV writes (graphics) and R32_FLOAT SRV reads (compute DTVS).
	void InitParticleDepthTextures();

	// Create (or recreate on resize) the R32_FLOAT SRVs for both depth textures in the main
	// descriptor heap, and update the GPU handle cache. Requires descriptorHeap to exist.
	void InitParticleDepthSrvs();

	// Create all three descriptor heaps. Must be called before any Init function
	// that populates descriptors.
	void InitDescriptorHeaps();

	void InitConstantBuffers();

	void InitCamera();

	// particleFieldDB[]: one DoubleBufferGpuBuffer per particle attribute.
	// positionUploadBuffer / velocityUploadBuffer: CPU-writable staging for initial data.
	void InitParticleFields();

	// permBuffer: uint per particle - sortCS writes each particle's sorted destination index here;
	// permutateCS reads it to scatter all fields into their sorted positions.
	void InitPermBuffer();

	// cellCountBuffer: uint per cell, particle count / running atomic counter for sorting.
	// cellPrefixSumBuffer: exclusive prefix sum of cellCount, gives each cell's start offset.
	// groupSumBuffer: per-group totals scratch for the 3-pass Blelloch parallel prefix sum.
	void InitGridBuffers();

	// lodBuffer: uint per particle - LOD countdown, written by lod shader, decremented by solver.
	// lodReductionBuffer: 2 uints [minDTC bits, maxDTC bits] - DTC/DTVS reduction accumulator.
	void InitLodBuffers();

	// Readback buffers: CPU-readable copies of density and LOD, copied by the compute queue.
	// These are the way we get data back to the CPU. Currently used to display average density
	// and LOD in the UI.
	void InitReadbackBuffers();

	// Double-buffered snapshot buffers for position, density, and LOD (+ grid snapshots).
	void InitSnapshotBuffers();

	// Create the single-buffered density volume texture and its descriptors.
	// The volume is a VOL_DIM^3 R32_TYPELESS Texture3D.
	// R32_UINT UAV (slot DENSITY_VOL_UAV): splatDensityVolumeCS writes float bits via CAS atomic add.
	// R32_FLOAT SRV (slot DENSITY_VOL_SRV): liquidPS reads the same memory as float density.
	// CPU UAV in static heap (slot DENSITY_VOL_UAV_CLEAR): required by ClearUnorderedAccessViewUint.
	// Home state: COMMON. Committed DEFAULT resource -> zero-initialized by OS (= 0.0f for first frame).
	void InitDensityVolume();

	// Load the cubemap texture create GPU resources and descriptors.
	// No GPU commands are recorded here - uploads happen later in UploadAll().
	void InitBackground();

	// Load the solid obstacle; create GPU resources and descriptors.
	// No GPU commands are recorded here - uploads happen later in UploadAll().
	void InitObstacle();

	// Batch all initial data uploads into a single command list execution.
	// All operate on independent resources so there are no state conflicts.
	// Most functions here will be recording GPU commands along the lines of:
	// transition to dest -> copy buffer region ->transition to common/initial state
	void UploadAll();

	// Copy initial particle positions into both snapshot slots so particles are visible
	// before physics starts. Expects command list to be recording.
	void RecordSnapshotUpload();

	// Sets all depth pixels in both depth texture slots to 1.0 (far plane)
	void RecordDepthTextureClear();

	// Sets frameCount = 1 and signals computeFence to 1 so the
	// graphics queue's GPU-side wait is immediately satisfied on the first frame.
	//
	// Only computeFence is pre-seeded here. graphicsFence is intentionally NOT pre-seeded:
	// Render() signals graphicsFence to (frameCount - 1) at the end of every frame.
	// On the first Render() call frameCount becomes 2, so the signal targets value 1.
	// If WaitFirstFrame also signaled graphicsFence to 1, the second signal (to the same
	// value) would be rejected by D3D12 (signal value must strictly increase), causing
	// cpuWaitForGraphics(1) to return immediately without waiting for the GPU to finish
	// frame 2's command list. The frame-3 allocator reset and CopyDescriptorsSimple calls
	// would then race against the still-executing frame-2 graphics work, producing:
	//   COMMAND_ALLOCATOR_SYNC (#552) -- allocator reset before GPU finishes
	//   STATIC_DESCRIPTOR_INVALID_DESCRIPTOR_CHANGE (#1001) -- descriptor modified while bound
	void WaitFirstFrame();

	// Build all graphics rendering pipelines (background, particles, DTVS depth-only, liquid, solid transform).
	void BuildGraphicsPipelines();

	// Build the background skybox rendering pipeline (shaders, material, mesh).
	// Called after all resources and descriptors are ready.
	void BuildBackgroundPipeline();

	// Build the particle rendering pipeline (shaders, material, mesh).
	// Called after all resources and descriptors are ready.
	void BuildParticlePipeline();

	// Build the depth-only PSO for the DTVS particle depth pass.
	// Reuses particleVS + particleGS for correct billboard coverage;
	// dtvsDepthOnlyPS discards outside the sphere and writes no color.
	void BuildParticleDepthOnlyPipeline();

	// Build the liquid surface rendering pipeline (liquidVS + liquidPS, fullscreen quad).
	// The PS ray-marches through the density volume (t0 = densityVol SRV) and writes SV_Depth
	// for correct depth-buffer occlusion against the solid obstacle.
	void BuildLiquidPipeline();

	// Rebuild the solid's world transform from solidPosition and solidEulerDeg (XYZ Euler, degrees).
	void SetSolidTransform();

	ParticleInitData GenerateParticles();

	// Map the upload buffers and copy initial particle data (positions + velocities) from CPU memory
	// to the upload buffers (upload heap, as opposed to default heap).
	// This is a CPU-side operation; the actual GPU transfer is recorded by RecordParticleUpload().
	void FillUploadBuffers(const ParticleInitData& initData);

	// Record copy commands for particle data into the already-open command list.
	// The command list must have been Reset() before calling this.
	void RecordParticleUpload();

	// Create all compute shader PSOs and their descriptor table bindings.
	// Each shader gets its own contiguous region in the main heap; fixedHandle is baked in at creation.
	void BuildComputePipelines();

	void PrepareComputeCommandList();

	void PrepareCommandList();

	void ExecuteGraphics();

	void ExecuteCompute();

	void SortParticles();

	// Override Render() to decouple physics (compute queue) from graphics (direct queue).
	//
	// Physics step: CPU waits for compute frame N-1 to finish (so the allocator can be reused),
	// records compute frame N onto computeList, submits it, signals computeFence to N.
	//
	// Graphics step: graphics queue GPU-waits on frame N-1
	// to ensure the snapshot is ready, records and submits the scene draw, presents,
	// signals graphicsFence, and CPU-waits on it.
	virtual void Render() override;

	void RecordComputeCommands();

	void WriteSnapshot();

	void CalculateLod();

	void RecordGraphicsCommands();

	// Fill the density volume and draw the ray-marched liquid surface, all on the graphics command list.
	// densityVolumeCS is dispatched here (graphics queue) reading from the previous frame's position and
	// grid snapshots. liquidPS then ray-marches through the freshly filled volume in the same frame.
	void DrawLiquidSurface();

	// Record the DTVS depth-only particle draw into the already-open graphics command list.
	// Writes into particleDepthDB->getBack(). Leaves the texture in COMMON state so compute can read it next frame.
	void DrawParticleDepth();

	void BuildImGui();

	void UpdateExternalForce();

	void UpdatePerFrameCb();

	void UpdateComputeCb(float dt);

	virtual void Update(float dt, float T) override;

	void CalculateAvgDensity();

	void CalculateAvgLod();

	// This function cannot be called more than once every targetPeriod time: rate limit
	// a better way of doing this would be a fixed timestep accumulation, where we decouple
	// physics dt from render dt entirely, accumulate wall-clock time, and step physics at a
	// fixed interval
	void Throttle();

public:
	// Recreate the window-resolution depth textures whenever the swap chain is (re)created.
	virtual void CreateSwapChainResources() override;

	virtual void ReleaseSwapChainResources() override;

	// Allocate all GPU resources that persist across frames: descriptor heaps,
	// buffers for particles and sorting, textures for the environment and obstacle, etc.
	// After this returns, every ID3D12Resource and descriptor heap slot exists,
    // but no data has been uploaded to the GPU yet.
	virtual void CreateResources() override;

	// upload initial data to the GPU and build rendering/compute pipelines.
	virtual void LoadAssets() override;

	// Call once after CreateResources + LoadAssets, from main.cpp where the HWND is available.
	// Sets up ImGui context and its Win32 + D3D12 backends. At this point the D3D12 device, command queue, and
	//imguiSrvHeap all exist.
	void InitImGui(HWND hwnd);

	void ShutdownImGui();

	// Forward window messages (keyboard, mouse) to the camera, and handle app-level hotkeys
	virtual void ProcessMessage(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) override;
};
