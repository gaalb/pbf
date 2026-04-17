#include "PbfApp.h"

using namespace Egg::Math;

// Create the two window-resolution depth textures and their 2-slot DSV heap.
// Called from CreateSwapChainResources (and again on resize). Both textures start in COMMON state.
// R32_TYPELESS allows both D32_FLOAT DSV writes (graphics) and R32_FLOAT SRV reads (compute DTVS).
void PbfApp::InitParticleDepthTextures() {
	UINT width  = (UINT)scissorRect.right;
	UINT height = (UINT)scissorRect.bottom;

	// Fresh 2-slot DSV allocator for this resolution (recreated every resize).
	particleDsvAllocator = DescriptorAllocator::Create(
		device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 2, /*shaderVisible*/false);

	D3D12_CLEAR_VALUE clearValue = {};
	clearValue.Format = DXGI_FORMAT_D32_FLOAT;
	clearValue.DepthStencil.Depth = 1.0f;

	const wchar_t* names[2] = { L"Particle Depth Texture [0] (DTVS)", L"Particle Depth Texture [1] (DTVS)" };
	for (int i = 0; i < 2; i++) {
		particleDepthTexture[i] = GpuTexture::Create2DWithClearValue(
			device.Get(), width, height,
			DXGI_FORMAT_R32_TYPELESS,
			D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
			names[i],
			D3D12_RESOURCE_STATE_COMMON,
			&clearValue);
		particleDepthTexture[i]->CreateDsv(device.Get(), *particleDsvAllocator, DXGI_FORMAT_D32_FLOAT);
	}
}

// Create (or recreate on resize) the R32_FLOAT SRVs for both depth textures in the main
// descriptor heap. Slots are allocated once on the first call and reused on resize.
void PbfApp::InitParticleDepthSrvs() {
	for (int i = 0; i < 2; i++) {
		if (particleDepthSrvSlot[i] == UINT_MAX) {
			// First call: allocate a slot and record it.
			particleDepthSrvSlot[i] = mainAllocator->Allocate();
		}
		// On every call (including resize): recreate the SRV at the same slot so it
		// points at the new texture resource.
		particleDepthTexture[i]->CreateSrvAt(
			device.Get(),
			mainAllocator->GetCpuHandle(particleDepthSrvSlot[i]),
			mainAllocator->GetGpuHandle(particleDepthSrvSlot[i]),
			DXGI_FORMAT_R32_FLOAT);
		particleDepthHandle[i] = particleDepthTexture[i]->GetSrvGpuHandle();
	}
}

// Create all descriptor allocators. Must be called before any Init function
// that populates descriptors.
void PbfApp::InitDescriptorHeaps() {
	// ImGui SRV heap: 1 slot, shader-visible, exclusively for ImGui's font texture atlas.
	imguiAllocator = DescriptorAllocator::Create(
		device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, /*shaderVisible*/true);

	// Main shader-visible heap: 64 slots (39 used by current layout, 25 spare).
	mainAllocator = DescriptorAllocator::Create(
		device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 64, /*shaderVisible*/true);

	// CPU-only staging heap: source for per-frame CopyDescriptorsSimple calls.
	// All snapshot SRVs are created here once; each frame CopyDescriptorsSimple
	// copies the active slot into the main shader-visible heap.
	stagingAllocator = DescriptorAllocator::Create(
		device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 32, /*shaderVisible*/false);
}

void PbfApp::InitConstantBuffers() {
	perFrameCb.CreateResources(device.Get());
	computeCb.CreateResources(device.Get());
}

void PbfApp::InitCamera() {
	camera = Egg::Cam::FirstPerson::Create();
	camera->SetView(Float3(0.0f, 5.0f, -20.0f), Float3(0.0f, 0.0f, 1.0f));
	camera->SetSpeed(10.0f);
	camera->SetAspect(aspectRatio);
}

// particleFields[]: one default-heap UAV buffer per particle attribute.
// positionUploadBuffer / velocityUploadBuffer: CPU-writable staging for initial data.
void PbfApp::InitParticleFields() {
	// Allocate PF_COUNT contiguous UAV slots for the particle field descriptor table.
	UINT pfStart = mainAllocator->Allocate(PF_COUNT);

	for (UINT f = 0; f < PF_COUNT; f++) {
		particleFields[f] = GpuBuffer::Create(
			device.Get(), numParticles, fieldStrides[f],
			(std::wstring(fieldNames[f]) + L" Buffer").c_str(),
			D3D12_RESOURCE_STATE_COMMON,
			D3D12_HEAP_TYPE_DEFAULT);
		particleFields[f]->CreateUavAt(
			device.Get(),
			mainAllocator->GetCpuHandle(pfStart + f),
			mainAllocator->GetGpuHandle(pfStart + f));
	}

	// Upload buffers: CPU-writable staging used once at startup.
	positionUploadBuffer = GpuBuffer::Create(
		device.Get(), numParticles, sizeof(Float3),
		L"Position Upload Buffer",
		D3D12_RESOURCE_STATE_GENERIC_READ,
		D3D12_HEAP_TYPE_UPLOAD);
	velocityUploadBuffer = GpuBuffer::Create(
		device.Get(), numParticles, sizeof(Float3),
		L"Velocity Upload Buffer",
		D3D12_RESOURCE_STATE_GENERIC_READ,
		D3D12_HEAP_TYPE_UPLOAD);

	// Reserve 3 contiguous slots for the particle SRV table (t0=pos, t1=den, t2=lod).
	// Lod is placed at +2 by InitLodBuffers using particleSrvTableStart.
	particleSrvTableStart = mainAllocator->Allocate(3);
	particleFields[PF_POSITION]->CreateSrvAt(
		device.Get(),
		mainAllocator->GetCpuHandle(particleSrvTableStart),
		mainAllocator->GetGpuHandle(particleSrvTableStart));
	particleFields[PF_DENSITY]->CreateSrvAt(
		device.Get(),
		mainAllocator->GetCpuHandle(particleSrvTableStart + 1),
		mainAllocator->GetGpuHandle(particleSrvTableStart + 1));

	// GPU handle cache: base of the UAV table, swapped with sortedFieldsHandle after each sort.
	particleFieldsHandle = mainAllocator->GetGpuHandle(pfStart);
}

// sortedFields[]: mirror buffers for spatially sorting particle data each frame.
void PbfApp::InitSortedFields() {
	UINT sfStart = mainAllocator->Allocate(PF_COUNT);

	for (UINT f = 0; f < PF_COUNT; f++) {
		sortedFields[f] = GpuBuffer::Create(
			device.Get(), numParticles, fieldStrides[f],
			(std::wstring(L"Sorted ") + fieldNames[f] + L" Buffer").c_str(),
			D3D12_RESOURCE_STATE_COMMON,
			D3D12_HEAP_TYPE_DEFAULT);
		sortedFields[f]->CreateUavAt(
			device.Get(),
			mainAllocator->GetCpuHandle(sfStart + f),
			mainAllocator->GetGpuHandle(sfStart + f));
	}

	// GPU handle cache: swapped with particleFieldsHandle in SortParticles() each frame.
	sortedFieldsHandle = mainAllocator->GetGpuHandle(sfStart);
}

// permBuffer: uint per particle - sortCS writes sorted destination index; permutateCS reads it.
void PbfApp::InitPermBuffer() {
	permBuffer = GpuBuffer::Create(
		device.Get(), numParticles, sizeof(UINT),
		L"Permutation Buffer",
		D3D12_RESOURCE_STATE_COMMON,
		D3D12_HEAP_TYPE_DEFAULT);
	permBuffer->CreateUav(device.Get(), *mainAllocator);
	permHandle = permBuffer->GetUavGpuHandle();
}

// cellCountBuffer, cellPrefixSumBuffer, groupSumBuffer.
void PbfApp::InitGridBuffers() {
	// cellCount and cellPrefixSum must be contiguous (gridHandle covers both as a 2-slot table).
	UINT gridStart = mainAllocator->Allocate(2);

	cellCountBuffer = GpuBuffer::Create(
		device.Get(), numCells, sizeof(UINT),
		L"Cell Count Buffer", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
	cellCountBuffer->CreateUavAt(
		device.Get(),
		mainAllocator->GetCpuHandle(gridStart),
		mainAllocator->GetGpuHandle(gridStart));

	cellPrefixSumBuffer = GpuBuffer::Create(
		device.Get(), numCells, sizeof(UINT),
		L"Cell Prefix Sum Buffer", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
	cellPrefixSumBuffer->CreateUavAt(
		device.Get(),
		mainAllocator->GetCpuHandle(gridStart + 1),
		mainAllocator->GetGpuHandle(gridStart + 1));

	UINT numPass1Groups = numCells / (2 * THREAD_GROUP_SIZE);
	groupSumBuffer = GpuBuffer::Create(
		device.Get(), numPass1Groups, sizeof(UINT),
		L"Prefix Sum Group Sum Buffer", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
	groupSumBuffer->CreateUav(device.Get(), *mainAllocator);

	// GPU handle cache
	gridHandle          = mainAllocator->GetGpuHandle(gridStart);
	cellPrefixSumHandle = mainAllocator->GetGpuHandle(gridStart + 1);
	groupSumHandle      = groupSumBuffer->GetUavGpuHandle();
}

// lodBuffer: uint per particle - LOD countdown.
// lodReductionBuffer: 2 uints [minDTC bits, maxDTC bits].
void PbfApp::InitLodBuffers() {
	lodBuffer = GpuBuffer::Create(
		device.Get(), numParticles, sizeof(UINT),
		L"LOD Buffer", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
	lodBuffer->CreateUav(device.Get(), *mainAllocator);

	// Place LOD SRV at the 3rd slot of the particle SRV table (t2), adjacent to pos(t0) and den(t1).
	// Overwritten each frame by CopyDescriptorsSimple to redirect to the active snapshot LOD.
	lodBuffer->CreateSrvAt(
		device.Get(),
		mainAllocator->GetCpuHandle(particleSrvTableStart + 2),
		mainAllocator->GetGpuHandle(particleSrvTableStart + 2));

	lodReductionBuffer = GpuBuffer::Create(
		device.Get(), 2, sizeof(UINT),
		L"LOD Reduction Buffer", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
	lodReductionBuffer->CreateUav(device.Get(), *mainAllocator);

	lodHandle          = lodBuffer->GetUavGpuHandle();
	lodReductionHandle = lodReductionBuffer->GetUavGpuHandle();
}

// Readback buffers: CPU-readable copies of density and LOD.
void PbfApp::InitReadbackBuffers() {
	densityReadbackBuffer = GpuBuffer::Create(
		device.Get(), numParticles, sizeof(float),
		L"Density Readback Buffer",
		D3D12_RESOURCE_STATE_COPY_DEST,
		D3D12_HEAP_TYPE_READBACK);
	densityReadbackData.resize(numParticles);

	lodReadbackBuffer = GpuBuffer::Create(
		device.Get(), numParticles, sizeof(uint32_t),
		L"LOD Readback Buffer",
		D3D12_RESOURCE_STATE_COPY_DEST,
		D3D12_HEAP_TYPE_READBACK);
	lodReadbackData.resize(numParticles);
}

// Double-buffered snapshot buffers for position, density, and LOD.
void PbfApp::InitSnapshotBuffers() {
	const wchar_t* posNames[2] = { L"Snapshot Position [0]", L"Snapshot Position [1]" };
	const wchar_t* denNames[2] = { L"Snapshot Density [0]",  L"Snapshot Density [1]"  };
	const wchar_t* lodNames[2] = { L"Snapshot LOD [0]",      L"Snapshot LOD [1]"      };

	for (int i = 0; i < 2; i++) {
		snapshotPosition[i] = GpuBuffer::Create(
			device.Get(), numParticles, sizeof(Float3),
			posNames[i], D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);

		snapshotDensity[i] = GpuBuffer::Create(
			device.Get(), numParticles, sizeof(float),
			denNames[i], D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);

		snapshotLod[i] = GpuBuffer::Create(
			device.Get(), numParticles, sizeof(UINT),
			lodNames[i], D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);

		// SRV for posSnapshot in the main heap FIRST: used as t0 by densityVolumeCS (graphics queue).
		UINT gfxSlot = mainAllocator->Allocate();
		snapshotPosition[i]->CreateSrvAt(
			device.Get(),
			mainAllocator->GetCpuHandle(gfxSlot),
			mainAllocator->GetGpuHandle(gfxSlot));
		posSnapshotGfxHandle[i] = mainAllocator->GetGpuHandle(gfxSlot);

		// SRVs in the CPU-only staging heap LAST: GetSrvCpuHandle() must return the staging handle
		// for CopyDescriptorsSimple calls in RecordGraphicsCommands and DrawLiquidSurface.
		snapshotPosition[i]->CreateSrv(device.Get(), *stagingAllocator);
		snapshotDensity[i]->CreateSrv(device.Get(), *stagingAllocator);
		snapshotLod[i]->CreateSrv(device.Get(), *stagingAllocator);
	}
}

// Double-buffered grid snapshot buffers (cellCount + cellPrefixSum pairs).
void PbfApp::InitGridSnapshotBuffers() {
	const wchar_t* countNames[2]  = { L"Cell Count Snapshot [0]",      L"Cell Count Snapshot [1]"      };
	const wchar_t* prefixNames[2] = { L"Cell Prefix Sum Snapshot [0]", L"Cell Prefix Sum Snapshot [1]" };

	for (int i = 0; i < 2; i++) {
		cellCountSnapshot[i] = GpuBuffer::Create(
			device.Get(), numCells, sizeof(UINT),
			countNames[i], D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);

		cellPrefixSumSnapshot[i] = GpuBuffer::Create(
			device.Get(), numCells, sizeof(UINT),
			prefixNames[i], D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);

		// Main heap: allocate a contiguous 2-slot block so (count, prefix) form a table for densityVolumeCS.
		UINT pairStart = mainAllocator->Allocate(2);
		cellCountSnapshot[i]->CreateSrvAt(
			device.Get(),
			mainAllocator->GetCpuHandle(pairStart),
			mainAllocator->GetGpuHandle(pairStart));
		cellPrefixSumSnapshot[i]->CreateSrvAt(
			device.Get(),
			mainAllocator->GetCpuHandle(pairStart + 1),
			mainAllocator->GetGpuHandle(pairStart + 1));
		gridSnapshotHandle[i] = mainAllocator->GetGpuHandle(pairStart);

		// Staging heap: contiguous count+prefix pair for CopyDescriptorsSimple(2,...) in DrawLiquidSurface.
		UINT stagingPairStart = stagingAllocator->Allocate(2);
		cellCountSnapshot[i]->CreateSrvAt(
			device.Get(),
			stagingAllocator->GetCpuHandle(stagingPairStart),
			{}); // GPU handle unused in CPU-only heap
		cellPrefixSumSnapshot[i]->CreateSrvAt(
			device.Get(),
			stagingAllocator->GetCpuHandle(stagingPairStart + 1),
			{});
	}
}

// Single-buffered density volume: VOL_DIM^3, R32_TYPELESS.
// R32_UINT UAV for CAS float atomic add; R32_FLOAT SRV for liquidPS.
// CPU-only UAV in staging heap for ClearUnorderedAccessViewUint.
void PbfApp::InitDensityVolume() {
	const UINT volDim = (UINT)VOL_DIM;
	densityVolume = GpuTexture::Create3D(
		device.Get(), volDim, volDim, volDim,
		DXGI_FORMAT_R32_TYPELESS,
		D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
		L"Density Volume",
		D3D12_RESOURCE_STATE_COMMON);

	// R32_UINT UAV in main heap: splatDensityVolumeCS writes via CAS float atomic add.
	densityVolume->CreateUav(device.Get(), *mainAllocator, DXGI_FORMAT_R32_UINT);
	densityVolumeHandle = densityVolume->GetUavGpuHandle();

	// CPU-only R32_UINT UAV in staging heap: required by ClearUnorderedAccessViewUint.
	UINT clearSlot = stagingAllocator->Allocate();
	densityVolume->CreateUavAt(
		device.Get(),
		stagingAllocator->GetCpuHandle(clearSlot),
		{}, // GPU handle unused in CPU-only heap
		DXGI_FORMAT_R32_UINT);
	densityVolClearCpuHandle = stagingAllocator->GetCpuHandle(clearSlot);

	// 4-slot contiguous liquid table: [density(t0), pos(t1), gridCount(t2), gridPrefix(t3)].
	// density SRV is placed at liquidTableStartSlot+0 and is static; the other three are
	// overwritten per-frame in DrawLiquidSurface() via CopyDescriptorsSimple.
	liquidTableStartSlot = mainAllocator->Allocate(4);
	densityVolume->CreateSrvAt(
		device.Get(),
		mainAllocator->GetCpuHandle(liquidTableStartSlot),
		mainAllocator->GetGpuHandle(liquidTableStartSlot),
		DXGI_FORMAT_R32_FLOAT);
}

// Load the cubemap texture; create GPU resources and descriptors.
// No GPU commands are recorded here - uploads happen later in UploadAll().
void PbfApp::InitBackground() {
	envTexture = Egg::Importer::ImportTextureCube(device.Get(), "../Media/pretty_sky.dds");

	// SRV at the cubemap slot (t0): sampled by the background pixel shader.
	cubemapSrvSlot = mainAllocator->Allocate();
	envTexture.CreateSRV(device.Get(), mainAllocator->GetHeap(), cubemapSrvSlot);
}

// Load the solid obstacle; create GPU resources and descriptors.
// No GPU commands are recorded here - uploads happen later in UploadAll().
void PbfApp::InitObstacle() {
	solidObstacle = SolidObstacle::Create();
	solidObstacle->Load(device.Get(), psoManager, "dragonite.obj", "dragonite.sdf", perFrameCb);

	// Allocates one slot in mainAllocator and creates SDF Texture3D SRV.
	solidObstacle->CreateSdfSrv(device.Get(), *mainAllocator);
	sdfHandle = solidObstacle->GetSdfGpuHandle();
}

// Batch all initial data uploads into a single command list execution.
// All operate on independent resources so there are no state conflicts.
// Most functions here will be recording GPU commands along the lines of:
// transition to dest -> copy buffer region ->transition to common/initial state
void PbfApp::UploadAll() {
	ParticleInitData initData = GenerateParticles(); // initData holds particle data on CPU
	FillUploadBuffers(initData); // memcpy particle data into the upload buffers on the GPU

	DX_API("Failed to reset command allocator (UploadAll)")
		commandAllocator->Reset();
	DX_API("Failed to reset command list (UploadAll)")
		commandList->Reset(commandAllocator.Get(), nullptr);

	envTexture.UploadResource(commandList.Get()); // record cubemap copy + barrier
	RecordParticleUpload(); // record particle copy + barriers
	solidObstacle->UploadSdf(commandList.Get()); // record SDF texture copy + barrier
	RecordSnapshotUpload(); // record initial state of snapshot buffers for frame 1

	// Pre-clear both depth texture slots to 1.0 (far plane) so the first DTVS compute frame
	// sees valid depth data even before any graphics depth pass has run.
	RecordDepthTextureClear();

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
void PbfApp::RecordSnapshotUpload() {
	D3D12_RESOURCE_BARRIER barriers[3];
	// Transition position source to COPY_SOURCE and both snapshotPosition slots to COPY_DEST.
	// Buffers always start in COMMON regardless of the InitialState passed to CreateCommittedResource.
	barriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(
		particleFields[PF_POSITION]->Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
	barriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(
		snapshotPosition[0]->Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
	barriers[2] = CD3DX12_RESOURCE_BARRIER::Transition(
		snapshotPosition[1]->Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
	commandList->ResourceBarrier(3, barriers);

	const UINT64 posBytes = (UINT64)numParticles * sizeof(Float3);
	commandList->CopyBufferRegion(snapshotPosition[0]->Get(), 0, particleFields[PF_POSITION]->Get(), 0, posBytes);
	commandList->CopyBufferRegion(snapshotPosition[1]->Get(), 0, particleFields[PF_POSITION]->Get(), 0, posBytes);

	barriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(
		particleFields[PF_POSITION]->Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	barriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(
		snapshotPosition[0]->Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	barriers[2] = CD3DX12_RESOURCE_BARRIER::Transition(
		snapshotPosition[1]->Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	commandList->ResourceBarrier(3, barriers);

	D3D12_RESOURCE_BARRIER toSrv[8] = {
		CD3DX12_RESOURCE_BARRIER::Transition(snapshotDensity[0]->Get(),
			D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
		CD3DX12_RESOURCE_BARRIER::Transition(snapshotDensity[1]->Get(),
			D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
		CD3DX12_RESOURCE_BARRIER::Transition(snapshotLod[0]->Get(),
			D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
		CD3DX12_RESOURCE_BARRIER::Transition(snapshotLod[1]->Get(),
			D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
		CD3DX12_RESOURCE_BARRIER::Transition(cellCountSnapshot[0]->Get(),
			D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
		CD3DX12_RESOURCE_BARRIER::Transition(cellCountSnapshot[1]->Get(),
			D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
		CD3DX12_RESOURCE_BARRIER::Transition(cellPrefixSumSnapshot[0]->Get(),
			D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
		CD3DX12_RESOURCE_BARRIER::Transition(cellPrefixSumSnapshot[1]->Get(),
			D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
	};
	commandList->ResourceBarrier(8, toSrv);
}

// Sets all depth pixels in both depth texture slots to 1.0 (far plane)
void PbfApp::RecordDepthTextureClear() {
	D3D12_RESOURCE_BARRIER toWrite[2] = {
		CD3DX12_RESOURCE_BARRIER::Transition(particleDepthTexture[0]->Get(),
			D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_DEPTH_WRITE),
		CD3DX12_RESOURCE_BARRIER::Transition(particleDepthTexture[1]->Get(),
			D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_DEPTH_WRITE),
	};
	commandList->ResourceBarrier(2, toWrite);
	for (int i = 0; i < 2; i++) {
		commandList->ClearDepthStencilView(
			particleDsvAllocator->GetCpuHandle(i), D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
	}
	D3D12_RESOURCE_BARRIER toCommon[2] = {
		CD3DX12_RESOURCE_BARRIER::Transition(particleDepthTexture[0]->Get(),
			D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COMMON),
		CD3DX12_RESOURCE_BARRIER::Transition(particleDepthTexture[1]->Get(),
			D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COMMON),
	};
	commandList->ResourceBarrier(2, toCommon);
}

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
void PbfApp::WaitFirstFrame() {
	frameCount = 1;
	computeFence.signal(commandQueue, frameCount); // when reached, we're done calculating frame 1
	cpuWaitForCompute(frameCount);
	// graphicsFence deliberately NOT pre-seeded here (see comment above).

	lastFrame = clock::now();
}

// Build all graphics rendering pipelines (background, particles, DTVS depth-only, liquid, solid transform).
void PbfApp::BuildGraphicsPipelines() {
	BuildBackgroundPipeline();
	BuildParticlePipeline();
	BuildParticleDepthOnlyPipeline();
	BuildLiquidPipeline();
	SetSolidTransform();
}

// Build the background skybox rendering pipeline (shaders, material, mesh).
// Called after all resources and descriptors are ready.
void PbfApp::BuildBackgroundPipeline() {
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
	bgMaterial->SetSrvHeap(1, mainAllocator->GetHeap(), cubemapSrvSlot * mainAllocator->GetDescriptorSize());

	// The fullscreen quad from Egg's prefab library - 2 triangles covering the entire screen
	// the geometry of a mesh is what handles raw vertex data
	Egg::Mesh::Geometry::P bgGeometry = Egg::Mesh::Prefabs::FullScreenQuad(device.Get());

	// Mesh = material + geometry
	backgroundMesh = Egg::Mesh::Shaded::Create(psoManager, bgMaterial, bgGeometry);
}

// Build the particle rendering pipeline (shaders, material, mesh).
// Called after all resources and descriptors are ready.
void PbfApp::BuildParticlePipeline() {
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
	material->SetSrvHeap(1, mainAllocator->GetHeap(), particleSrvTableStart * mainAllocator->GetDescriptorSize());

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
void PbfApp::BuildParticleDepthOnlyPipeline() {
	com_ptr<ID3DBlob> vertexShader = Egg::Shader::LoadCso("Shaders/particleVS.cso");
	com_ptr<ID3DBlob> geometryShader = Egg::Shader::LoadCso("Shaders/particleGS.cso");
	com_ptr<ID3DBlob> pixelShader = Egg::Shader::LoadCso("Shaders/dtvsDepthOnlyPS.cso");
	depthOnlyRootSig = Egg::Shader::LoadRootSignature(device.Get(), vertexShader.Get());

	D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
	psoDesc.pRootSignature = depthOnlyRootSig.Get();
	psoDesc.VS = { vertexShader->GetBufferPointer(), vertexShader->GetBufferSize() };
	psoDesc.GS = { geometryShader->GetBufferPointer(), geometryShader->GetBufferSize() };
	psoDesc.PS = { pixelShader->GetBufferPointer(), pixelShader->GetBufferSize() };
	psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
	psoDesc.SampleMask = UINT_MAX;
	psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
	psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT); // depth test + write on
	psoDesc.InputLayout = { nullptr, 0 }; // no vertex buffer: positions read from SRV
	psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
	psoDesc.NumRenderTargets = 0; // depth-only: no color output
	psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT; // match the depth texture format
	psoDesc.SampleDesc.Count = 1;
	psoDesc.SampleDesc.Quality = 0;

	DX_API("Failed to create particle depth-only PSO")
		device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(depthOnlyPso.ReleaseAndGetAddressOf()));
}

// Build the liquid surface rendering pipeline (liquidVS + liquidPS, fullscreen quad).
// The PS ray-marches through the density volume (t0 = densityVol SRV) and writes SV_Depth
// for correct depth-buffer occlusion against the solid obstacle.
void PbfApp::BuildLiquidPipeline() {
	com_ptr<ID3DBlob> vertexShader = Egg::Shader::LoadCso("Shaders/liquidVS.cso");
	com_ptr<ID3DBlob> pixelShader = Egg::Shader::LoadCso("Shaders/liquidPS.cso");
	// extract the root signature from the vertex shader blob (same LiquidRootSig as in PS)
	com_ptr<ID3D12RootSignature> rootSig = Egg::Shader::LoadRootSignature(device.Get(), vertexShader.Get());

	Egg::Mesh::Material::P material = Egg::Mesh::Material::Create();
	material->SetRootSignature(rootSig);
	material->SetVertexShader(vertexShader);
	material->SetPixelShader(pixelShader);
	// depth test + write: liquid SV_Depth correctly occludes / is occluded by the solid obstacle
	material->SetDepthStencilState(CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT));
	material->SetDSVFormat(DXGI_FORMAT_D32_FLOAT);
	// per-frame CB (root param 0): camera matrices, bbMin/bbMax, threshold
	material->SetConstantBuffer(perFrameCb);
	// Single 4-slot descriptor table (t0..t3) starting at liquidTableStartSlot.
	// density (static), position (per-frame), grid count (per-frame), grid prefix (per-frame).
	material->SetSrvHeap(1, mainAllocator->GetHeap(), liquidTableStartSlot * mainAllocator->GetDescriptorSize());

	// same fullscreen quad as the background skybox
	Egg::Mesh::Geometry::P geometry = Egg::Mesh::Prefabs::FullScreenQuad(device.Get());

	liquidMesh = Egg::Mesh::Shaded::Create(psoManager, material, geometry);
}

// Rebuild the solid's world transform from solidPosition and solidEulerDeg (XYZ Euler, degrees).
void PbfApp::SetSolidTransform() {
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


ParticleInitData PbfApp::GenerateParticles() {
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

// Map the upload buffers and copy initial particle data (positions + velocities) from CPU memory
// to the upload buffers (upload heap, as opposed to default heap).
// This is a CPU-side operation; the actual GPU transfer is recorded by RecordParticleUpload().
void PbfApp::FillUploadBuffers(const ParticleInitData& initData) {
	void* posData; // will point to the mapped CPU memory of the upload buffer
	CD3DX12_RANGE readRange(0, 0); // empty range: CPU won't read anything from this buffer
	// make the upload buffer's memory CPU accessible, i.e. positionUploadBuffer - posData association
	// 0 is subresource index, on success, posData points to the buffer
	DX_API("Failed to map position upload buffer")
		positionUploadBuffer->Get()->Map(0, &readRange, &posData);
	memcpy(posData, initData.positions.data(), initData.positions.size() * sizeof(Float3));
	positionUploadBuffer->Get()->Unmap(0, nullptr);

	void* velData;
	DX_API("Failed to map velocity upload buffer")
		velocityUploadBuffer->Get()->Map(0, &readRange, &velData);
	memcpy(velData, initData.velocities.data(), initData.velocities.size() * sizeof(Float3));
	velocityUploadBuffer->Get()->Unmap(0, nullptr);
}

// Record copy commands for particle data into the already-open command list.
// The command list must have been Reset() before calling this.
void PbfApp::RecordParticleUpload() {
	// In order to copy to them, we must transition position and velocity buffers to COPY_DEST
	// That's done by inserting transition type resource barriers to the command list.
	D3D12_RESOURCE_BARRIER toCopyDest[2] = {
		CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_POSITION]->Get(),
			D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
		CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_VELOCITY]->Get(),
			D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
	};
	commandList->ResourceBarrier(2, toCopyDest);
	commandList->CopyBufferRegion(particleFields[PF_POSITION]->Get(), 0,
		positionUploadBuffer->Get(), 0, numParticles * sizeof(Float3));
	commandList->CopyBufferRegion(particleFields[PF_VELOCITY]->Get(), 0,
		velocityUploadBuffer->Get(), 0, numParticles * sizeof(Float3));

	D3D12_RESOURCE_BARRIER toUav[2] = {
		CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_POSITION]->Get(),
			D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
		CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_VELOCITY]->Get(),
			D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
	};
	commandList->ResourceBarrier(2, toUav);
}

// Create all 16 compute shader PSOs and their descriptor table bindings.
// Requires CacheDescriptorHandles() to have been called first.
void PbfApp::BuildComputePipelines() {
	D3D12_GPU_VIRTUAL_ADDRESS cbv = computeCb.GetGPUVirtualAddress();

	using std::vector;
	using TableBinding = ComputeShader::TableBinding;
	using P = com_ptr<ID3D12Resource>*;

	predictShader = ComputeShader::Create(device.Get(), "Shaders/predictCS.cso", cbv,
		vector<TableBinding>{ {1, &particleFieldsHandle}, {2, &sdfHandle} },
		vector<P>{ particleFields[PF_POSITION]->GetResourcePtr(), particleFields[PF_VELOCITY]->GetResourcePtr() },
		vector<P>{ particleFields[PF_VELOCITY]->GetResourcePtr(), particleFields[PF_PREDICTED_POSITION]->GetResourcePtr() });

	collisionPredictedPositionShader = ComputeShader::Create(device.Get(), "Shaders/collisionPredictedPositionCS.cso", cbv,
		vector<TableBinding>{ {1, &particleFieldsHandle}, {2, &lodHandle}, {3, &sdfHandle} },
		vector<P>{ particleFields[PF_PREDICTED_POSITION]->GetResourcePtr(), lodBuffer->GetResourcePtr() },
		vector<P>{ particleFields[PF_PREDICTED_POSITION]->GetResourcePtr() });

	positionFromScratchShader = ComputeShader::Create(device.Get(), "Shaders/positionFromScratchCS.cso", cbv,
		vector<TableBinding>{ {1, &particleFieldsHandle}, {2, &lodHandle} },
		vector<P>{ particleFields[PF_SCRATCH]->GetResourcePtr(), lodBuffer->GetResourcePtr() },
		vector<P>{ particleFields[PF_PREDICTED_POSITION]->GetResourcePtr(), lodBuffer->GetResourcePtr() });

	updateVelocityShader = ComputeShader::Create(device.Get(), "Shaders/updateVelocityCS.cso", cbv,
		vector<TableBinding>{ {1, &particleFieldsHandle} },
		vector<P>{ particleFields[PF_POSITION]->GetResourcePtr(), particleFields[PF_PREDICTED_POSITION]->GetResourcePtr() },
		vector<P>{ particleFields[PF_VELOCITY]->GetResourcePtr() });

	velocityFromScratchShader = ComputeShader::Create(device.Get(), "Shaders/velocityFromScratchCS.cso", cbv,
		vector<TableBinding>{ {1, &particleFieldsHandle} },
		vector<P>{ particleFields[PF_SCRATCH]->GetResourcePtr() },
		vector<P>{ particleFields[PF_VELOCITY]->GetResourcePtr() });

	updatePositionShader = ComputeShader::Create(device.Get(), "Shaders/updatePositionCS.cso", cbv,
		vector<TableBinding>{ {1, &particleFieldsHandle} },
		vector<P>{ particleFields[PF_PREDICTED_POSITION]->GetResourcePtr() },
		vector<P>{ particleFields[PF_POSITION]->GetResourcePtr() });

	clearGridShader = ComputeShader::Create(device.Get(), "Shaders/clearGridCS.cso", cbv,
		vector<TableBinding>{ {1, &gridHandle} },
		vector<P>{ cellCountBuffer->GetResourcePtr() },
		vector<P>{ cellCountBuffer->GetResourcePtr() });

	// Three-pass Blelloch parallel prefix sum
	prefixSumPass1Shader = ComputeShader::Create(device.Get(), "Shaders/prefixSumPass1CS.cso", cbv,
		vector<TableBinding>{ {1, &gridHandle}, {2, &groupSumHandle} },
		vector<P>{ cellCountBuffer->GetResourcePtr() },
		vector<P>{ cellPrefixSumBuffer->GetResourcePtr(), groupSumBuffer->GetResourcePtr() });

	prefixSumPass2Shader = ComputeShader::Create(device.Get(), "Shaders/prefixSumPass2CS.cso", cbv,
		vector<TableBinding>{ {1, &groupSumHandle} },
		vector<P>{ groupSumBuffer->GetResourcePtr() },
		vector<P>{ groupSumBuffer->GetResourcePtr() });

	prefixSumPass3Shader = ComputeShader::Create(device.Get(), "Shaders/prefixSumPass3CS.cso", cbv,
		vector<TableBinding>{ {1, &cellPrefixSumHandle}, {2, &groupSumHandle} },
		vector<P>{ groupSumBuffer->GetResourcePtr(), cellPrefixSumBuffer->GetResourcePtr() },
		vector<P>{ cellPrefixSumBuffer->GetResourcePtr() });

	countGridShader = ComputeShader::Create(device.Get(), "Shaders/countGridCS.cso", cbv,
		vector<TableBinding>{ {1, &particleFieldsHandle}, {2, &gridHandle} },
		vector<P>{ particleFields[PF_PREDICTED_POSITION]->GetResourcePtr(), cellCountBuffer->GetResourcePtr() },
		vector<P>{ cellCountBuffer->GetResourcePtr() });

	lambdaShader = ComputeShader::Create(device.Get(), "Shaders/lambdaCS.cso", cbv,
		vector<TableBinding>{ {1, &particleFieldsHandle}, {2, &gridHandle}, {3, &lodHandle} },
		vector<P>{ particleFields[PF_PREDICTED_POSITION]->GetResourcePtr(), cellCountBuffer->GetResourcePtr(), cellPrefixSumBuffer->GetResourcePtr(), lodBuffer->GetResourcePtr() },
		vector<P>{ particleFields[PF_LAMBDA]->GetResourcePtr(), particleFields[PF_DENSITY]->GetResourcePtr() });

	deltaShader = ComputeShader::Create(device.Get(), "Shaders/deltaCS.cso", cbv,
		vector<TableBinding>{ {1, &particleFieldsHandle}, {2, &gridHandle}, {3, &lodHandle} },
		vector<P>{ particleFields[PF_PREDICTED_POSITION]->GetResourcePtr(), particleFields[PF_LAMBDA]->GetResourcePtr(), cellCountBuffer->GetResourcePtr(), cellPrefixSumBuffer->GetResourcePtr(), lodBuffer->GetResourcePtr() },
		vector<P>{ particleFields[PF_SCRATCH]->GetResourcePtr() });

	vorticityShader = ComputeShader::Create(device.Get(), "Shaders/vorticityCS.cso", cbv,
		vector<TableBinding>{ {1, &particleFieldsHandle}, {2, &gridHandle} },
		vector<P>{ particleFields[PF_POSITION]->GetResourcePtr(), particleFields[PF_VELOCITY]->GetResourcePtr(), cellCountBuffer->GetResourcePtr(), cellPrefixSumBuffer->GetResourcePtr() },
		vector<P>{ particleFields[PF_OMEGA]->GetResourcePtr() });

	confinementViscosityShader = ComputeShader::Create(device.Get(), "Shaders/confinementViscosityCS.cso", cbv,
		vector<TableBinding>{ {1, &particleFieldsHandle}, {2, &gridHandle} },
		vector<P>{ particleFields[PF_POSITION]->GetResourcePtr(), particleFields[PF_VELOCITY]->GetResourcePtr(), particleFields[PF_OMEGA]->GetResourcePtr(), cellCountBuffer->GetResourcePtr(), cellPrefixSumBuffer->GetResourcePtr() },
		vector<P>{ particleFields[PF_SCRATCH]->GetResourcePtr() });

	sortShader = ComputeShader::Create(device.Get(), "Shaders/sortCS.cso", cbv,
		vector<TableBinding>{ {1, &particleFieldsHandle}, {2, &gridHandle}, {3, &permHandle} },
		vector<P>{ particleFields[PF_PREDICTED_POSITION]->GetResourcePtr(), cellPrefixSumBuffer->GetResourcePtr(), cellCountBuffer->GetResourcePtr() },
		vector<P>{ permBuffer->GetResourcePtr(), cellCountBuffer->GetResourcePtr() });

	permutateShader = ComputeShader::Create(device.Get(), "Shaders/permutateCS.cso", cbv,
		vector<TableBinding>{ {1, &particleFieldsHandle}, {2, &sortedFieldsHandle}, {3, &permHandle} },
		vector<P>{ particleFields[PF_POSITION]->GetResourcePtr(), particleFields[PF_VELOCITY]->GetResourcePtr(), particleFields[PF_PREDICTED_POSITION]->GetResourcePtr(), particleFields[PF_LAMBDA]->GetResourcePtr(),
		           particleFields[PF_DENSITY]->GetResourcePtr(), particleFields[PF_OMEGA]->GetResourcePtr(), particleFields[PF_SCRATCH]->GetResourcePtr(), permBuffer->GetResourcePtr() },
		vector<P>{ sortedFields[PF_POSITION]->GetResourcePtr(), sortedFields[PF_VELOCITY]->GetResourcePtr(), sortedFields[PF_PREDICTED_POSITION]->GetResourcePtr(), sortedFields[PF_LAMBDA]->GetResourcePtr(),
		           sortedFields[PF_DENSITY]->GetResourcePtr(), sortedFields[PF_OMEGA]->GetResourcePtr(), sortedFields[PF_SCRATCH]->GetResourcePtr() });

	clearDtcReductionShader = ComputeShader::Create(device.Get(), "Shaders/clearDtcReductionCS.cso", cbv,
		vector<TableBinding>{ {1, &lodReductionHandle} },
		vector<P>{},
		vector<P>{ lodReductionBuffer->GetResourcePtr() });

	lodReductionShader = ComputeShader::Create(device.Get(), "Shaders/dtcReductionCS.cso", cbv,
		vector<TableBinding>{ {1, &particleFieldsHandle}, {2, &lodReductionHandle} },
		vector<P>{ particleFields[PF_PREDICTED_POSITION]->GetResourcePtr(), lodReductionBuffer->GetResourcePtr() },
		vector<P>{ lodReductionBuffer->GetResourcePtr() });

	lodShader = ComputeShader::Create(device.Get(), "Shaders/dtcLodCS.cso", cbv,
		vector<TableBinding>{ {1, &particleFieldsHandle}, {2, &lodHandle}, {3, &lodReductionHandle} },
		vector<P>{ particleFields[PF_PREDICTED_POSITION]->GetResourcePtr(), lodReductionBuffer->GetResourcePtr() },
		vector<P>{ lodBuffer->GetResourcePtr() });

	setLodMaxShader = ComputeShader::Create(device.Get(), "Shaders/setLodMaxCS.cso", cbv,
		vector<TableBinding>{ {1, &lodHandle} },
		vector<P>{},
		vector<P>{ lodBuffer->GetResourcePtr() });

	clearDtvsReductionShader = ComputeShader::Create(device.Get(), "Shaders/clearDtvsReductionCS.cso", cbv,
		vector<TableBinding>{ {1, &lodReductionHandle} },
		vector<P>{},
		vector<P>{ lodReductionBuffer->GetResourcePtr() });

	dtvsReductionShader = ComputeShader::Create(device.Get(), "Shaders/dtvsReductionCS.cso", cbv,
		vector<TableBinding>{ {1, &particleFieldsHandle}, {2, &lodReductionHandle}, {3, &particleDepthActiveHandle} },
		vector<P>{ particleFields[PF_PREDICTED_POSITION]->GetResourcePtr(), lodReductionBuffer->GetResourcePtr() },
		vector<P>{ lodReductionBuffer->GetResourcePtr() });

	dtvsLodShader = ComputeShader::Create(device.Get(), "Shaders/dtvsLodCS.cso", cbv,
		vector<TableBinding>{ {1, &particleFieldsHandle}, {2, &lodHandle}, {3, &lodReductionHandle}, {4, &particleDepthActiveHandle} },
		vector<P>{ particleFields[PF_PREDICTED_POSITION]->GetResourcePtr(), lodReductionBuffer->GetResourcePtr() },
		vector<P>{ lodBuffer->GetResourcePtr() });

	densityVolumeShader = ComputeShader::Create(device.Get(), "Shaders/densityVolumeCS.cso", cbv,
		vector<TableBinding>{ {1, &posSnapshotActiveHandle}, {2, &gridSnapshotActiveHandle}, {3, &densityVolumeHandle} },
		vector<P>{},
		vector<P>{});

	// densityVolume listed in outputs so dispatch_then_barrier emits the UAV barrier automatically.
	splatDensityShader = ComputeShader::Create(device.Get(), "Shaders/splatDensityVolumeCS.cso", cbv,
		vector<TableBinding>{ {1, &posSnapshotActiveHandle}, {2, &densityVolumeHandle} },
		vector<P>{},
		vector<P>{ densityVolume->GetResourcePtr() });

	gsmLambdaShader = ComputeShader::Create(device.Get(), "Shaders/GSM_lambdaCS.cso", cbv,
		vector<TableBinding>{ {1, &particleFieldsHandle}, {2, &gridHandle}, {3, &lodHandle} },
		vector<P>{ particleFields[PF_PREDICTED_POSITION]->GetResourcePtr(), cellCountBuffer->GetResourcePtr(), cellPrefixSumBuffer->GetResourcePtr(), lodBuffer->GetResourcePtr() },
		vector<P>{ particleFields[PF_LAMBDA]->GetResourcePtr(), particleFields[PF_DENSITY]->GetResourcePtr() });

	gsmDeltaShader = ComputeShader::Create(device.Get(), "Shaders/GSM_deltaCS.cso", cbv,
		vector<TableBinding>{ {1, &particleFieldsHandle}, {2, &gridHandle}, {3, &lodHandle} },
		vector<P>{ particleFields[PF_PREDICTED_POSITION]->GetResourcePtr(), particleFields[PF_LAMBDA]->GetResourcePtr(), cellCountBuffer->GetResourcePtr(), cellPrefixSumBuffer->GetResourcePtr(), lodBuffer->GetResourcePtr() },
		vector<P>{ particleFields[PF_SCRATCH]->GetResourcePtr() });

	gsmVorticityShader = ComputeShader::Create(device.Get(), "Shaders/GSM_vorticity.cso", cbv,
		vector<TableBinding>{ {1, &particleFieldsHandle}, {2, &gridHandle} },
		vector<P>{ particleFields[PF_POSITION]->GetResourcePtr(), particleFields[PF_VELOCITY]->GetResourcePtr(), cellCountBuffer->GetResourcePtr(), cellPrefixSumBuffer->GetResourcePtr() },
		vector<P>{ particleFields[PF_OMEGA]->GetResourcePtr() });

	gsmConfinementViscosityShader = ComputeShader::Create(device.Get(), "Shaders/GSM_confinementViscosityCS.cso", cbv,
		vector<TableBinding>{ {1, &particleFieldsHandle}, {2, &gridHandle} },
		vector<P>{ particleFields[PF_POSITION]->GetResourcePtr(), particleFields[PF_VELOCITY]->GetResourcePtr(), particleFields[PF_OMEGA]->GetResourcePtr(), cellCountBuffer->GetResourcePtr(), cellPrefixSumBuffer->GetResourcePtr() },
		vector<P>{ particleFields[PF_SCRATCH]->GetResourcePtr() });
}

void PbfApp::PrepareComputeCommandList() {
	// Reset compute allocator and command list for the next compute frame
	DX_API("Failed to reset compute allocator")
		computeAllocator->Reset();
	DX_API("Failed to reset compute list")
		computeList->Reset(computeAllocator.Get(), nullptr);

	// Bind the shared descriptor heap: must be done each time the compute list is reset
	ID3D12DescriptorHeap* computeHeaps[] = { mainAllocator->GetHeap() };
	computeList->SetDescriptorHeaps(1, computeHeaps);
}

void PbfApp::PrepareCommandList() {
	// reset the command allocator, freeing the memory used by the previous frame's commands
	// this can only be done after the GPU finished executing those commands
	DX_API("Failed to reset graphics command allocator")
		commandAllocator->Reset();

	// command list must be reset before we start recording commands into it
	// second param is initial pipeline state, don't need it yet
	DX_API("Failed to reset graphics command list")
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
	ID3D12DescriptorHeap* graphicsHeaps[] = { mainAllocator->GetHeap() };
	commandList->SetDescriptorHeaps(1, graphicsHeaps);
}

void PbfApp::ExecuteGraphics() {
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

void PbfApp::ExecuteCompute() {
	DX_API("Failed to close compute list")
		computeList->Close();
	ID3D12CommandList* computeCls[] = { computeList.Get() };
	computeCommandQueue->ExecuteCommandLists(_countof(computeCls), computeCls);
}

void PbfApp::SortParticles() {
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
	for (UINT f = 0; f < PF_COUNT; f++) GpuBuffer::SwapInternals(particleFields[f], sortedFields[f]);
}

// Override Render() to decouple physics (compute queue) from graphics (direct queue).
//
// Physics step: CPU waits for compute frame N-1 to finish (so the allocator can be reused),
// records compute frame N onto computeList, submits it, signals computeFence to N.
//
// Graphics step: graphics queue GPU-waits on frame N-1
// to ensure the snapshot is ready, records and submits the scene draw, presents,
// signals graphicsFence, and CPU-waits on it.
void PbfApp::Render() {
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

// writeIdx: which snapshot slot to write to this step (caller sets and flips).
void PbfApp::RecordComputeCommands(int writeIdx) {
	// ceil(numParticles / THREAD_GROUP_SIZE) groups cover all particles; the shader discards extra threads
	UINT numGroups = (numParticles + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;

	// apply forces, wall-correct velocity, predict p* = position + v*dt
	predictShader->dispatch_then_barrier(computeList.Get(), numGroups);

	// Clamp p* to the simulation box before building the spatial grid.
	collisionPredictedPositionShader->dispatch_then_barrier(computeList.Get(), numGroups);

	SortParticles(); // sort particle data for improved cache coherence -> fewer cache misses

	CalculateLod(writeIdx); // calculate LOD based on predicted position, write to lodBuffer for use in the solver and graphics

	// constraint solver loop
	// particles are now in grid-sorted order, and cellCount + cellPrefixSum describe
	// exactly where each cell's particles live in the buffer, so neighbor lookups
	// use simple arithmetic: particles[cellPrefixSum[ci] + s] for s in [0, cellCount[ci])
	for (int iter = 0; iter < solverIterations; iter++) {
		(gsmEnabled ? gsmLambdaShader : lambdaShader)->dispatch_then_barrier(computeList.Get(), numGroups); // compute lambda and density
		(gsmEnabled ? gsmDeltaShader : deltaShader)->dispatch_then_barrier(computeList.Get(), numGroups); // delta_p -> scratch
		lambdaShader->dispatch_then_barrier(computeList.Get(), numGroups); // compute lambda and density
		deltaShader->dispatch_then_barrier(computeList.Get(), numGroups); // delta_p -> scratch
		positionFromScratchShader->dispatch_then_barrier(computeList.Get(), numGroups); // scratch -> predictedPosition
		collisionPredictedPositionShader->dispatch_then_barrier(computeList.Get(), numGroups); // rectify collisions
	}

	updateVelocityShader->dispatch_then_barrier(computeList.Get(), numGroups); // v = (p* - x) / dt
	(gsmEnabled ? gsmVorticityShader : vorticityShader)->dispatch_then_barrier(computeList.Get(), numGroups); // estimate curl(v) -> omega
	vorticityShader->dispatch_then_barrier(computeList.Get(), numGroups); // estimate curl(v) -> omega
	(gsmEnabled ? gsmConfinementViscosityShader : confinementViscosityShader)->dispatch_then_barrier(computeList.Get(), numGroups); // vorticity confinement + XSPH viscosity -> scratch
	velocityFromScratchShader->dispatch_then_barrier(computeList.Get(), numGroups); // scratch -> velocity
	updatePositionShader->dispatch_then_barrier(computeList.Get(), numGroups); // position = predictedPosition

	WriteSnapshot(writeIdx); // copy positions, densities, and grid data into snapshot slot [writeIdx] for the graphics queue
}

void PbfApp::WriteSnapshot(int writeIdx) {
	// Write snapshot: copy position and density into snapshot slot [writeIdx].
	// Transition particle buffers to COPY_SOURCE, snapshot buffers to COPY_DEST.
	D3D12_RESOURCE_BARRIER toCopySrc[2] = {
		CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_POSITION]->Get(),
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
		CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_DENSITY]->Get(),
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
	};
	D3D12_RESOURCE_BARRIER snapshotToDest[2] = {
		CD3DX12_RESOURCE_BARRIER::Transition(snapshotPosition[writeIdx]->Get(),
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
		CD3DX12_RESOURCE_BARRIER::Transition(snapshotDensity[writeIdx]->Get(),
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
	};
	// insert them into the command list so the GPU knows about the state changes before the copy calls
	computeList->ResourceBarrier(2, toCopySrc);
	computeList->ResourceBarrier(2, snapshotToDest);

	computeList->CopyBufferRegion(snapshotPosition[writeIdx]->Get(), 0,
		particleFields[PF_POSITION]->Get(), 0, (UINT64)numParticles * sizeof(Float3));
	computeList->CopyBufferRegion(snapshotDensity[writeIdx]->Get(), 0,
		particleFields[PF_DENSITY]->Get(), 0, (UINT64)numParticles * sizeof(float));

	// Copy density to readback buffer (CPU reads it after the next cpuWaitForCompute).
	computeList->CopyBufferRegion(densityReadbackBuffer->Get(), 0,
		particleFields[PF_DENSITY]->Get(), 0, (UINT64)numParticles * sizeof(float));

	// Transition everything back to its home state.
	D3D12_RESOURCE_BARRIER toUav[2] = {
		CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_POSITION]->Get(),
			D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
		CD3DX12_RESOURCE_BARRIER::Transition(particleFields[PF_DENSITY]->Get(),
			D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
	};
	D3D12_RESOURCE_BARRIER snapshotToSrv[2] = {
		CD3DX12_RESOURCE_BARRIER::Transition(snapshotPosition[writeIdx]->Get(),
			D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
		CD3DX12_RESOURCE_BARRIER::Transition(snapshotDensity[writeIdx]->Get(),
			D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
	};
	computeList->ResourceBarrier(2, toUav);
	computeList->ResourceBarrier(2, snapshotToSrv);

	// Copy grid buffers into snapshot slot [writeIdx] so the graphics queue can read them
	// as SRVs in the next frame's densityVolumeCS dispatch. The cell buffers are in UAV state here
	// (returned to that state by clearGridCS/countGridCS/prefixSumCS dispatch_then_barrier chains).
	D3D12_RESOURCE_BARRIER gridToCopySrc[2] = {
		CD3DX12_RESOURCE_BARRIER::Transition(cellCountBuffer->Get(),
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
		CD3DX12_RESOURCE_BARRIER::Transition(cellPrefixSumBuffer->Get(),
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
	};
	D3D12_RESOURCE_BARRIER gridSnapToDest[2] = {
		CD3DX12_RESOURCE_BARRIER::Transition(cellCountSnapshot[writeIdx]->Get(),
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
		CD3DX12_RESOURCE_BARRIER::Transition(cellPrefixSumSnapshot[writeIdx]->Get(),
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
	};
	computeList->ResourceBarrier(2, gridToCopySrc);
	computeList->ResourceBarrier(2, gridSnapToDest);

	const UINT64 gridBufSize = (UINT64)numCells * sizeof(UINT);
	computeList->CopyBufferRegion(cellCountSnapshot[writeIdx]->Get(), 0,
		cellCountBuffer->Get(), 0, gridBufSize);
	computeList->CopyBufferRegion(cellPrefixSumSnapshot[writeIdx]->Get(), 0,
		cellPrefixSumBuffer->Get(), 0, gridBufSize);

	D3D12_RESOURCE_BARRIER gridBackToUav[2] = {
		CD3DX12_RESOURCE_BARRIER::Transition(cellCountBuffer->Get(),
			D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
		CD3DX12_RESOURCE_BARRIER::Transition(cellPrefixSumBuffer->Get(),
			D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
	};
	D3D12_RESOURCE_BARRIER gridSnapToSrv[2] = {
		CD3DX12_RESOURCE_BARRIER::Transition(cellCountSnapshot[writeIdx]->Get(),
			D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
		CD3DX12_RESOURCE_BARRIER::Transition(cellPrefixSumSnapshot[writeIdx]->Get(),
			D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
	};
	computeList->ResourceBarrier(2, gridBackToUav);
	computeList->ResourceBarrier(2, gridSnapToSrv);
}

void PbfApp::CalculateLod(int writeIdx) {
	// ceil(numParticles / THREAD_GROUP_SIZE) groups cover all particles; the shader discards extra threads
	UINT numGroups = (numParticles + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;

	// APBF LOD assignment
	if (lodMode == LodMode::DTC) {
		// DTC: reduce per-frame min/max camera distance, interpolate LOD
		clearDtcReductionShader->dispatch_then_barrier(computeList.Get(), 1); // clear the max/min DTC values
		lodReductionShader->dispatch_then_barrier(computeList.Get(), numGroups); // re-calculate the max/min DTC values
		lodShader->dispatch_then_barrier(computeList.Get(), numGroups);
	}
	else if (lodMode == LodMode::DTVS) {
		// DTVS: read particleDepthTexture[writeIdx] - the slot graphics is NOT writing this frame.
		// select the correct slot for dispatch, which is the one NOT being written by graphics this frame,
		// i.e. particleDepthTexture[writeIdx]
		// writeIdx is a bit confusing here, since we actually *read* the depth texture at writeIdx,
		// but writeIdx actually refers to the snapshot slot that compute is writing this frame, 
		// which is the same slot that graphics is reading this frame, opposite that the graphics is touching
		particleDepthActiveHandle = particleDepthHandle[writeIdx];
		// the depth texture is currently in COMMON state, so transition it to SRV state for the shader to read
		D3D12_RESOURCE_BARRIER toSrv = CD3DX12_RESOURCE_BARRIER::Transition(
			particleDepthTexture[writeIdx]->Get(),
			D3D12_RESOURCE_STATE_COMMON,
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
		computeList->ResourceBarrier(1, &toSrv);

		clearDtvsReductionShader->dispatch_then_barrier(computeList.Get(), 1);
		dtvsReductionShader->dispatch_then_barrier(computeList.Get(), numGroups);
		dtvsLodShader->dispatch_then_barrier(computeList.Get(), numGroups);

		D3D12_RESOURCE_BARRIER backToCommon = CD3DX12_RESOURCE_BARRIER::Transition(
			particleDepthTexture[writeIdx]->Get(),
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
			D3D12_RESOURCE_STATE_COMMON);
		computeList->ResourceBarrier(1, &backToCommon);
	}
	else {
		// NONE: every particle runs the full solver (maxLOD iterations)
		setLodMaxShader->dispatch_then_barrier(computeList.Get(), numGroups);
	}

	// Snapshot LOD immediately after assignment - the solver loop decrements lodBuffer
	// each iteration, so by the end of the loop all values
	// would be 0. We capture the initial per-particle LOD here, before any decrement.
	D3D12_RESOURCE_BARRIER toLodSrc = CD3DX12_RESOURCE_BARRIER::Transition(
		lodBuffer->Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
	D3D12_RESOURCE_BARRIER toLodDest = CD3DX12_RESOURCE_BARRIER::Transition(
		snapshotLod[writeIdx]->Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
	computeList->ResourceBarrier(1, &toLodSrc);
	computeList->ResourceBarrier(1, &toLodDest);
	computeList->CopyBufferRegion(snapshotLod[writeIdx]->Get(), 0,
		lodBuffer->Get(), 0, (UINT64)numParticles * sizeof(UINT));
	// Copy LOD to readback buffer as well, making use of the fact that we've already
	// transitioned the lodBuffer to copy source (CPU reads it after the next cpuWaitForCompute).
	computeList->CopyBufferRegion(lodReadbackBuffer->Get(), 0,
		lodBuffer->Get(), 0, (UINT64)numParticles * sizeof(UINT));
	D3D12_RESOURCE_BARRIER fromLodSrc = CD3DX12_RESOURCE_BARRIER::Transition(
		lodBuffer->Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	D3D12_RESOURCE_BARRIER fromLodDest = CD3DX12_RESOURCE_BARRIER::Transition(
		snapshotLod[writeIdx]->Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	computeList->ResourceBarrier(1, &fromLodSrc);
	computeList->ResourceBarrier(1, &fromLodDest);
}

// readIdx: which snapshot slot the graphics queue reads from (always 1 - snapshotWriteIdx).
void PbfApp::RecordGraphicsCommands(int readIdx) {
	backgroundMesh->Draw(commandList.Get()); // draw skybox at the back first
	solidObstacle->Draw(commandList.Get());  // draw solid with depth test before particles

	// Before the particle draw, redirect particle SRV table slots to the active snapshot.
	// The particle VS fetches position (t0), density (t1), and LOD (t2) from three contiguous
	// slots starting at particleSrvTableStart. CopyDescriptorsSimple overwrites each slot so
	// the shader reads the latest complete snapshot without any root signature change.
	device->CopyDescriptorsSimple(1,
		mainAllocator->GetCpuHandle(particleSrvTableStart),
		snapshotPosition[readIdx]->GetSrvCpuHandle(),
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	device->CopyDescriptorsSimple(1,
		mainAllocator->GetCpuHandle(particleSrvTableStart + 1),
		snapshotDensity[readIdx]->GetSrvCpuHandle(),
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	device->CopyDescriptorsSimple(1,
		mainAllocator->GetCpuHandle(particleSrvTableStart + 2),
		snapshotLod[readIdx]->GetSrvCpuHandle(),
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

	// In liquid mode snapshotPosition[readIdx] is read by liquidPS from the pixel stage (t1).
	// DrawParticleDepth (DTVS) also binds it via an ALL-visible descriptor table on the graphics
	// pipeline, which makes the D3D12 debug layer implicitly promote the resource to
	// NON_PIXEL|PIXEL at first use - before the explicit snapToPixel barrier inside
	// DrawLiquidSurface. That causes a "before state mismatch" validation error (#527).
	// Promoting explicitly here, before any draw, keeps the tracked state consistent.
	constexpr D3D12_RESOURCE_STATES SRV_ALL =
		D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
	if (shadingMode == ShadingMode::LIQUID) {
		D3D12_RESOURCE_BARRIER snapPosToPixel = CD3DX12_RESOURCE_BARRIER::Transition(
			snapshotPosition[readIdx]->Get(),
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, SRV_ALL);
		commandList->ResourceBarrier(1, &snapPosToPixel);
	}

	// DTVS depth-only pass: render billboard particles into particleDepthTexture[readIdx]
	// (the slot compute is NOT reading this frame) so compute can sample it next frame.
	// Run regardless of shading mode: the DTVS LOD calculation in the compute queue still needs it.
	if (lodMode == LodMode::DTVS) DrawParticleDepth(readIdx);

	// In liquid mode, the ray-marched surface replaces the individual particle billboards.
	if (shadingMode == ShadingMode::LIQUID) {
		DrawLiquidSurface(readIdx);
		// Return snapshotPosition to NON_PIXEL home state after all pixel-stage accesses.
		D3D12_RESOURCE_BARRIER snapPosToNonPixel = CD3DX12_RESOURCE_BARRIER::Transition(
			snapshotPosition[readIdx]->Get(),
			SRV_ALL, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
		commandList->ResourceBarrier(1, &snapPosToNonPixel);
	}
	else {
		particleMesh->Draw(commandList.Get()); // draw particle billboards
	}
}

// Fill the density volume and draw the ray-marched liquid surface, all on the graphics command list.
// densityVolumeCS is dispatched here (graphics queue) reading from the previous frame's position and
// grid snapshots at [readIdx]. liquidPS then ray-marches through the freshly filled volume in the same frame.
// The GPU-wait at the front of the graphics list (graphicsWaitForCompute(N-1)) guarantees that
// snapshotPosition[readIdx] and the grid snapshots are fully written by compute before this runs.
void PbfApp::DrawLiquidSurface(int readIdx) {
	// Point the active handle members at the correct snapshot slot; TableBindings dereference them at dispatch time.
	posSnapshotActiveHandle  = posSnapshotGfxHandle[readIdx];
	gridSnapshotActiveHandle = gridSnapshotHandle[readIdx];

	// Update the liquid table t1 (pos) and t2+t3 (gridCount, gridPrefix) each frame.
	device->CopyDescriptorsSimple(1,
		mainAllocator->GetCpuHandle(liquidTableStartSlot + 1),
		snapshotPosition[readIdx]->GetSrvCpuHandle(),
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	// Staging SRVs for cellCount[readIdx] and cellPrefixSum[readIdx] are contiguous; copy both at once.
	device->CopyDescriptorsSimple(2,
		mainAllocator->GetCpuHandle(liquidTableStartSlot + 2),
		cellCountSnapshot[readIdx]->GetSrvCpuHandle(),
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

	// Transition densityVolume from COMMON to UAV for the splat pass.
	D3D12_RESOURCE_BARRIER toUav = CD3DX12_RESOURCE_BARRIER::Transition(
		densityVolume->Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	commandList->ResourceBarrier(1, &toUav);

	// Splat: one thread per particle; each writes Poly6 to densityVolume via CAS float atomic add.
	// dispatch_then_barrier emits a UAV barrier on densityVolume (it's in outputs), ensuring all
	// splat writes are visible to the liquidPS SRV read that follows.
	UINT numGroups = ((UINT)numParticles + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
	splatDensityShader->dispatch_then_barrier(commandList.Get(), numGroups);

	// Transition densityVolume UAV -> combined SRV state for liquidPS.
	// Both NON_PIXEL_SHADER_RESOURCE and PIXEL_SHADER_RESOURCE bits must be set:
	// D3D12 validates that descriptor-table SRVs carry both flags (#538 error otherwise).
	constexpr D3D12_RESOURCE_STATES SRV_ALL =
		D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
	D3D12_RESOURCE_BARRIER toSrv = CD3DX12_RESOURCE_BARRIER::Transition(
		densityVolume->Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, SRV_ALL);
	commandList->ResourceBarrier(1, &toSrv);

	D3D12_RESOURCE_BARRIER snapToPixel[2] = {
		CD3DX12_RESOURCE_BARRIER::Transition(cellCountSnapshot[readIdx]->Get(),
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
		CD3DX12_RESOURCE_BARRIER::Transition(cellPrefixSumSnapshot[readIdx]->Get(),
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
	};
	commandList->ResourceBarrier(2, snapToPixel);

	liquidMesh->Draw(commandList.Get());

	D3D12_RESOURCE_BARRIER snapToNonPixel[2] = {
		CD3DX12_RESOURCE_BARRIER::Transition(cellCountSnapshot[readIdx]->Get(),
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
		CD3DX12_RESOURCE_BARRIER::Transition(cellPrefixSumSnapshot[readIdx]->Get(),
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
			D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
	};
	commandList->ResourceBarrier(2, snapToNonPixel);

	D3D12_RESOURCE_BARRIER srvToUav = CD3DX12_RESOURCE_BARRIER::Transition(
		densityVolume->Get(), SRV_ALL, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	commandList->ResourceBarrier(1, &srvToUav);

	const UINT clearVal[4] = { 0u, 0u, 0u, 0u };
	commandList->ClearUnorderedAccessViewUint(
		densityVolumeHandle, densityVolClearCpuHandle,
		densityVolume->Get(), clearVal, 0, nullptr);

	D3D12_RESOURCE_BARRIER uavToCommon = CD3DX12_RESOURCE_BARRIER::Transition(
		densityVolume->Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
	commandList->ResourceBarrier(1, &uavToCommon);
}

// Record the DTVS depth-only particle draw into the already-open graphics command list.
// Writes into particleDepthTexture[readIdx] - the slot NOT being read by compute this frame.
// Leaves the texture in COMMON state so compute can read it next frame.
void PbfApp::DrawParticleDepth(int readIdx) {
	D3D12_RESOURCE_BARRIER toDepthWrite = CD3DX12_RESOURCE_BARRIER::Transition(
		particleDepthTexture[readIdx]->Get(),
		D3D12_RESOURCE_STATE_COMMON,
		D3D12_RESOURCE_STATE_DEPTH_WRITE);
	commandList->ResourceBarrier(1, &toDepthWrite);

	D3D12_CPU_DESCRIPTOR_HANDLE particleDepthDsv = particleDsvAllocator->GetCpuHandle(readIdx);

	// Depth-only render target: no RTV, just the DSV
	commandList->OMSetRenderTargets(0, nullptr, FALSE, &particleDepthDsv);
	commandList->ClearDepthStencilView(particleDepthDsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

	// Bind the depth-only PSO and root signature
	commandList->SetGraphicsRootSignature(depthOnlyRootSig.Get());
	commandList->SetPipelineState(depthOnlyPso.Get());
	commandList->SetGraphicsRootConstantBufferView(0, perFrameCb.GetGPUVirtualAddress());

	// Root param 1: SRV table (pos/density/lod) - pos at t0 drives the VS; same table as main draw.
	commandList->SetGraphicsRootDescriptorTable(1, mainAllocator->GetGpuHandle(particleSrvTableStart));

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
		particleDepthTexture[readIdx]->Get(),
		D3D12_RESOURCE_STATE_DEPTH_WRITE,
		D3D12_RESOURCE_STATE_COMMON);
	commandList->ResourceBarrier(1, &toCommon);
}

void PbfApp::BuildImGui() {
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
	static const char* shadingModeItems[] = { "Unicolor", "Density", "LOD", "Liquid" };
	ImGui::Combo("Shading", &shadingMode, shadingModeItems, IM_ARRAYSIZE(shadingModeItems));
	if (shadingMode == ShadingMode::LIQUID)
		ImGui::InputFloat("Liquid iso threshold", &liquidIsoThreshold, 1.0f, 10.0f, "%.1f");
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
	ImGui::SameLine();
	ImGui::Checkbox("GSM", &gsmEnabled);
	ImGui::PopItemWidth(); // restore default width for any subsequent widgets
	// show derived values as read-only text for reference
	ImGui::Separator(); // horizontal line to separate tunable parameters from derived values
	ImGui::Text("%d particles, %u cells", numParticles, gridDim * gridDim * gridDim);
	ImGui::Text("%.1f FPS, render: %.2f ms", ImGui::GetIO().Framerate, debugTimer);
	ImGui::Text("avg density: %.2f (rho0: %.2f)", avgDensity, rho0);
	ImGui::Text("avg LOD: %.2f", avgLod);
	ImGui::Separator();
	ImGui::Text("Dragonite");
	ImGui::PushItemWidth(200);
	ImGui::DragFloat3("Pos", &solidPosition.x, 0.1f);
	ImGui::DragFloat3("Rot (deg)", &solidEulerDeg.x, 1.0f);
	ImGui::DragFloat("Scale", &solidScale, 0.01f, 0.01f, 100.0f);
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
	ID3D12DescriptorHeap* imguiHeaps[] = { imguiAllocator->GetHeap() };
	commandList->SetDescriptorHeaps(1, imguiHeaps);
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

void PbfApp::UpdateExternalForce() {
	// build a horizontal acceleration vector from held arrow keys
	// left/right push along X, up/down push along Z
	externalForce = Float3(0.0f, 0.0f, 0.0f);
	if (arrowLeft) externalForce.x -= externalAcceleration;
	if (arrowRight) externalForce.x += externalAcceleration;
	if (arrowUp) externalForce.z += externalAcceleration;
	if (arrowDown) externalForce.z -= externalAcceleration;
}

void PbfApp::UpdatePerFrameCb() {
	perFrameCb->viewProjTransform = // calculate the combined view-projection matrix and store it in the constant buffer
		camera->GetViewMatrix() * // view matrix: world space -> camera space
		camera->GetProjMatrix(); // projection matrix: camera space -> clip space
	perFrameCb->rayDirTransform = camera->GetRayDirMatrix(); // clip-space coords -> world-space view direction
	perFrameCb->cameraPos = Egg::Math::Float4(camera->GetEyePosition(), 1.0f);
	perFrameCb->lightDir = Egg::Math::Float4(0.5f, 1.0f, 0.3f, 0.0f); // light pointing down-left
	perFrameCb->particleParams = Float4(rho0, 0.0f, 0.0f, PARTICLE_RADIUS); // x = rho0 (for density coloring in PS), w = particle display radius (for billboard sizing in GS)
	perFrameCb->shadingMode = (UINT)shadingMode;
	perFrameCb->minLOD = (UINT)minLOD;
	perFrameCb->maxLOD = (UINT)solverIterations;
	// bbMin.xyz = adjustable collision box min; bbMin.w = liquid density iso-surface threshold.
	// bbMax.xyz = collision box max; bbMax.w unused.
	perFrameCb->bbMin = Float4(boxMin, liquidIsoThreshold);
	perFrameCb->bbMax = Float4(boxMax, 0.0f);
	perFrameCb.Upload(); // memcpy the data to the GPU-visible constant buffer
}

void PbfApp::UpdateComputeCb(float dt) {
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
	computeCb->viewportWidth = (float)scissorRect.right;
	computeCb->viewportHeight = (float)scissorRect.bottom;
	computeCb->pushRadius = (shadingMode == ShadingMode::LIQUID) ? 0.0f : PUSH_RADIUS;
	computeCb.Upload();
}

void PbfApp::Update(float dt, float T)  {
	camera->Animate(dt); // real dt for responsive camera
	lastDt = std::min(dt, 1.0f / 25.0f); // cap at 25Hz: prevents energy spikes on window drag or stutter
	UpdateExternalForce();
	UpdatePerFrameCb();
	SetSolidTransform();
}

void PbfApp::CalculateAvgDensity() {
	// map the readback buffer to CPU memory and copy the density data into a vector
	const UINT64 bufferSize = numParticles * sizeof(float);
	void* pData; // ptr will be set by Map to point at the readback buffer's CPU visible memory
	CD3DX12_RANGE readRange(0, bufferSize);
	// in the Map call, we map with the range we intend to read
	if (SUCCEEDED(densityReadbackBuffer->Get()->Map(0, &readRange, &pData))) { // prepare pData for reading
		memcpy(densityReadbackData.data(), pData, bufferSize); // actual data movement call
		// during the unmap, we unmap while indicating which bytes we dirtied
		CD3DX12_RANGE writeRange(0, 0); // in this case, we wrote nothing
		densityReadbackBuffer->Get()->Unmap(0, &writeRange); // release mapping: invalidate pData
	}

	// Compute average density from readback data
	double densitySum = 0.0;
	int cnt = 0;
	for (int i = 0; i < numParticles; i += AVG_COARSENESS) {
		densitySum += densityReadbackData[i];
		cnt++;
	}

	avgDensity = static_cast<float>(densitySum / cnt);
}

void PbfApp::CalculateAvgLod() {
	// map the readback buffer to CPU memory and copy the LOD data into a vector
	const UINT64 bufferSize = numParticles * sizeof(uint32_t);
	void* pData; // ptr will be set by Map to point at the readback buffer's CPU visible memory
	CD3DX12_RANGE readRange(0, bufferSize);
	// in the Map call, we map with the range we intend to read
	if (SUCCEEDED(lodReadbackBuffer->Get()->Map(0, &readRange, &pData))) { // prepare pData for reading
		memcpy(lodReadbackData.data(), pData, bufferSize); // actual data movement call
		CD3DX12_RANGE writeRange(0, 0); // in this case, we wrote nothing
		lodReadbackBuffer->Get()->Unmap(0, &writeRange); // release mapping: invalidate pData
	}

	// Compute average LOD from readback data
	double lodSum = 0.0;
	int cnt = 0;
	for (int i = 0; i < numParticles; i += AVG_COARSENESS) {
		lodSum += lodReadbackData[i];
		cnt++;
	}

	avgLod = static_cast<float>(lodSum / cnt);
}

// This function cannot be called more than once every targetPeriod time: rate limit
// a better way of doing this would be a fixed timestep accumulation, where we decouple
// physics dt from render dt entirely, accumulate wall-clock time, and step physics at a 
// fixed interval
void PbfApp::Throttle() {
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

// Recreate the window-resolution depth textures whenever the swap chain is (re)created.
void PbfApp::CreateSwapChainResources()  {
	AsyncComputeApp::CreateSwapChainResources(); // base class: RTVs, DSV, viewport/scissorRect
	InitParticleDepthTextures();

	// InitParticleDepthSrvs must be re-ran, because the underlying depth texture resources were recreated 
	// with new sizes, so the SRV descriptors must be updated to point to the new resources. 
	// However, the heap itself might not exist yet on the first call, so guard against that with the nullptr check.
	if (mainAllocator != nullptr) InitParticleDepthSrvs();
}

void PbfApp::ReleaseSwapChainResources()  {
	particleDepthTexture[0] = nullptr;
	particleDepthTexture[1] = nullptr;
	particleDsvAllocator = nullptr;
	AsyncComputeApp::ReleaseSwapChainResources();
}

// Allocate all GPU resources that persist across frames: descriptor heaps, 
// buffers for particles and sorting, textures for the environment and obstacle, etc.
// After this returns, every ID3D12Resource and descriptor heap slot exists,
// but no data has been uploaded to the GPU yet.
void PbfApp::CreateResources() {
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
	InitGridSnapshotBuffers();
	InitDensityVolume();
	InitBackground();
	InitObstacle();
	// depth textures already exist (InitParticleDepthTextures was called from CreateSwapChainResources);
	// write their SRVs into the main heap now that descriptorHeap exists.
	InitParticleDepthSrvs();
}

// upload initial data to the GPU and build rendering/compute pipelines.
void PbfApp::LoadAssets() {
	UploadAll();
	BuildGraphicsPipelines();
	BuildComputePipelines();
}

// Call once after CreateResources + LoadAssets, from main.cpp where the HWND is available.
// Sets up ImGui context and its Win32 + D3D12 backends. At this point the D3D12 device, command queue, and
//imguiSrvHeap all exist.
void PbfApp::InitImGui(HWND hwnd) {
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
	initInfo.SrvDescriptorHeap = imguiAllocator->GetHeap();
	initInfo.LegacySingleSrvCpuDescriptor = imguiAllocator->GetCpuHandle(0);
	initInfo.LegacySingleSrvGpuDescriptor = imguiAllocator->GetGpuHandle(0);
	ImGui_ImplDX12_Init(&initInfo);
}

void PbfApp::ShutdownImGui() {
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
void PbfApp::ProcessMessage(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
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
