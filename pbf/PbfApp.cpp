#include "PbfApp.h"

using namespace Egg::Math;

// Recreate the window-resolution depth textures whenever the swap chain is (re)created.
void PbfApp::CreateSwapChainResources() {
	AsyncComputeApp::CreateSwapChainResources();
	InitParticleDepthTextures();
	// On resize (mainAllocator exists), InitParticleDepthSrvs is a no-op (DB already exists;
	// resize2D above rebuilt its descriptors in-place).
	if (mainAllocator != nullptr) InitParticleDepthSrvs();
}

// Create the two window-resolution depth textures and their 2-slot DSV heap.
// Called from CreateSwapChainResources (and again on resize). Both textures start in COMMON state.
// R32_TYPELESS allows both D32_FLOAT DSV writes (graphics) and R32_FLOAT SRV reads (compute DTVS).
void PbfApp::InitParticleDepthTextures() {
	UINT width  = (UINT)scissorRect.right;
	UINT height = (UINT)scissorRect.bottom;

	if (!particleDepthDB) {
		// First call (before mainAllocator/staticAllocator exist): only create the DSV heap.
		// particleDepthDB itself is created in InitParticleDepthSrvs once mainAllocator is ready.
		particleDsvAllocator = DescriptorAllocator::Create(
			device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 2, /*shaderVisible*/false);
	} else {
		// Resize: recreate both textures at new dimensions. rebuildDescriptors reuses the same
		// static-heap DSV/SRV slots so registered main-heap targets stay valid.
		particleDepthDB->resize2D(width, height);
	}
}

// Create the particle depth DoubleBufferGpuTexture and its descriptors.
// Called once from CreateResources. On resize, InitParticleDepthTextures calls resize2D instead.
void PbfApp::InitParticleDepthSrvs() {
	if (particleDepthDB) return; // already created; resize2D handles updates

	UINT width  = (UINT)scissorRect.right;
	UINT height = (UINT)scissorRect.bottom;

	D3D12_CLEAR_VALUE clearValue = {};
	clearValue.Format = DXGI_FORMAT_D32_FLOAT;
	clearValue.DepthStencil.Depth = 1.0f;

	particleDepthDB = DoubleBufferGpuTexture::Create(
		device.Get(), width, height,
		DXGI_FORMAT_R32_TYPELESS,
		D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
		L"Particle Depth Texture [0] (DTVS)",
		L"Particle Depth Texture [1] (DTVS)",
		D3D12_RESOURCE_STATE_COMMON,
		&clearValue,
		*staticAllocator,
		particleDsvAllocator.get(),
		/*needUav*/false, /*needSrv*/true, /*needDsv*/true,
		DXGI_FORMAT_UNKNOWN,        // uavFmt (unused)
		DXGI_FORMAT_R32_FLOAT,      // srvFmt
		DXGI_FORMAT_D32_FLOAT);     // dsvFmt
}

// Create all descriptor allocators. Must be called before any Init function
// that populates descriptors.
void PbfApp::InitDescriptorHeaps() {
	// ImGui SRV heap: 1 slot, shader-visible, exclusively for ImGui's font texture atlas.
	imguiAllocator = DescriptorAllocator::Create(
		device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, /*shaderVisible*/true);

	// Main shader-visible heap: 256 slots for all per-shader contiguous regions + graphics tables.
	mainAllocator = DescriptorAllocator::Create(
		device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 256, /*shaderVisible*/true);

	// CPU-only static heap: UAV/SRV descriptors written once; source for all CopyDescriptorsSimple calls.
	staticAllocator = DescriptorAllocator::Create(
		device.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 64, /*shaderVisible*/false);
}

void PbfApp::InitConstantBuffers() {
	perFrameCb.CreateResources(device.Get());
	computeCb.CreateResources(device.Get());
}

void PbfApp::InitCamera() {
	camera = Egg::Cam::FirstPerson::Create();
	camera->SetView(Float3(0.0f, 10.0f, -40.0f), Float3(0.0f, 0.0f, 1.0f));
	camera->SetSpeed(10.0f);
	camera->SetAspect(aspectRatio);
}

// particleFieldDB[]: double-buffered UAV buffers for each particle attribute.
// positionUploadBuffer / velocityUploadBuffer: CPU-writable staging for initial data.
void PbfApp::InitParticleFields() {
	for (UINT f = 0; f < PF_COUNT; f++) {
		particleFieldDB[f] = DoubleBufferGpuBuffer::Create(
			device.Get(), numParticles, fieldStrides[f],
			(std::wstring(fieldNames[f]) + L" Buffer").c_str(),
			(std::wstring(L"Sorted ") + fieldNames[f] + L" Buffer").c_str(),
			D3D12_RESOURCE_STATE_COMMON,
			*staticAllocator,
			/*needUav*/true, /*needSrv*/false);
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

	// Reserve 3 contiguous main-heap slots for the particle SRV table (t0=pos, t1=den, t2=lod).
	// Snapshot DBs register their front SRV targets here in InitSnapshotBuffers.
	particleSrvTableStart = mainAllocator->Allocate(3);
}

// lodBuffer: uint per particle - LOD countdown.
// lodReductionBuffer: 2 uints [minDTC bits, maxDTC bits].
void PbfApp::InitLodBuffers() {
	lodBuffer = GpuBuffer::Create(
		device.Get(), numParticles, sizeof(UINT),
		L"LOD Buffer", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
	lodBuffer->CreateUav(device.Get(), *staticAllocator);

	lodReductionBuffer = GpuBuffer::Create(
		device.Get(), 2, sizeof(UINT),
		L"LOD Reduction Buffer", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
	lodReductionBuffer->CreateUav(device.Get(), *staticAllocator);
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

// Double-buffered snapshot buffers for position, density, LOD, cellCount, and cellPrefixSum.
// Snapshot DBs register front SRV targets into the main heap so flip() updates them automatically.
void PbfApp::InitSnapshotBuffers() {
	positionSnapshotDB = DoubleBufferGpuBuffer::Create(
		device.Get(), numParticles, sizeof(Float3),
		L"Snapshot Position [front]", L"Snapshot Position [back]",
		D3D12_RESOURCE_STATE_COMMON, *staticAllocator,
		/*needUav*/false, /*needSrv*/true);

	densitySnapshotDB = DoubleBufferGpuBuffer::Create(
		device.Get(), numParticles, sizeof(float),
		L"Snapshot Density [front]", L"Snapshot Density [back]",
		D3D12_RESOURCE_STATE_COMMON, *staticAllocator,
		/*needUav*/false, /*needSrv*/true);

	lodSnapshotDB = DoubleBufferGpuBuffer::Create(
		device.Get(), numParticles, sizeof(UINT),
		L"Snapshot LOD [front]", L"Snapshot LOD [back]",
		D3D12_RESOURCE_STATE_COMMON, *staticAllocator,
		/*needUav*/false, /*needSrv*/true);

	// Register front SRV targets for the particle graphics SRV table (t0=pos, t1=den, t2=lod).
	// flip() keeps these slots pointing at the current front automatically.
	positionSnapshotDB->registerFrontTarget(mainAllocator->GetCpuHandle(particleSrvTableStart),     true);
	densitySnapshotDB ->registerFrontTarget(mainAllocator->GetCpuHandle(particleSrvTableStart + 1), true);
	lodSnapshotDB     ->registerFrontTarget(mainAllocator->GetCpuHandle(particleSrvTableStart + 2), true);

	// Grid snapshot DBs (used by densityVolumeCS / splatDensityVolumeCS in liquid mode).
	cellCountSnapshotDB = DoubleBufferGpuBuffer::Create(
		device.Get(), numCells, sizeof(UINT),
		L"Cell Count Snapshot [front]", L"Cell Count Snapshot [back]",
		D3D12_RESOURCE_STATE_COMMON, *staticAllocator,
		/*needUav*/false, /*needSrv*/true);

	cellPrefixSumSnapshotDB = DoubleBufferGpuBuffer::Create(
		device.Get(), numCells, sizeof(UINT),
		L"Cell Prefix Sum Snapshot [front]", L"Cell Prefix Sum Snapshot [back]",
		D3D12_RESOURCE_STATE_COMMON, *staticAllocator,
		/*needUav*/false, /*needSrv*/true);
}

// Single-buffered density volume: VOL_DIM^3, R32_TYPELESS.
// R32_UINT UAV for CAS float atomic add; R32_FLOAT SRV for liquidPS.
// CPU-only UAV in static heap for ClearUnorderedAccessViewUint.
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

	// CPU-only R32_UINT UAV in static heap: required by ClearUnorderedAccessViewUint.
	UINT clearSlot = staticAllocator->Allocate();
	densityVolume->CreateUavAt(
		device.Get(),
		staticAllocator->GetCpuHandle(clearSlot),
		{}, // GPU handle unused in CPU-only heap
		DXGI_FORMAT_R32_UINT);
	densityVolClearCpuHandle = staticAllocator->GetCpuHandle(clearSlot);

	// 4-slot contiguous liquid table: [density(t0=+0), pos(+1), gridCount(+2), gridPrefix(+3)].
	// density SRV (+0) is static. Snapshot DB front targets handle the other three automatically.
	liquidTableStartSlot = mainAllocator->Allocate(4);
	densityVolume->CreateSrvAt(
		device.Get(),
		mainAllocator->GetCpuHandle(liquidTableStartSlot),
		mainAllocator->GetGpuHandle(liquidTableStartSlot),
		DXGI_FORMAT_R32_FLOAT);

	// Register snapshot front targets: flip() will keep +1/+2/+3 pointing at the current front.
	positionSnapshotDB    ->registerFrontTarget(mainAllocator->GetCpuHandle(liquidTableStartSlot + 1), true);
	cellCountSnapshotDB   ->registerFrontTarget(mainAllocator->GetCpuHandle(liquidTableStartSlot + 2), true);
	cellPrefixSumSnapshotDB->registerFrontTarget(mainAllocator->GetCpuHandle(liquidTableStartSlot + 3), true);
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

	// Allocates one slot in staticAllocator and creates SDF Texture3D SRV.
	solidObstacle->CreateSdfSrv(device.Get(), *staticAllocator);
	sdfCpuHandle = solidObstacle->GetSdfCpuHandle();
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
	particleFieldDB[PF_POSITION]->getFront()->Transition(D3D12_RESOURCE_STATE_COPY_SOURCE, commandList.Get());
	positionSnapshotDB->getFront()->Transition(D3D12_RESOURCE_STATE_COPY_DEST, commandList.Get());
	positionSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_COPY_DEST, commandList.Get());

	const UINT64 posBytes = (UINT64)numParticles * sizeof(Float3);
	commandList->CopyBufferRegion(positionSnapshotDB->getFront()->Get(), 0,
		particleFieldDB[PF_POSITION]->getFront()->Get(), 0, posBytes);
	commandList->CopyBufferRegion(positionSnapshotDB->getBack()->Get(), 0,
		particleFieldDB[PF_POSITION]->getFront()->Get(), 0, posBytes);

	particleFieldDB[PF_POSITION]->getFront()->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, commandList.Get());
	positionSnapshotDB->getFront()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
	positionSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());

	// Transition all remaining snapshot buffers from COMMON to their home NON_PIXEL_SHADER_RESOURCE state.
	densitySnapshotDB->getFront()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
	densitySnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
	lodSnapshotDB->getFront()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
	lodSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
	cellCountSnapshotDB->getFront()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
	cellCountSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
	cellPrefixSumSnapshotDB->getFront()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
	cellPrefixSumSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
}

// Sets all depth pixels in both depth texture slots to 1.0 (far plane)
void PbfApp::RecordDepthTextureClear() {
	particleDepthDB->getFront()->Transition(D3D12_RESOURCE_STATE_DEPTH_WRITE, commandList.Get());
	particleDepthDB->getBack()->Transition(D3D12_RESOURCE_STATE_DEPTH_WRITE, commandList.Get());
	commandList->ClearDepthStencilView(
		particleDepthDB->getFront()->GetDsvCpuHandle(), D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
	commandList->ClearDepthStencilView(
		particleDepthDB->getBack()->GetDsvCpuHandle(), D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
	particleDepthDB->getFront()->Transition(D3D12_RESOURCE_STATE_COMMON, commandList.Get());
	particleDepthDB->getBack()->Transition(D3D12_RESOURCE_STATE_COMMON, commandList.Get());
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
	// Copy to both front and back so that the CPU-side flip() at the start of the first physics
	// frame leaves a valid initial state in whichever buffer becomes the new front.
	particleFieldDB[PF_POSITION]->getFront()->Transition(D3D12_RESOURCE_STATE_COPY_DEST, commandList.Get());
	particleFieldDB[PF_VELOCITY]->getFront()->Transition(D3D12_RESOURCE_STATE_COPY_DEST, commandList.Get());
	particleFieldDB[PF_POSITION]->getBack()->Transition(D3D12_RESOURCE_STATE_COPY_DEST, commandList.Get());
	particleFieldDB[PF_VELOCITY]->getBack()->Transition(D3D12_RESOURCE_STATE_COPY_DEST, commandList.Get());
	commandList->CopyBufferRegion(particleFieldDB[PF_POSITION]->getFront()->Get(), 0,
		positionUploadBuffer->Get(), 0, numParticles * sizeof(Float3));
	commandList->CopyBufferRegion(particleFieldDB[PF_VELOCITY]->getFront()->Get(), 0,
		velocityUploadBuffer->Get(), 0, numParticles * sizeof(Float3));
	commandList->CopyBufferRegion(particleFieldDB[PF_POSITION]->getBack()->Get(), 0,
		positionUploadBuffer->Get(), 0, numParticles * sizeof(Float3));
	commandList->CopyBufferRegion(particleFieldDB[PF_VELOCITY]->getBack()->Get(), 0,
		velocityUploadBuffer->Get(), 0, numParticles * sizeof(Float3));
	particleFieldDB[PF_POSITION]->getFront()->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, commandList.Get());
	particleFieldDB[PF_VELOCITY]->getFront()->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, commandList.Get());
	particleFieldDB[PF_POSITION]->getBack()->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, commandList.Get());
	particleFieldDB[PF_VELOCITY]->getBack()->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, commandList.Get());
}

// Create all compute shader PSOs and wire each shader to its contiguous descriptor region.
// Each shader's region is allocated from mainAllocator; front/back DB targets are registered
// so flip() keeps the descriptors current without any per-frame CopyDescriptorsSimple in hot paths.
void PbfApp::BuildComputePipelines() {
	D3D12_GPU_VIRTUAL_ADDRESS cbv = computeCb.GetGPUVirtualAddress();
	using P = com_ptr<ID3D12Resource>*;

	auto copy1 = [&](UINT slot, D3D12_CPU_DESCRIPTOR_HANDLE src) {
		device->CopyDescriptorsSimple(1, mainAllocator->GetCpuHandle(slot), src,
			D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	};
	// SpatialGrid owns cellCount/cellPrefixSum/perm/groupSum buffers and all grid+sort shaders.
	// Must be created first: solver shaders below reference the grid buffers via GetCellCountBuffer()/GetPrefixSumBuffer().
	spatialGrid = SpatialGrid::Create(device.Get(), numCells, numParticles,
		*mainAllocator, *staticAllocator, cbv, particleFieldDB);

	// ---------- predictCS: UAV(u0-2) SRV(t0) = 4 slots ----------
	// [0]=position, [1]=velocity, [2]=predictedPosition, [3]=SDF SRV
	{
		UINT s = mainAllocator->Allocate(4);
		particleFieldDB[PF_POSITION]->registerFrontTarget(mainAllocator->GetCpuHandle(s),     false);
		particleFieldDB[PF_VELOCITY]->registerFrontTarget(mainAllocator->GetCpuHandle(s + 1), false);
		particleFieldDB[PF_PREDICTED_POSITION]->registerFrontTarget(mainAllocator->GetCpuHandle(s + 2), false);
		copy1(s + 3, sdfCpuHandle);
		predictShader = ComputeShader::Create(device.Get(), "Shaders/predictCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_POSITION]->getFront()->GetResourcePtr(),
			                particleFieldDB[PF_VELOCITY]->getFront()->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_VELOCITY]->getFront()->GetResourcePtr(),
			                particleFieldDB[PF_PREDICTED_POSITION]->getFront()->GetResourcePtr() });
	}

	// ---------- collisionPredictedPositionCS: UAV(u0-1) SRV(t0) = 3 slots ----------
	// [0]=predictedPosition back, [1]=lod, [2]=SDF SRV
	{
		UINT s = mainAllocator->Allocate(3);
		particleFieldDB[PF_PREDICTED_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s),     false);
		copy1(s + 1, lodBuffer->GetUavCpuHandle());
		copy1(s + 2, sdfCpuHandle);
		collisionPredictedPositionShader = ComputeShader::Create(device.Get(), "Shaders/collisionPredictedPositionCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr(),
			                lodBuffer->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr() });
	}

	// ---------- positionFromScratchCS: UAV(u0-2) = 3 slots ----------
	// [0]=predictedPosition back, [1]=scratch back, [2]=lod
	{
		UINT s = mainAllocator->Allocate(3);
		particleFieldDB[PF_PREDICTED_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s),     false);
		particleFieldDB[PF_SCRATCH]->registerBackTarget(mainAllocator->GetCpuHandle(s + 1),            false);
		copy1(s + 2, lodBuffer->GetUavCpuHandle());
		positionFromScratchShader = ComputeShader::Create(device.Get(), "Shaders/positionFromScratchCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_SCRATCH]->getBack()->GetResourcePtr(),
			                lodBuffer->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr(),
			                lodBuffer->GetResourcePtr() });
	}

	// ---------- updateVelocityCS: UAV(u0-2) = 3 slots ----------
	// [0]=position back, [1]=velocity back, [2]=predictedPosition back
	{
		UINT s = mainAllocator->Allocate(3);
		particleFieldDB[PF_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s),          false);
		particleFieldDB[PF_VELOCITY]->registerBackTarget(mainAllocator->GetCpuHandle(s + 1),      false);
		particleFieldDB[PF_PREDICTED_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s + 2), false);
		updateVelocityShader = ComputeShader::Create(device.Get(), "Shaders/updateVelocityCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_POSITION]->getBack()->GetResourcePtr(),
			                particleFieldDB[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_VELOCITY]->getBack()->GetResourcePtr() });
	}

	// ---------- velocityFromScratchCS: UAV(u0-1) = 2 slots ----------
	// [0]=velocity back, [1]=scratch back
	{
		UINT s = mainAllocator->Allocate(2);
		particleFieldDB[PF_VELOCITY]->registerBackTarget(mainAllocator->GetCpuHandle(s),     false);
		particleFieldDB[PF_SCRATCH]->registerBackTarget(mainAllocator->GetCpuHandle(s + 1),  false);
		velocityFromScratchShader = ComputeShader::Create(device.Get(), "Shaders/velocityFromScratchCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_SCRATCH]->getBack()->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_VELOCITY]->getBack()->GetResourcePtr() });
	}

	// ---------- updatePositionCS: UAV(u0-1) = 2 slots ----------
	// [0]=position back, [1]=predictedPosition back
	{
		UINT s = mainAllocator->Allocate(2);
		particleFieldDB[PF_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s),          false);
		particleFieldDB[PF_PREDICTED_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s + 1), false);
		updatePositionShader = ComputeShader::Create(device.Get(), "Shaders/updatePositionCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_POSITION]->getBack()->GetResourcePtr() });
	}

	// ---------- lambdaCS: UAV(u0-5) = 6 slots ----------
	// [0]=predictedPosition, [1]=lambda, [2]=density, [3]=cellCount, [4]=cellPrefixSum, [5]=lod
	{
		UINT s = mainAllocator->Allocate(6);
		particleFieldDB[PF_PREDICTED_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s),     false);
		particleFieldDB[PF_LAMBDA]->registerBackTarget(mainAllocator->GetCpuHandle(s + 1),             false);
		particleFieldDB[PF_DENSITY]->registerBackTarget(mainAllocator->GetCpuHandle(s + 2),            false);
		copy1(s + 3, spatialGrid->GetCellCountBuffer()->GetUavCpuHandle());
		copy1(s + 4, spatialGrid->GetPrefixSumBuffer()->GetUavCpuHandle());
		copy1(s + 5, lodBuffer->GetUavCpuHandle());
		lambdaShader = ComputeShader::Create(device.Get(), "Shaders/lambdaCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr(),
			                spatialGrid->GetCellCountBuffer()->GetResourcePtr(),
			                spatialGrid->GetPrefixSumBuffer()->GetResourcePtr(),
			                lodBuffer->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_LAMBDA]->getBack()->GetResourcePtr(),
			                particleFieldDB[PF_DENSITY]->getBack()->GetResourcePtr() });
	}

	// ---------- deltaCS: UAV(u0-5) = 6 slots ----------
	// [0]=predictedPosition, [1]=lambda, [2]=scratch, [3]=cellCount, [4]=cellPrefixSum, [5]=lod
	{
		UINT s = mainAllocator->Allocate(6);
		particleFieldDB[PF_PREDICTED_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s),     false);
		particleFieldDB[PF_LAMBDA]->registerBackTarget(mainAllocator->GetCpuHandle(s + 1),             false);
		particleFieldDB[PF_SCRATCH]->registerBackTarget(mainAllocator->GetCpuHandle(s + 2),            false);
		copy1(s + 3, spatialGrid->GetCellCountBuffer()->GetUavCpuHandle());
		copy1(s + 4, spatialGrid->GetPrefixSumBuffer()->GetUavCpuHandle());
		copy1(s + 5, lodBuffer->GetUavCpuHandle());
		deltaShader = ComputeShader::Create(device.Get(), "Shaders/deltaCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr(),
			                particleFieldDB[PF_LAMBDA]->getBack()->GetResourcePtr(),
			                spatialGrid->GetCellCountBuffer()->GetResourcePtr(),
			                spatialGrid->GetPrefixSumBuffer()->GetResourcePtr(),
			                lodBuffer->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_SCRATCH]->getBack()->GetResourcePtr() });
	}

	// ---------- vorticityCS: UAV(u0-4) = 5 slots ----------
	// [0]=position, [1]=velocity, [2]=omega, [3]=cellCount, [4]=cellPrefixSum
	{
		UINT s = mainAllocator->Allocate(5);
		particleFieldDB[PF_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s),     false);
		particleFieldDB[PF_VELOCITY]->registerBackTarget(mainAllocator->GetCpuHandle(s + 1), false);
		particleFieldDB[PF_OMEGA]->registerBackTarget(mainAllocator->GetCpuHandle(s + 2),    false);
		copy1(s + 3, spatialGrid->GetCellCountBuffer()->GetUavCpuHandle());
		copy1(s + 4, spatialGrid->GetPrefixSumBuffer()->GetUavCpuHandle());
		vorticityShader = ComputeShader::Create(device.Get(), "Shaders/vorticityCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_POSITION]->getBack()->GetResourcePtr(),
			                particleFieldDB[PF_VELOCITY]->getBack()->GetResourcePtr(),
			                spatialGrid->GetCellCountBuffer()->GetResourcePtr(),
			                spatialGrid->GetPrefixSumBuffer()->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_OMEGA]->getBack()->GetResourcePtr() });
	}

	// ---------- confinementViscosityCS: UAV(u0-5) = 6 slots ----------
	// [0]=position, [1]=velocity, [2]=omega, [3]=scratch, [4]=cellCount, [5]=cellPrefixSum
	{
		UINT s = mainAllocator->Allocate(6);
		particleFieldDB[PF_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s),     false);
		particleFieldDB[PF_VELOCITY]->registerBackTarget(mainAllocator->GetCpuHandle(s + 1), false);
		particleFieldDB[PF_OMEGA]->registerBackTarget(mainAllocator->GetCpuHandle(s + 2),    false);
		particleFieldDB[PF_SCRATCH]->registerBackTarget(mainAllocator->GetCpuHandle(s + 3),  false);
		copy1(s + 4, spatialGrid->GetCellCountBuffer()->GetUavCpuHandle());
		copy1(s + 5, spatialGrid->GetPrefixSumBuffer()->GetUavCpuHandle());
		confinementViscosityShader = ComputeShader::Create(device.Get(), "Shaders/confinementViscosityCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_POSITION]->getBack()->GetResourcePtr(),
			                particleFieldDB[PF_VELOCITY]->getBack()->GetResourcePtr(),
			                particleFieldDB[PF_OMEGA]->getBack()->GetResourcePtr(),
			                spatialGrid->GetCellCountBuffer()->GetResourcePtr(),
			                spatialGrid->GetPrefixSumBuffer()->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_SCRATCH]->getBack()->GetResourcePtr() });
	}

	// ---------- clearDtcReductionCS: UAV(u0) = 1 slot ----------
	{
		UINT s = mainAllocator->Allocate(1);
		copy1(s, lodReductionBuffer->GetUavCpuHandle());
		clearDtcReductionShader = ComputeShader::Create(device.Get(), "Shaders/clearDtcReductionCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{},
			std::vector<P>{ lodReductionBuffer->GetResourcePtr() });
	}

	// ---------- dtcReductionCS: UAV(u0-1) = 2 slots ----------
	// [0]=predictedPosition back, [1]=lodReduction
	{
		UINT s = mainAllocator->Allocate(2);
		particleFieldDB[PF_PREDICTED_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s), false);
		copy1(s + 1, lodReductionBuffer->GetUavCpuHandle());
		lodReductionShader = ComputeShader::Create(device.Get(), "Shaders/dtcReductionCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr(),
			                lodReductionBuffer->GetResourcePtr() },
			std::vector<P>{ lodReductionBuffer->GetResourcePtr() });
	}

	// ---------- dtcLodCS: UAV(u0-2) = 3 slots ----------
	// [0]=predictedPosition back, [1]=lod, [2]=lodReduction
	{
		UINT s = mainAllocator->Allocate(3);
		particleFieldDB[PF_PREDICTED_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s), false);
		copy1(s + 1, lodBuffer->GetUavCpuHandle());
		copy1(s + 2, lodReductionBuffer->GetUavCpuHandle());
		lodShader = ComputeShader::Create(device.Get(), "Shaders/dtcLodCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr(),
			                lodReductionBuffer->GetResourcePtr() },
			std::vector<P>{ lodBuffer->GetResourcePtr() });
	}

	// ---------- setLodMaxCS: UAV(u0) = 1 slot ----------
	{
		UINT s = mainAllocator->Allocate(1);
		copy1(s, lodBuffer->GetUavCpuHandle());
		setLodMaxShader = ComputeShader::Create(device.Get(), "Shaders/setLodMaxCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{},
			std::vector<P>{ lodBuffer->GetResourcePtr() });
	}

	// ---------- clearDtvsReductionCS: UAV(u0) = 1 slot ----------
	{
		UINT s = mainAllocator->Allocate(1);
		copy1(s, lodReductionBuffer->GetUavCpuHandle());
		clearDtvsReductionShader = ComputeShader::Create(device.Get(), "Shaders/clearDtvsReductionCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{},
			std::vector<P>{ lodReductionBuffer->GetResourcePtr() });
	}

	// ---------- dtvsReductionCS: UAV(u0-1) SRV(t0) = 3 slots ----------
	// [0]=predictedPosition back, [1]=lodReduction, [2]=depth SRV (front depth texture)
	{
		UINT s = mainAllocator->Allocate(3);
		particleFieldDB[PF_PREDICTED_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s), false);
		copy1(s + 1, lodReductionBuffer->GetUavCpuHandle());
		particleDepthDB->registerFrontTarget(mainAllocator->GetCpuHandle(s + 2), /*isSrv*/true);
		dtvsReductionShader = ComputeShader::Create(device.Get(), "Shaders/dtvsReductionCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr(),
			                lodReductionBuffer->GetResourcePtr() },
			std::vector<P>{ lodReductionBuffer->GetResourcePtr() });
	}

	// ---------- dtvsLodCS: UAV(u0-2) SRV(t0) = 4 slots ----------
	// [0]=predictedPosition back, [1]=lod, [2]=lodReduction, [3]=depth SRV
	{
		UINT s = mainAllocator->Allocate(4);
		particleFieldDB[PF_PREDICTED_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s), false);
		copy1(s + 1, lodBuffer->GetUavCpuHandle());
		copy1(s + 2, lodReductionBuffer->GetUavCpuHandle());
		particleDepthDB->registerFrontTarget(mainAllocator->GetCpuHandle(s + 3), /*isSrv*/true);
		dtvsLodShader = ComputeShader::Create(device.Get(), "Shaders/dtvsLodCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr(),
			                lodReductionBuffer->GetResourcePtr() },
			std::vector<P>{ lodBuffer->GetResourcePtr() });
	}

	// ---------- splatDensityVolumeCS: SRV(t0) UAV(u0) = 2 slots ----------
	// [0]=posSnapshot front SRV, [1]=densityVol UAV
	{
		UINT s = mainAllocator->Allocate(2);
		positionSnapshotDB->registerFrontTarget(mainAllocator->GetCpuHandle(s), true);
		copy1(s + 1, densityVolume->GetUavCpuHandle());
		splatDensityShader = ComputeShader::Create(device.Get(), "Shaders/splatDensityVolumeCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{},
			std::vector<P>{ densityVolume->GetResourcePtr() });
	}

	// ---------- GSM variants ----------

	// GSM_lambdaCS: UAV(u0-5) = 6 slots
	{
		UINT s = mainAllocator->Allocate(6);
		particleFieldDB[PF_PREDICTED_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s),     false);
		particleFieldDB[PF_LAMBDA]->registerBackTarget(mainAllocator->GetCpuHandle(s + 1),             false);
		particleFieldDB[PF_DENSITY]->registerBackTarget(mainAllocator->GetCpuHandle(s + 2),            false);
		copy1(s + 3, spatialGrid->GetCellCountBuffer()->GetUavCpuHandle());
		copy1(s + 4, spatialGrid->GetPrefixSumBuffer()->GetUavCpuHandle());
		copy1(s + 5, lodBuffer->GetUavCpuHandle());
		gsmLambdaShader = ComputeShader::Create(device.Get(), "Shaders/GSM_lambdaCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr(),
			                spatialGrid->GetCellCountBuffer()->GetResourcePtr(),
			                spatialGrid->GetPrefixSumBuffer()->GetResourcePtr(),
			                lodBuffer->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_LAMBDA]->getBack()->GetResourcePtr(),
			                particleFieldDB[PF_DENSITY]->getBack()->GetResourcePtr() });
	}

	// GSM_deltaCS: UAV(u0-5) = 6 slots
	{
		UINT s = mainAllocator->Allocate(6);
		particleFieldDB[PF_PREDICTED_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s),     false);
		particleFieldDB[PF_LAMBDA]->registerBackTarget(mainAllocator->GetCpuHandle(s + 1),             false);
		particleFieldDB[PF_SCRATCH]->registerBackTarget(mainAllocator->GetCpuHandle(s + 2),            false);
		copy1(s + 3, spatialGrid->GetCellCountBuffer()->GetUavCpuHandle());
		copy1(s + 4, spatialGrid->GetPrefixSumBuffer()->GetUavCpuHandle());
		copy1(s + 5, lodBuffer->GetUavCpuHandle());
		gsmDeltaShader = ComputeShader::Create(device.Get(), "Shaders/GSM_deltaCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr(),
			                particleFieldDB[PF_LAMBDA]->getBack()->GetResourcePtr(),
			                spatialGrid->GetCellCountBuffer()->GetResourcePtr(),
			                spatialGrid->GetPrefixSumBuffer()->GetResourcePtr(),
			                lodBuffer->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_SCRATCH]->getBack()->GetResourcePtr() });
	}

	// GSM_vorticity: UAV(u0-4) = 5 slots
	{
		UINT s = mainAllocator->Allocate(5);
		particleFieldDB[PF_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s),     false);
		particleFieldDB[PF_VELOCITY]->registerBackTarget(mainAllocator->GetCpuHandle(s + 1), false);
		particleFieldDB[PF_OMEGA]->registerBackTarget(mainAllocator->GetCpuHandle(s + 2),    false);
		copy1(s + 3, spatialGrid->GetCellCountBuffer()->GetUavCpuHandle());
		copy1(s + 4, spatialGrid->GetPrefixSumBuffer()->GetUavCpuHandle());
		gsmVorticityShader = ComputeShader::Create(device.Get(), "Shaders/GSM_vorticity.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_POSITION]->getBack()->GetResourcePtr(),
			                particleFieldDB[PF_VELOCITY]->getBack()->GetResourcePtr(),
			                spatialGrid->GetCellCountBuffer()->GetResourcePtr(),
			                spatialGrid->GetPrefixSumBuffer()->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_OMEGA]->getBack()->GetResourcePtr() });
	}

	// GSM_confinementViscosityCS: UAV(u0-5) = 6 slots
	{
		UINT s = mainAllocator->Allocate(6);
		particleFieldDB[PF_POSITION]->registerBackTarget(mainAllocator->GetCpuHandle(s),     false);
		particleFieldDB[PF_VELOCITY]->registerBackTarget(mainAllocator->GetCpuHandle(s + 1), false);
		particleFieldDB[PF_OMEGA]->registerBackTarget(mainAllocator->GetCpuHandle(s + 2),    false);
		particleFieldDB[PF_SCRATCH]->registerBackTarget(mainAllocator->GetCpuHandle(s + 3),  false);
		copy1(s + 4, spatialGrid->GetCellCountBuffer()->GetUavCpuHandle());
		copy1(s + 5, spatialGrid->GetPrefixSumBuffer()->GetUavCpuHandle());
		gsmConfinementViscosityShader = ComputeShader::Create(device.Get(), "Shaders/GSM_confinementViscosityCS.cso", cbv,
			mainAllocator->GetGpuHandle(s),
			std::vector<P>{ particleFieldDB[PF_POSITION]->getBack()->GetResourcePtr(),
			                particleFieldDB[PF_VELOCITY]->getBack()->GetResourcePtr(),
			                particleFieldDB[PF_OMEGA]->getBack()->GetResourcePtr(),
			                spatialGrid->GetCellCountBuffer()->GetResourcePtr(),
			                spatialGrid->GetPrefixSumBuffer()->GetResourcePtr() },
			std::vector<P>{ particleFieldDB[PF_SCRATCH]->getBack()->GetResourcePtr() });
	}
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
		// Wait for compute frame N-1 before reusing the allocator and readback buffers.
		cpuWaitForCompute(frameCount - 1);

		// Both compute AND graphics frame N-1 are done at this point (graphics waited at end
		// of last Render()). Safe to flip all double buffers:
		// - particleFieldDBs: back (sorted by permutateCS in N-1) becomes the new front for compute N
		// - snapshot DBs: back (written by compute N-1) becomes the new front for graphics N
		// - particleDepthDB: back (written by graphics N-1) becomes the new front for compute N
		for (UINT f = 0; f < PF_COUNT; f++) particleFieldDB[f]->flip();
		positionSnapshotDB    ->flip();
		densitySnapshotDB     ->flip();
		lodSnapshotDB         ->flip();
		cellCountSnapshotDB   ->flip();
		cellPrefixSumSnapshotDB->flip();
		particleDepthDB       ->flip();

		CalculateAvgDensity();
		CalculateAvgLod();

		UpdateComputeCb(lastDt);

		PrepareComputeCommandList();
		RecordComputeCommands();
		ExecuteCompute();
	}
	computeFence.signal(computeCommandQueue, frameCount);

	// GPU-stall graphics until compute frame N-1's snapshot writes are complete.
	graphicsWaitForCompute(frameCount - 1);

	PrepareCommandList();
	RecordGraphicsCommands();
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

void PbfApp::RecordComputeCommands() {
	UINT numGroups = (numParticles + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;

	predictShader->dispatch_then_barrier(computeList.Get(), numGroups);

	spatialGrid->Build(computeList.Get());

	// collisionPredictedPositionCS moved to after sort: grid posToCell() clamps OOB positions to
	// boundary cells, so sorting against unclamped predictions is safe; collision corrects afterward.
	collisionPredictedPositionShader->dispatch_then_barrier(computeList.Get(), numGroups);

	CalculateLod();

	for (int iter = 0; iter < solverIterations; iter++) {
		(gsmEnabled ? gsmLambdaShader : lambdaShader)->dispatch_then_barrier(computeList.Get(), numGroups);
		(gsmEnabled ? gsmDeltaShader : deltaShader)->dispatch_then_barrier(computeList.Get(), numGroups);
		positionFromScratchShader->dispatch_then_barrier(computeList.Get(), numGroups);
		collisionPredictedPositionShader->dispatch_then_barrier(computeList.Get(), numGroups);
	}

	updateVelocityShader->dispatch_then_barrier(computeList.Get(), numGroups);
	(gsmEnabled ? gsmVorticityShader : vorticityShader)->dispatch_then_barrier(computeList.Get(), numGroups);
	(gsmEnabled ? gsmConfinementViscosityShader : confinementViscosityShader)->dispatch_then_barrier(computeList.Get(), numGroups);
	velocityFromScratchShader->dispatch_then_barrier(computeList.Get(), numGroups);
	updatePositionShader->dispatch_then_barrier(computeList.Get(), numGroups);

	WriteSnapshot();
}

void PbfApp::WriteSnapshot() {
	// Copy position and density into the back snapshot buffers.
	// After this frame's flip(), back becomes the new front for graphics to read.
	// Sorted particle data lives in back (permutateCS wrote there; flip happens CPU-side in Render()).
	particleFieldDB[PF_POSITION]->getBack()->Transition(D3D12_RESOURCE_STATE_COPY_SOURCE, computeList.Get());
	particleFieldDB[PF_DENSITY]->getBack()->Transition(D3D12_RESOURCE_STATE_COPY_SOURCE, computeList.Get());
	positionSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_COPY_DEST, computeList.Get());
	densitySnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_COPY_DEST, computeList.Get());

	computeList->CopyBufferRegion(positionSnapshotDB->getBack()->Get(), 0,
		particleFieldDB[PF_POSITION]->getBack()->Get(), 0, (UINT64)numParticles * sizeof(Float3));
	computeList->CopyBufferRegion(densitySnapshotDB->getBack()->Get(), 0,
		particleFieldDB[PF_DENSITY]->getBack()->Get(), 0, (UINT64)numParticles * sizeof(float));

	computeList->CopyBufferRegion(densityReadbackBuffer->Get(), 0,
		particleFieldDB[PF_DENSITY]->getBack()->Get(), 0, (UINT64)numParticles * sizeof(float));

	particleFieldDB[PF_POSITION]->getBack()->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, computeList.Get());
	particleFieldDB[PF_DENSITY]->getBack()->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, computeList.Get());
	positionSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, computeList.Get());
	densitySnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, computeList.Get());

	// Copy grid buffers into the back grid snapshot slots.
	spatialGrid->GetCellCountBuffer()->Transition(D3D12_RESOURCE_STATE_COPY_SOURCE, computeList.Get());
	spatialGrid->GetPrefixSumBuffer()->Transition(D3D12_RESOURCE_STATE_COPY_SOURCE, computeList.Get());
	cellCountSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_COPY_DEST, computeList.Get());
	cellPrefixSumSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_COPY_DEST, computeList.Get());

	const UINT64 gridBufSize = (UINT64)numCells * sizeof(UINT);
	computeList->CopyBufferRegion(cellCountSnapshotDB->getBack()->Get(), 0,
		spatialGrid->GetCellCountBuffer()->Get(), 0, gridBufSize);
	computeList->CopyBufferRegion(cellPrefixSumSnapshotDB->getBack()->Get(), 0,
		spatialGrid->GetPrefixSumBuffer()->Get(), 0, gridBufSize);

	spatialGrid->GetCellCountBuffer()->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, computeList.Get());
	spatialGrid->GetPrefixSumBuffer()->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, computeList.Get());
	cellCountSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, computeList.Get());
	cellPrefixSumSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, computeList.Get());
}

void PbfApp::CalculateLod() {
	UINT numGroups = (numParticles + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;

	if (lodMode == LodMode::DTC) {
		clearDtcReductionShader->dispatch_then_barrier(computeList.Get(), 1);
		lodReductionShader->dispatch_then_barrier(computeList.Get(), numGroups);
		lodShader->dispatch_then_barrier(computeList.Get(), numGroups);
	}
	else if (lodMode == LodMode::DTVS) {
		// particleDepthDB->getFront() = last frame's depth written by graphics.
		// Descriptor slot is already correct (registered as front target in BuildComputePipelines).
		particleDepthDB->getFront()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, computeList.Get());

		clearDtvsReductionShader->dispatch_then_barrier(computeList.Get(), 1);
		dtvsReductionShader->dispatch_then_barrier(computeList.Get(), numGroups);
		dtvsLodShader->dispatch_then_barrier(computeList.Get(), numGroups);

		particleDepthDB->getFront()->Transition(D3D12_RESOURCE_STATE_COMMON, computeList.Get());
	}
	else {
		setLodMaxShader->dispatch_then_barrier(computeList.Get(), numGroups);
	}

	// Snapshot LOD before solver loop decrements it. Write into back LOD snapshot.
	lodBuffer->Transition(D3D12_RESOURCE_STATE_COPY_SOURCE, computeList.Get());
	lodSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_COPY_DEST, computeList.Get());
	computeList->CopyBufferRegion(lodSnapshotDB->getBack()->Get(), 0,
		lodBuffer->Get(), 0, (UINT64)numParticles * sizeof(UINT));
	computeList->CopyBufferRegion(lodReadbackBuffer->Get(), 0,
		lodBuffer->Get(), 0, (UINT64)numParticles * sizeof(UINT));
	lodBuffer->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, computeList.Get());
	lodSnapshotDB->getBack()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, computeList.Get());
}

void PbfApp::RecordGraphicsCommands() {
	backgroundMesh->Draw(commandList.Get());
	solidObstacle->Draw(commandList.Get());

	// Snapshot DB flip() keeps the particleSrvTableStart slots current — no CopyDescriptorsSimple here.

	// Promote positionSnapshot front to pixel-visible before any draw that uses it from the pixel stage.
	constexpr D3D12_RESOURCE_STATES SRV_ALL =
		D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
	if (shadingMode == ShadingMode::LIQUID)
		positionSnapshotDB->getFront()->Transition(SRV_ALL, commandList.Get());

	if (lodMode == LodMode::DTVS) DrawParticleDepth();

	if (shadingMode == ShadingMode::LIQUID) {
		DrawLiquidSurface();
		positionSnapshotDB->getFront()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
	}
	else {
		particleMesh->Draw(commandList.Get());
	}
}

// Fill the density volume and draw the ray-marched liquid surface, all on the graphics command list.
// densityVolumeCS is dispatched here (graphics queue) reading from the previous frame's position and
// grid snapshots at [readIdx]. liquidPS then ray-marches through the freshly filled volume in the same frame.
// The GPU-wait at the front of the graphics list (graphicsWaitForCompute(N-1)) guarantees that
// snapshotPosition[readIdx] and the grid snapshots are fully written by compute before this runs.
void PbfApp::DrawLiquidSurface() {
	// liquidTableStartSlot +1/+2/+3 are updated by snapshot DB flip() — no CopyDescriptorsSimple needed.

	densityVolume->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, commandList.Get());

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
	densityVolume->Transition(SRV_ALL, commandList.Get());
	cellCountSnapshotDB->getFront()->Transition(SRV_ALL, commandList.Get());
	cellPrefixSumSnapshotDB->getFront()->Transition(SRV_ALL, commandList.Get());

	liquidMesh->Draw(commandList.Get());

	cellCountSnapshotDB->getFront()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
	cellPrefixSumSnapshotDB->getFront()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, commandList.Get());
	densityVolume->Transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, commandList.Get());

	const UINT clearVal[4] = { 0u, 0u, 0u, 0u };
	commandList->ClearUnorderedAccessViewUint(
		densityVolumeHandle, densityVolClearCpuHandle,
		densityVolume->Get(), clearVal, 0, nullptr);

	densityVolume->Transition(D3D12_RESOURCE_STATE_COMMON, commandList.Get());
}

// Record the DTVS depth-only particle draw into the already-open graphics command list.
// Writes into particleDepthDB->getBack() (graphics writes back, compute reads front next frame).
// Leaves the texture in COMMON state so compute can read it via the front SRV next frame.
void PbfApp::DrawParticleDepth() {
	particleDepthDB->getBack()->Transition(D3D12_RESOURCE_STATE_DEPTH_WRITE, commandList.Get());

	D3D12_CPU_DESCRIPTOR_HANDLE dsv = particleDepthDB->getBack()->GetDsvCpuHandle();

	commandList->OMSetRenderTargets(0, nullptr, FALSE, &dsv);
	commandList->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

	commandList->SetGraphicsRootSignature(depthOnlyRootSig.Get());
	commandList->SetPipelineState(depthOnlyPso.Get());
	commandList->SetGraphicsRootConstantBufferView(0, perFrameCb.GetGPUVirtualAddress());
	commandList->SetGraphicsRootDescriptorTable(1, mainAllocator->GetGpuHandle(particleSrvTableStart));

	commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
	commandList->DrawInstanced(numParticles, 1, 0, 0);

	CD3DX12_CPU_DESCRIPTOR_HANDLE mainDsv(dsvHeap->GetCPUDescriptorHandleForHeapStart());
	CD3DX12_CPU_DESCRIPTOR_HANDLE rtv(
		rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart(),
		swapChainBackBufferIndex,
		rtvDescriptorHandleIncrementSize);
	commandList->OMSetRenderTargets(1, &rtv, FALSE, &mainDsv);

	particleDepthDB->getBack()->Transition(D3D12_RESOURCE_STATE_COMMON, commandList.Get());
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
	computeCb->sdfMin = smin;
	computeCb->sdfMax = smax;
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

void PbfApp::ReleaseSwapChainResources()  {
	// Do NOT null out particleDepthDB or particleDsvAllocator: resize2D reuses their allocations.
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
	InitLodBuffers();
	InitReadbackBuffers();
	InitSnapshotBuffers();
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
