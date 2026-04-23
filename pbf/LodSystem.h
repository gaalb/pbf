#pragma once
#include "Egg/Common.h"
#include "Egg/Shader.h"
#include "Egg/d3dx12.h"
#include "ComputeShader.h"
#include "GpuBuffer.h"
#include "DoubleBufferGpuBuffer.h"
#include "DoubleBufferGpuTexture.h"
#include "DescriptorAllocator.h"
#include "ParticleTypes.h"
#include "SharedConfig.hlsli"

// Encapsulates the LOD (level-of-detail) subsystem: assigns a per-particle solver-iteration
// count each frame based on camera distance (DTC) or distance to visible surface (DTVS).                           
// Lifecycle:
//   Construct          — from CreateResources, after descriptor heaps exist. Creates all GPU
//                        buffers, depth textures, and DSV/SRV descriptors.
//   BuildPipelines()   — from BuildComputePipelines, after the particle field DBs are ready.
//                        Wires descriptor table slots and creates all PSOs.
//   Resize()           — from CreateSwapChainResources on window resize. Recreates depth textures.
//   RecordDepthClear() — once during initialization (UploadAll) to pre-fill both depth slots.
//   CalculateLod()     — each compute frame; dispatches LOD passes based on current mode.
//   DrawParticleDepth() — each graphics frame when mode == DTVS; records the depth-only draw.
//
// PbfApp retains lodSnapshotDB and lodReadbackBuffer (snapshot/readback concerns).
// After CalculateLod(), PbfApp copies lodBuffer -> lodSnapshotDB via GetLodBuffer(),
// before the solver loop that decrements the values.
// PbfApp retains minLOD; it flows through the shared compute constant buffer.
GG_CLASS(LodSystem)
    UINT numParticles_ = 0;
    // per-particle LOD countdown (UINT); written before the solver loop, decremented by positionFromScratchCS each iteration.
    GpuBuffer::P lodBuffer; 
    GpuBuffer::P lodReductionBuffer; // 2-element scratch for DTC/DTVS GPU reductions [min, max].
    // double-buffered window-resolution R32_TYPELESS depth texture: graphics writes D32_FLOAT via DSV; compute reads R32_FLOAT via SRV.
    DoubleBufferGpuTexture::P particleDepthDB; // window-resolution R32_TYPELESS depth DB
    DescriptorAllocator::P  particleDsvAllocator; //2-slot non-shader-visible DSV heap for particleDepthDB.

    // DTC path: assign LOD from camera distance to particle
    ComputeShader::P clearDtcReductionShader;  // zero lodReductionBuffer before DTC reduction
    ComputeShader::P lodReductionShader;       // compute per-frame DTC min/max via GPU atomics
    ComputeShader::P lodShader;                // assign per-particle LOD from DTC range

    // Non-adaptive path
    ComputeShader::P setLodMaxShader;          // fill all lod[i] = maxLOD

    // DTVS path: assign LOD from distance to visible surface
    ComputeShader::P clearDtvsReductionShader; // reset lodReductionBuffer[0] before DTVS reduction
    ComputeShader::P dtvsReductionShader;      // accumulate max DTVS into lodReductionBuffer[0]
    ComputeShader::P dtvsLodShader;            // assign per-particle LOD from DTVS / maxDTVS

    // Depth-only graphics pipeline (DTVS)
    com_ptr<ID3D12RootSignature> depthOnlyRootSig;
    com_ptr<ID3D12PipelineState> depthOnlyPso;

public:
    enum class Mode { NONE = 0, DTC = 1, DTVS = 2 };
    Mode mode = Mode::DTVS;

    // Create all GPU resources: lodBuffer, lodReductionBuffer, particleDsvAllocator,
    // and particleDepthDB (with DSV and SRV descriptors written into staticAlloc).
    // Must be called after both mainAlloc and staticAlloc exist.
    LodSystem(
        ID3D12Device* device,
        UINT numParticles,
        UINT width, UINT height,
        DescriptorAllocator& mainAlloc,
        DescriptorAllocator& staticAlloc)
        : numParticles_(numParticles)
    {
        // per-particle LOD countdown (uint, one entry per particle)
        lodBuffer = GpuBuffer::Create(
            device, numParticles, sizeof(UINT),
            L"LOD Buffer", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
        lodBuffer->CreateUav(device, staticAlloc);

        // two-element scratch buffer: [minVal_bits, maxVal_bits] for GPU reductions
        lodReductionBuffer = GpuBuffer::Create(
            device, 2, sizeof(UINT),
            L"LOD Reduction Buffer", D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
        lodReductionBuffer->CreateUav(device, staticAlloc);

        // 2-slot non-shader-visible DSV heap for the two depth texture slots
        particleDsvAllocator = DescriptorAllocator::Create(
            device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 2, /*shaderVisible=*/false);

        // Double-buffered window-resolution depth texture.
        D3D12_CLEAR_VALUE clearValue = {};
        clearValue.Format = DXGI_FORMAT_D32_FLOAT;
        clearValue.DepthStencil.Depth = 1.0f;

        // R32_TYPELESS: D32_FLOAT DSV (graphics depth write) + R32_FLOAT SRV (compute read).
        // D3D12 forbids creating both a DSV and an SRV on the same resource (even though that's
        // exactly what we need if we want to read the data written to the DSV) if that resource was
        // created with a typed format. So the resource foramt is R32_TYPELESS, i.e. no inherent type
        // just 32 bits per texel. Then, the graphics pipeline interprets those 32 bits as normalized
        // depth float when writing (D32_FLOAT). The compute shader interprets those same 32 bits as
        // a plain float when reading (R32_FLOAT).
        particleDepthDB = DoubleBufferGpuTexture::Create(
            device, width, height,
            DXGI_FORMAT_R32_TYPELESS,
            D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            L"Particle Depth Texture [0] (DTVS)",
            L"Particle Depth Texture [1] (DTVS)",
            D3D12_RESOURCE_STATE_COMMON,
            &clearValue,
            staticAlloc,
            particleDsvAllocator.get(),
            DXGI_FORMAT_UNKNOWN,    // uavFmt (no UAV: ALLOW_DEPTH_STENCIL excludes ALLOW_UNORDERED_ACCESS)
            DXGI_FORMAT_R32_FLOAT,  // srvFmt (compute reads depth as float)
            DXGI_FORMAT_D32_FLOAT); // dsvFmt (graphics writes depth)
    }

    // Build all compute and graphics pipelines, wiring descriptor table slots.
    // particleFields must be a PF_COUNT-element array of the simulation's particle double-buffers.
    // Must be called after the constructor.
    void BuildPipelines(
        ID3D12Device* device,
        D3D12_GPU_VIRTUAL_ADDRESS cbv,
        DescriptorAllocator& mainAlloc,
        DoubleBufferGpuBuffer::P* particleFields)
    {
        using P = com_ptr<ID3D12Resource>*;

        auto copyToMainHeap = [&](UINT slot, D3D12_CPU_DESCRIPTOR_HANDLE src) {
            device->CopyDescriptorsSimple(1, mainAlloc.GetCpuHandle(slot), src,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        };

        UINT slot;
        // clearDtcReductionCS: UAV(u0) = 1 slot 
        slot = mainAlloc.Allocate(1); // needs only 1 descriptor
        copyToMainHeap(slot, lodReductionBuffer->GetUavCpuHandle()); // wire the slot in the main heap
        clearDtcReductionShader = ComputeShader::Create(device, "Shaders/clearDtcReductionCS.cso", cbv,
			mainAlloc.GetGpuHandle(slot), // GPU handle to the start of this shader's contiguous descriptor region in the main heap
			std::vector<P>{}, // no input resources
			std::vector<P>{ lodReductionBuffer->GetResourcePtr() }); // UAV outputs

        // dtcReductionCS: UAV(u0-1) = 2 slots
        // [0]=predictedPosition back, [1]=lodReduction
        slot = mainAlloc.Allocate(2);
        particleFields[PF_PREDICTED_POSITION]->registerBackTarget(mainAlloc.GetCpuHandle(slot), false);
        copyToMainHeap(slot + 1, lodReductionBuffer->GetUavCpuHandle());
        lodReductionShader = ComputeShader::Create(device, "Shaders/dtcReductionCS.cso", cbv,
            mainAlloc.GetGpuHandle(slot),
            std::vector<P>{ particleFields[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr(),
                            lodReductionBuffer->GetResourcePtr() },
            std::vector<P>{ lodReductionBuffer->GetResourcePtr() });
        
        // dtcLodCS: UAV(u0-2) = 3 slots
        // [0]=predictedPosition back, [1]=lod, [2]=lodReduction
        slot = mainAlloc.Allocate(3);
        particleFields[PF_PREDICTED_POSITION]->registerBackTarget(mainAlloc.GetCpuHandle(slot), false);
        copyToMainHeap(slot + 1, lodBuffer->GetUavCpuHandle());
        copyToMainHeap(slot + 2, lodReductionBuffer->GetUavCpuHandle());
        lodShader = ComputeShader::Create(device, "Shaders/dtcLodCS.cso", cbv,
            mainAlloc.GetGpuHandle(slot),
            std::vector<P>{ particleFields[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr(),
                            lodReductionBuffer->GetResourcePtr() },
            std::vector<P>{ lodBuffer->GetResourcePtr() });

        // setLodMaxCS: UAV(u0) = 1 slot 
        slot = mainAlloc.Allocate(1);
        copyToMainHeap(slot, lodBuffer->GetUavCpuHandle());
        setLodMaxShader = ComputeShader::Create(device, "Shaders/setLodMaxCS.cso", cbv,
            mainAlloc.GetGpuHandle(slot),
            std::vector<P>{},
            std::vector<P>{ lodBuffer->GetResourcePtr() });

        // clearDtvsReductionCS: UAV(u0) = 1 slot
        slot = mainAlloc.Allocate(1);
        copyToMainHeap(slot, lodReductionBuffer->GetUavCpuHandle());
        clearDtvsReductionShader = ComputeShader::Create(device, "Shaders/clearDtvsReductionCS.cso", cbv,
            mainAlloc.GetGpuHandle(slot),
            std::vector<P>{},
            std::vector<P>{ lodReductionBuffer->GetResourcePtr() });

        // dtvsReductionCS: UAV(u0-1) SRV(t0) = 3 slots
        // [0]=predictedPosition back, [1]=lodReduction, [2]=depth SRV (front of depth DB)
		slot = mainAlloc.Allocate(3); // needs 3 slots for the two UAVs and one SRV
        // particleFields[PF_PREDICTED_POSITION] is a double buffer, and we're going to be 
        // reading from its back buffer in the 0th uav, i.e. the one pointed to by "slot", wire it in
        particleFields[PF_PREDICTED_POSITION]->registerBackTarget(mainAlloc.GetCpuHandle(slot), /*isSrv=*/false);
        copyToMainHeap(slot + 1, lodReductionBuffer->GetUavCpuHandle());
        particleDepthDB->registerFrontTarget(mainAlloc.GetCpuHandle(slot + 2), /*isSrv=*/true);
        dtvsReductionShader = ComputeShader::Create(device, "Shaders/dtvsReductionCS.cso", cbv,
            mainAlloc.GetGpuHandle(slot),
            std::vector<P>{ particleFields[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr(),
                            lodReductionBuffer->GetResourcePtr() },
            std::vector<P>{ lodReductionBuffer->GetResourcePtr() });

        // dtvsLodCS: UAV(u0-2) SRV(t0) = 4 slots
        // [0]=predictedPosition back, [1]=lod, [2]=lodReduction, [3]=depth SRV
        slot = mainAlloc.Allocate(4);
        particleFields[PF_PREDICTED_POSITION]->registerBackTarget(mainAlloc.GetCpuHandle(slot), false);
        copyToMainHeap(slot + 1, lodBuffer->GetUavCpuHandle());
        copyToMainHeap(slot + 2, lodReductionBuffer->GetUavCpuHandle());
        particleDepthDB->registerFrontTarget(mainAlloc.GetCpuHandle(slot + 3), /*isSrv=*/true);
        dtvsLodShader = ComputeShader::Create(device, "Shaders/dtvsLodCS.cso", cbv,
            mainAlloc.GetGpuHandle(slot ),
            std::vector<P>{ particleFields[PF_PREDICTED_POSITION]->getBack()->GetResourcePtr(),
                            lodReductionBuffer->GetResourcePtr() },
            std::vector<P>{ lodBuffer->GetResourcePtr() });

        // Depth-only graphics PSO (DTVS)
		// Writes no color, just depth: vertex shader reads particle positions from SRV and expands 
		// to quads in the geometry shader; pixel shader discards pixels outside the sphere and writes depth for the rest.
        // Reuses particleVS + particleGS
        com_ptr<ID3DBlob> vertexShader   = Egg::Shader::LoadCso("Shaders/particleVS.cso");
        com_ptr<ID3DBlob> geometryShader = Egg::Shader::LoadCso("Shaders/particleGS.cso");
        com_ptr<ID3DBlob> pixelShader    = Egg::Shader::LoadCso("Shaders/dtvsDepthOnlyPS.cso");
        depthOnlyRootSig = Egg::Shader::LoadRootSignature(device, vertexShader.Get());

        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature    = depthOnlyRootSig.Get();
        psoDesc.VS = { vertexShader->GetBufferPointer(),   vertexShader->GetBufferSize()   };
        psoDesc.GS = { geometryShader->GetBufferPointer(), geometryShader->GetBufferSize() };
        psoDesc.PS = { pixelShader->GetBufferPointer(),    pixelShader->GetBufferSize()    };
        psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        psoDesc.SampleMask = UINT_MAX;
        psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT); // depth test + write on
        psoDesc.InputLayout = { nullptr, 0 }; // no vertex buffer: positions read from SRV
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
        psoDesc.NumRenderTargets  = 0;  // depth-only: no color output
        psoDesc.DSVFormat  = DXGI_FORMAT_D32_FLOAT;
        psoDesc.SampleDesc.Count  = 1;
        psoDesc.SampleDesc.Quality = 0;

        DX_API("Failed to create particle depth-only PSO")
            device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(depthOnlyPso.ReleaseAndGetAddressOf()));
    }

    // Recreate depth textures at new window dimensions. Called from CreateSwapChainResources on resize.
    void Resize(UINT width, UINT height) {
        particleDepthDB->resize2D(width, height);
    }

    // Clear both depth texture slots to 1.0 (far plane). Call once during initialization
    // so the first DTVS compute frame sees valid depth data before any graphics pass has run.
    void RecordDepthClear(ID3D12GraphicsCommandList* cmd) {
        particleDepthDB->getFront()->Transition(D3D12_RESOURCE_STATE_DEPTH_WRITE, cmd);
        particleDepthDB->getBack() ->Transition(D3D12_RESOURCE_STATE_DEPTH_WRITE, cmd);
        cmd->ClearDepthStencilView(
            particleDepthDB->getFront()->GetDsvCpuHandle(), D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
        cmd->ClearDepthStencilView(
            particleDepthDB->getBack() ->GetDsvCpuHandle(), D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
        particleDepthDB->getFront()->Transition(D3D12_RESOURCE_STATE_COMMON, cmd);
        particleDepthDB->getBack() ->Transition(D3D12_RESOURCE_STATE_COMMON, cmd);
    }

    // Dispatch the LOD assignment sequence on the given compute command list.
    // Mode::DTC  — distance-to-camera reduction + per-particle assignment.
    // Mode::DTVS — distance-to-visible-surface reduction + per-particle assignment.
    // Mode::NONE — fill every lod[i] = maxLOD (non-adaptive, all particles run at full quality).
    //
    // Does NOT copy lodBuffer to the snapshot. PbfApp copies it immediately after this call
    // via GetLodBuffer(), before the solver loop that decrements the values.
    void CalculateLod(ID3D12GraphicsCommandList* cmd) {
        UINT numGroups = (numParticles_ + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;

        if (mode == Mode::DTC) {
            clearDtcReductionShader->dispatch_then_barrier(cmd, 1);
            lodReductionShader ->dispatch_then_barrier(cmd, numGroups);
            lodShader ->dispatch_then_barrier(cmd, numGroups);
        }
        else if (mode == Mode::DTVS) {
            // particleDepthDB->getFront() = last frame's depth written by the graphics queue.
            // Descriptor slot already points at front (registered in BuildPipelines).
            particleDepthDB->getFront()->Transition(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, cmd);
            clearDtvsReductionShader->dispatch_then_barrier(cmd, 1);
            dtvsReductionShader ->dispatch_then_barrier(cmd, numGroups);
            dtvsLodShader ->dispatch_then_barrier(cmd, numGroups);
            particleDepthDB->getFront()->Transition(D3D12_RESOURCE_STATE_COMMON, cmd);
        }
        else { // Mode::NONE
            setLodMaxShader->dispatch_then_barrier(cmd, numGroups);
        }
    }

    // Record the DTVS depth-only particle draw into the open graphics command list.
    // Writes particle coverage into particleDepthDB->getBack() (graphics writes back;
    // after flip(), compute reads it as front next frame).
    // Restores the main render target (mainRtvHandle + mainDsvHandle) before returning.
    void DrawParticleDepth(
        ID3D12GraphicsCommandList*  cmd,
        D3D12_GPU_VIRTUAL_ADDRESS   perFrameCbv,
        D3D12_GPU_DESCRIPTOR_HANDLE particleSrvHandle,
        D3D12_CPU_DESCRIPTOR_HANDLE mainDsvHandle,
        D3D12_CPU_DESCRIPTOR_HANDLE mainRtvHandle)
    {
        particleDepthDB->getBack()->Transition(D3D12_RESOURCE_STATE_DEPTH_WRITE, cmd);

        D3D12_CPU_DESCRIPTOR_HANDLE dsv = particleDepthDB->getBack()->GetDsvCpuHandle();
        cmd->OMSetRenderTargets(0, nullptr, FALSE, &dsv);
        cmd->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

        cmd->SetGraphicsRootSignature(depthOnlyRootSig.Get());
        cmd->SetPipelineState(depthOnlyPso.Get());
        cmd->SetGraphicsRootConstantBufferView(0, perFrameCbv);
        cmd->SetGraphicsRootDescriptorTable(1, particleSrvHandle);

        cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
        cmd->DrawInstanced(numParticles_, 1, 0, 0);

        // Restore the main render target so subsequent draws land in the right place.
        cmd->OMSetRenderTargets(1, &mainRtvHandle, FALSE, &mainDsvHandle);

        particleDepthDB->getBack()->Transition(D3D12_RESOURCE_STATE_COMMON, cmd);
    }

    // Expose the LOD buffer so PbfApp can copy its contents into lodSnapshotDB in
    // RecordComputeCommands(), right after CalculateLod() and before the solver loop.
    GpuBuffer::P GetLodBuffer() const { return lodBuffer; }

    // Expose the depth DB so Render() can include it in the double-buffer flip loop.
    DoubleBufferGpuTexture::P GetParticleDepthDB() const { return particleDepthDB; }
GG_ENDCLASS
