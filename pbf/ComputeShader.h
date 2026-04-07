#pragma once
#include "Egg/Common.h"
#include "Egg/Shader.h"
#include "Egg/d3dx12.h"
#include <vector>
#include <string>

// Encapsulates a single compute shader: its PSO, root signature, descriptor table bindings,
// and the resource lists used to insert UAV barriers before or after dispatch.
// Follows the Egg GG_CLASS pattern: store as ComputeShader::P (shared_ptr), create via ComputeShader::Create(...)
GG_CLASS(ComputeShader)
    com_ptr<ID3D12PipelineState> pso;
    com_ptr<ID3D12RootSignature> rootSig;

public:
    // Resources read by this shader. Used by barrier_then_dispatch to flush
    // prior writes before the dispatch begins.
    // Stored as pointers-to-com_ptr so that swapping the owning com_ptrs (double-buffering)
    // automatically updates the resource pointers resolved here at record time.
    std::vector<com_ptr<ID3D12Resource>*> inputs;

    // Resources written by this shader. Used by dispatch_then_barrier to flush
    // this shader's writes before the next consumer runs.
    std::vector<com_ptr<ID3D12Resource>*> outputs;

    // Address of the shared ComputeCb constant buffer. Always bound to root param 0.
    D3D12_GPU_VIRTUAL_ADDRESS cbvAddress = 0;

    // Descriptor table bindings applied before every dispatch.
    struct TableBinding {
        UINT rootParam; // root parameter index in this shader's root signature
        // Pointer to the handle member in PbfApp, resolved at record time via *handlePtr.
        // This lets the caller swap the handle value (e.g. particleFieldsHandle <-> sortedFieldsHandle)
        // between dispatches to re-route subsequent commands to different descriptor ranges,
        // without modifying the descriptor heap or rebuilding the pipeline.
        D3D12_GPU_DESCRIPTOR_HANDLE* handlePtr;
    };
    std::vector<TableBinding> tableBindings;

    // Loads the compiled shader object (.cso), extracts the embedded root signature,
    // creates the compute PSO, and stores all binding metadata.
    ComputeShader(
        ID3D12Device* device,
        const std::string& csoPath,
        D3D12_GPU_VIRTUAL_ADDRESS cbv,
        std::vector<TableBinding> tbs,
        std::vector<com_ptr<ID3D12Resource>*> ins,
        std::vector<com_ptr<ID3D12Resource>*> outs)
        : cbvAddress(cbv)
        , tableBindings(std::move(tbs)) // move: avoid copy
        , inputs(std::move(ins)) // move: avoid copy
        , outputs(std::move(outs)) // move: avoid copy
    {
        // extract roog signature
        com_ptr<ID3DBlob> shader = Egg::Shader::LoadCso(csoPath);
        rootSig = Egg::Shader::LoadRootSignature(device, shader.Get());

        // create the pipeline state object using the root signature and shader bytecode
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = rootSig.Get();
        psoDesc.CS = CD3DX12_SHADER_BYTECODE(shader.Get());
        DX_API("Failed to create compute PSO")
            device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(pso.ReleaseAndGetAddressOf()));
    }

    // Dispatches the shader, then inserts a UAV barrier on every output resource.
    // Use this pattern (current pipeline) to ensure writes are visible to the next pass.
    void dispatch_then_barrier(ID3D12GraphicsCommandList* cmd, UINT numGroups)
    {
        setup(cmd);
        cmd->Dispatch(numGroups, 1, 1);
        for (auto* r : outputs)
            cmd->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(r->Get()));
    }

    // Inserts a UAV barrier on every input resource, then dispatches the shader.
    // Use this pattern (future "wait-calculate" ordering) to ensure prior writes
    // to inputs are complete before this shader reads them.
    //
    // Note: using this instead of dispatch_then_barrier is an interesting idea, but
    // needs further consideration, due to it leading to WAR hazards. Specifically,
    // we had this:
    // 1. countGridShader -> writes cellCount(atomic increment per particle)
    // 2. prefixSumShader -> reads cellCount, writes cellPrefixSum
    // 3. clearGridShader -> zeroes cellCount
    // 4. sortShader      -> reads cellPrefixSum, cellCount, predictedPosition
    //
    // If we used barrier_then_dispatch here, then steps 2 and 3 *think* they
    // can run concurrently, because prefixSumShader waits for all operations
    // on cellCount to finish before dispatching, which is good. However, since
    // clearGridShader doesn't *read* cellCount, it didn't declare it as an input,
    // and therefore doesn't wait for reads and writes on it to be finished before
    // dispatching. This makes it possible for clearGridShader to start immediately
    // after prefixSumShader, and lets them run concurrently, which is incorrect,
    // because clearGridShader overwrites cell data taht prefixSumShader hasn't yet
    // had time to read. So, we must redefine "input" from "data this shader reads"
    // to "buffer that must be stable before shader is launched". This has already been
    // done in clearGridShader, cellCount was added as an input, but I haven't yet
    // thought this through for the other shaders. It's simpler to just use
    // dispatch_then_barrier, which is free of such issues (I think?)
    //
    void barrier_then_dispatch(ID3D12GraphicsCommandList* cmd, UINT numGroups)
    {
        for (auto* r : inputs)
            cmd->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(r->Get()));
        setup(cmd);
        cmd->Dispatch(numGroups, 1, 1);
    }

private:
    // Binds the root signature, CBV, descriptor tables, and PSO onto the command list.
    // handlePtr is dereferenced here at record time, so callers can swap the handle value
    // (e.g. particleFieldsHandle <-> sortedFieldsHandle) between dispatches to re-route
    // subsequent commands to a different descriptor range without modifying the heap.
    void setup(ID3D12GraphicsCommandList* cmd)
    {
        cmd->SetComputeRootSignature(rootSig.Get());
        cmd->SetComputeRootConstantBufferView(0, cbvAddress);
        for (auto& b : tableBindings)
            cmd->SetComputeRootDescriptorTable(b.rootParam, *b.handlePtr);
        cmd->SetPipelineState(pso.Get());
    }
GG_ENDCLASS
