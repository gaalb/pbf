#pragma once
#include "Egg/Common.h"
#include "Egg/Shader.h"
#include "Egg/d3dx12.h"
#include <vector>
#include <string>

// Encapsulates a single compute shader: its PSO, root signature, descriptor table bindings,
// and the resource lists used to insert UAV barriers before or after dispatch.
class ComputeShader {
public:
    com_ptr<ID3D12PipelineState>  pso;
    com_ptr<ID3D12RootSignature>  rootSig;

    // Resources read by this shader. Used by barrier_then_dispatch to flush
    // prior writes before the dispatch begins.
    std::vector<ID3D12Resource*> inputs;

    // Resources written by this shader. Used by dispatch_then_barrier to flush
    // this shader's writes before the next consumer runs.
    std::vector<ID3D12Resource*> outputs;

    // Address of the shared ComputeCb constant buffer. Always bound to root param 0.
    D3D12_GPU_VIRTUAL_ADDRESS cbvAddress = 0;

    // Descriptor table bindings set once during LoadCompute and applied before every dispatch.
    struct TableBinding {
        UINT rootParam;                    // root parameter index in this shader's root signature
        D3D12_GPU_DESCRIPTOR_HANDLE handle; // GPU handle to the descriptor table start
    };
    std::vector<TableBinding> tableBindings;

    // Loads the compiled shader object (.cso), extracts the embedded root signature,
    // and creates the compute PSO. Must be called once before any dispatch.
    void createResources(ID3D12Device* device, const std::string& csoPath);

    // Dispatches the shader, then inserts a UAV barrier on every output resource.
    // Use this pattern (current pipeline) to ensure writes are visible to the next pass.
    void dispatch_then_barrier(ID3D12GraphicsCommandList* cmd, UINT numGroups);

    // Inserts a UAV barrier on every input resource, then dispatches the shader.
    // Use this pattern (future "wait-calculate" ordering) to ensure prior writes
    // to inputs are complete before this shader reads them.
    void barrier_then_dispatch(ID3D12GraphicsCommandList* cmd, UINT numGroups);

private:
    // Binds the root signature, CBV, descriptor tables, and PSO onto the command list.
    void setup(ID3D12GraphicsCommandList* cmd);
};

inline void ComputeShader::createResources(ID3D12Device* device, const std::string& csoPath)
{
    com_ptr<ID3DBlob> shader = Egg::Shader::LoadCso(csoPath);
    rootSig = Egg::Shader::LoadRootSignature(device, shader.Get());

    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = rootSig.Get();
    psoDesc.CS = CD3DX12_SHADER_BYTECODE(shader.Get());
    DX_API("Failed to create compute PSO")
        device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(pso.ReleaseAndGetAddressOf()));
}

inline void ComputeShader::setup(ID3D12GraphicsCommandList* cmd)
{
    cmd->SetComputeRootSignature(rootSig.Get());
    cmd->SetComputeRootConstantBufferView(0, cbvAddress);
    for (auto& b : tableBindings)
        cmd->SetComputeRootDescriptorTable(b.rootParam, b.handle);
    cmd->SetPipelineState(pso.Get());
}

inline void ComputeShader::dispatch_then_barrier(ID3D12GraphicsCommandList* cmd, UINT numGroups)
{
    setup(cmd);
    cmd->Dispatch(numGroups, 1, 1);
    for (auto* r : outputs)
        cmd->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(r));
}

inline void ComputeShader::barrier_then_dispatch(ID3D12GraphicsCommandList* cmd, UINT numGroups)
{
    for (auto* r : inputs)
        cmd->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(r));
    setup(cmd);
    cmd->Dispatch(numGroups, 1, 1);
}
