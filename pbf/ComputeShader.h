#pragma once
#include "Egg/Common.h"
#include "Egg/Shader.h"
#include "Egg/d3dx12.h"
#include <vector>
#include <string>

// Encapsulates a single compute shader: its PSO, root signature, a fixed descriptor table handle,
// and the resource lists used to insert UAV barriers before or after dispatch.
// As per the Egg GG_CLASS pattern: store as ComputeShader::P (shared_ptr), create via ComputeShader::Create(...)
GG_CLASS(ComputeShader)
    com_ptr<ID3D12PipelineState> pso;
    com_ptr<ID3D12RootSignature> rootSig;

public:
    // Pointers-to-com_ptrs rather than raw resource pointers. GpuBuffer::SwapInternals changes the
    // value held inside the GpuBuffer (i.e. the com_ptr's value) without moving the GpuBuffer object,
    // so these addresses stay valid across flip() calls while still resolving to the current resource.
    std::vector<com_ptr<ID3D12Resource>*> inputs;
    std::vector<com_ptr<ID3D12Resource>*> outputs;

    // Address of the shared ComputeCb constant buffer. Always bound to root param 0.
    D3D12_GPU_VIRTUAL_ADDRESS cbvAddress = 0;

    // GPU handle to the start of this shader's contiguous descriptor region in the main heap.
    // Baked at construction time; setup() always binds it to root param 1.
    D3D12_GPU_DESCRIPTOR_HANDLE fixedHandle{};

    ComputeShader(
        ID3D12Device* device,
        const std::string& csoPath,
        D3D12_GPU_VIRTUAL_ADDRESS cbv,
        D3D12_GPU_DESCRIPTOR_HANDLE table,
        std::vector<com_ptr<ID3D12Resource>*> ins,
        std::vector<com_ptr<ID3D12Resource>*> outs)
        : cbvAddress(cbv)
        , fixedHandle(table)
        , inputs(std::move(ins))
        , outputs(std::move(outs))
    {
        com_ptr<ID3DBlob> shader = Egg::Shader::LoadCso(csoPath);
        rootSig = Egg::Shader::LoadRootSignature(device, shader.Get());

        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = rootSig.Get();
        psoDesc.CS = CD3DX12_SHADER_BYTECODE(shader.Get());
        DX_API("Failed to create compute PSO")
            device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(pso.ReleaseAndGetAddressOf()));
    }

    // Dispatches the shader, then inserts a UAV barrier on every output resource.
    void dispatch_then_barrier(ID3D12GraphicsCommandList* cmd, UINT numGroups)
    {
        setup(cmd);
        cmd->Dispatch(numGroups, 1, 1);
        for (auto* r : outputs)
            cmd->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(r->Get()));
    }

    // 3D variant of dispatch_then_barrier. Use when a 1D dispatch would exceed
    // D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION (65535) on a single axis.
    void dispatch3d_then_barrier(ID3D12GraphicsCommandList* cmd, UINT gx, UINT gy, UINT gz)
    {
        setup(cmd);
        cmd->Dispatch(gx, gy, gz);
        for (auto* r : outputs)
            cmd->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(r->Get()));
    }

    // Inserts a UAV barrier on every input resource, then dispatches the shader.
    void barrier_then_dispatch(ID3D12GraphicsCommandList* cmd, UINT numGroups)
    {
        for (auto* r : inputs)
            cmd->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(r->Get()));
        setup(cmd);
        cmd->Dispatch(numGroups, 1, 1);
    }

private:
    // SetComputeRootSignature resets all compute root bindings, so setup() must be called
    // before every dispatch even though fixedHandle never changes.
    void setup(ID3D12GraphicsCommandList* cmd)
    {
        cmd->SetComputeRootSignature(rootSig.Get());
        cmd->SetComputeRootConstantBufferView(0, cbvAddress);
        cmd->SetComputeRootDescriptorTable(1, fixedHandle);
        cmd->SetPipelineState(pso.Get());
    }
GG_ENDCLASS
