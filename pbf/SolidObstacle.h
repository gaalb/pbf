#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Egg/Common.h>
#include <Egg/Mesh/Shaded.h>
#include <Egg/Mesh/Material.h>
#include <Egg/Importer.h>
#include <Egg/Shader.h>
#include <Egg/ConstantBuffer.hpp>
#include <Egg/PsoManager.h>
#include <Egg/Math/Float4x4.h>
#include "ConstantBufferTypes.h"

using namespace Egg::Math;

// Encapsulates a static rigid solid obstacle for the PBF fluid simulation.
// Owns both the renderable mesh (via Egg::Mesh::Shaded) and the SDF volume
// texture used by the collision compute shaders.
//
// Typical usage:
// solidObstacle = SolidObstacle::Create();
// solidObstacle->Load(device, psoManager, "mesh.obj", "mesh.sdf", perFrameCb);
// solidObstacle->UploadSdf(commandList);
// ... execute command list, wait for GPU ...
// solidObstacle->ReleaseUploadResources();
// solidObstacle->CreateSdfSrv(device, descriptorHeap, slotIndex);
// solidObstacle->SetTransform(initialTransform);
GG_CLASS(SolidObstacle)

    Egg::Mesh::Shaded::P mesh;
    Egg::ConstantBuffer<SolidCb> solidCb; // GPU constant buffer for solidVS (model matrix)
    com_ptr<ID3D12Resource> sdfTexture; // Texture3D<float> on default heap
    com_ptr<ID3D12Resource> sdfUpload; // upload buffer, released after GPU copy
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT sdfFootprint; // layout of sdfUpload, used during UploadSdf()
    Float3 sdfObjMin; // SDF AABB min in object space
    Float3 sdfObjMax; // SDF AABB max in object space
    Float4x4 invTransform; // world-to-object, derived from the current transform

    // Read a .sdf binary file, allocate GPU resources, and fill the upload buffer.
    // Called internally by Load(). Does NOT record any GPU commands.
    void LoadSdfFile(ID3D12Device* device, const std::string& filePath) {
        std::string path = "../Media/" + filePath;
        std::ifstream file(path, std::ios::binary);
        if (!file)
            throw std::runtime_error("SolidObstacle: cannot open SDF file: " + path);

        int nx, ny, nz;
        file.read(reinterpret_cast<char*>(&nx), sizeof(int));
        file.read(reinterpret_cast<char*>(&ny), sizeof(int));
        file.read(reinterpret_cast<char*>(&nz), sizeof(int));

        float rawMin[3], rawMax[3];
        file.read(reinterpret_cast<char*>(rawMin), 3 * sizeof(float));
        file.read(reinterpret_cast<char*>(rawMax), 3 * sizeof(float));
        sdfObjMin = Float3(rawMin[0], rawMin[1], rawMin[2]);
        sdfObjMax = Float3(rawMax[0], rawMax[1], rawMax[2]);

        const size_t voxelCount = (size_t)nx * ny * nz;
        std::vector<float> sdfData(voxelCount);
        file.read(reinterpret_cast<char*>(sdfData.data()), voxelCount * sizeof(float));
        if (!file)
            throw std::runtime_error("SolidObstacle: SDF file truncated: " + path);

        // Create the Texture3D on the default heap (GPU-local, COPY_DEST initial state)
        D3D12_RESOURCE_DESC texDesc = {};
        texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        texDesc.Width = (UINT)nx;
        texDesc.Height = (UINT)ny;
        texDesc.DepthOrArraySize = (UINT16)nz;
        texDesc.MipLevels = 1;
        texDesc.Format = DXGI_FORMAT_R32_FLOAT;
        texDesc.SampleDesc.Count  = 1;
        texDesc.SampleDesc.Quality = 0;
        texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        texDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

        DX_API("Failed to create SDF Texture3D")
            device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &texDesc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr,
                IID_PPV_ARGS(sdfTexture.ReleaseAndGetAddressOf()));
        sdfTexture->SetName(L"SDF Texture3D");

        // Query the required upload buffer size and row/slice pitches
        UINT numRows = 0;
        UINT64 rowSizeInBytes = 0;
        UINT64 uploadBufSize = 0;
        device->GetCopyableFootprints(&texDesc, 0, 1, 0, &sdfFootprint, &numRows, &rowSizeInBytes, &uploadBufSize);

        // Create the upload buffer (upload heap, CPU-writable)
        DX_API("Failed to create SDF upload buffer")
            device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(uploadBufSize),
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(sdfUpload.ReleaseAndGetAddressOf()));

        // Map and copy data row-by-row (D3D12 requires RowPitch alignment padding)
        void* mappedPtr = nullptr;
        CD3DX12_RANGE noRead(0, 0);
        DX_API("Failed to map SDF upload buffer")
            sdfUpload->Map(0, &noRead, &mappedPtr);

        const UINT64 rowPitch   = sdfFootprint.Footprint.RowPitch;
        const UINT64 slicePitch = rowPitch * numRows;
        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                UINT8*       dst = static_cast<UINT8*>(mappedPtr)
                                 + sdfFootprint.Offset
                                 + (UINT64)z * slicePitch
                                 + (UINT64)y * rowPitch;
                const float* src = sdfData.data()
                                 + (size_t)z * ny * nx
                                 + (size_t)y * nx;
                memcpy(dst, src, (size_t)nx * sizeof(float));
            }
        }
        CD3DX12_RANGE writtenAll(0, uploadBufSize);
        sdfUpload->Unmap(0, &writtenAll);
    }

public:

    // Accessors for PbfApp::UpdateComputeCb()
    Float3 GetSdfMin() const { return sdfObjMin; }
    Float3 GetSdfMax() const { return sdfObjMax; }
    Float4x4 GetInvTransform() const { return invTransform; }

    // Update the world-space position and orientation of the obstacle.
    // Also uploads the new model matrix to the GPU constant buffer immediately.
    void SetTransform(const Float4x4& t) {
        solidCb->modelMat = t;
        solidCb.Upload();
        invTransform = t.Invert();
    }

    // Load the mesh and SDF file, create all GPU resources, and fill the upload
    // buffer. Must be followed by UploadSdf(commandList) + GPU sync +
    // ReleaseUploadResources(). perFrameCb is owned by PbfApp and stays valid
    // for the lifetime of the material.
    void Load(ID3D12Device*               device,
              Egg::PsoManager::P          psoManager,
              const std::string&          meshPath,
              const std::string&          sdfPath,
              Egg::ConstantBuffer<PerFrameCb>& perFrameCb)
    {
        // --- Mesh ---
        Egg::Mesh::Geometry::P geometry =
            Egg::Importer::ImportSimpleObj(device, meshPath);

        // --- Shaders ---
        com_ptr<ID3DBlob>            vs     = Egg::Shader::LoadCso("Shaders/solidVS.cso");
        com_ptr<ID3DBlob>            ps     = Egg::Shader::LoadCso("Shaders/solidPS.cso");
        com_ptr<ID3D12RootSignature> rootSig = Egg::Shader::LoadRootSignature(device, vs.Get());

        // --- Material ---
        Egg::Mesh::Material::P material = Egg::Mesh::Material::Create();
        material->SetRootSignature(rootSig);
        material->SetVertexShader(vs);
        material->SetPixelShader(ps);
        material->SetDepthStencilState(CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT));
        material->SetDSVFormat(DXGI_FORMAT_D32_FLOAT);

        // Initialise and bind the per-object constant buffer (model matrix)
        solidCb.CreateResources(device);
        solidCb->modelMat = Float4x4::Identity;
        solidCb.Upload();
        invTransform = Float4x4::Identity;

        // Material binds constant buffers by cbuffer name (via D3D12 reflection).
        // "SolidCb" maps to b0, "PerFrameCb" maps to b1 in the solidVS root signature.
        material->SetConstantBuffer(solidCb);
        material->SetConstantBuffer(perFrameCb);

        mesh = Egg::Mesh::Shaded::Create(psoManager, material, geometry);

        // --- SDF texture ---
        LoadSdfFile(device, sdfPath);
    }

    // Record CopyTextureRegion + state transition into the command list.
    // The caller must execute the command list and sync with the GPU before
    // calling ReleaseUploadResources().
    void UploadSdf(ID3D12GraphicsCommandList* commandList) {
        D3D12_TEXTURE_COPY_LOCATION dst = {};
        dst.pResource        = sdfTexture.Get();
        dst.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.SubresourceIndex = 0;

        D3D12_TEXTURE_COPY_LOCATION src = {};
        src.pResource        = sdfUpload.Get();
        src.Type             = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint  = sdfFootprint;

        commandList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

        // Transition to the state required by compute shaders
        commandList->ResourceBarrier(1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                sdfTexture.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
    }

    // Release the CPU-side upload buffer once the GPU copy is confirmed complete.
    void ReleaseUploadResources() {
        sdfUpload = nullptr;
    }

    // Create a Texture3D SRV at the given slot in the shared descriptor heap.
    void CreateSdfSrv(ID3D12Device* device, ID3D12DescriptorHeap* heap, UINT slot) {
        UINT descSize = device->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        CD3DX12_CPU_DESCRIPTOR_HANDLE handle(
            heap->GetCPUDescriptorHandleForHeapStart(), slot, descSize);

        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format                        = DXGI_FORMAT_R32_FLOAT;
        srvDesc.ViewDimension                 = D3D12_SRV_DIMENSION_TEXTURE3D;
        srvDesc.Shader4ComponentMapping       = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Texture3D.MipLevels           = 1;
        srvDesc.Texture3D.MostDetailedMip     = 0;
        srvDesc.Texture3D.ResourceMinLODClamp = 0.0f;

        device->CreateShaderResourceView(sdfTexture.Get(), &srvDesc, handle);
    }

    // Delegate to the underlying shaded mesh.
    void Draw(ID3D12GraphicsCommandList* commandList) {
        mesh->Draw(commandList);
    }

GG_ENDCLASS
