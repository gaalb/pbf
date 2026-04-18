#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Egg/Common.h>
#include <Egg/Mesh/Shaded.h>
#include "DescriptorAllocator.h"
#include <Egg/Mesh/Material.h>
#include <Egg/Importer.h>
#include <Egg/Shader.h>
#include <Egg/ConstantBuffer.hpp>
#include <Egg/PsoManager.h>
#include <Egg/Math/Float4x4.h>
#include "ConstantBufferTypes.h"

using namespace Egg::Math;

// Encapsulates a renderable mesh (via Egg::Mesh::Shaded) and its SDF volume texture
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
    D3D12_GPU_DESCRIPTOR_HANDLE sdfGpuHandle{};
    D3D12_CPU_DESCRIPTOR_HANDLE sdfCpuHandle{};

    // Read a .sdf binary file, allocate GPU resources, and fill the upload buffer.
    // Called internally by Load(). Does NOT record any GPU commands.
    void LoadSdfFile(ID3D12Device* device, const std::string& filePath) {
        std::string path = "../Media/" + filePath; // resource files such as .obj and .sdf are stored in the Media folder
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("SolidObstacle: cannot open SDF file: " + path); // file didn't open

        // .read(ptr count) copies count raw bytes from the file into the memory at ptr
        // read also advances the file position, so sequential reads pick up where
        // the previous one left off
        // reinterpret_cast is needed because .read() takes char*, but our data isn't chars
        // 
        // the .sdf file format is a flat binary blob with this layout:
        // [nx] [ny] [nz] � 3 ints : voxel grid dimensions
        // [rawMin.x y z][rawMax.x y z] � 6 floats : AABB bounds in object space
        // [d gx gy gz] �(nx * ny * nz) � 4 floats per voxel : signed distance + gradient xyz     
        int nx, ny, nz; // start with reading the grid resolution (e.g. 64x64x64)
        file.read(reinterpret_cast<char*>(&nx), sizeof(int));
        file.read(reinterpret_cast<char*>(&ny), sizeof(int));
        file.read(reinterpret_cast<char*>(&nz), sizeof(int));

        float rawMin[3], rawMax[3]; // axis aligned bounding box in raw float array form
        file.read(reinterpret_cast<char*>(rawMin), 3 * sizeof(float));
        file.read(reinterpret_cast<char*>(rawMax), 3 * sizeof(float));
        sdfObjMin = Float3(rawMin[0], rawMin[1], rawMin[2]);
        sdfObjMax = Float3(rawMax[0], rawMax[1], rawMax[2]);

        const size_t voxelCount = (size_t)(nx * ny * nz);
        std::vector<float> sdfData(voxelCount * 4); // float4 per voxel: (d, gx, gy, gz)
        // vector's .data() is raw pointer to contiguous internal array, that's where we direct
        // the bult read of the voxels' data arranged distance-gradx-grady-gradz, which will
        // map to the texture format of R32G32B32A32
        file.read(reinterpret_cast<char*>(sdfData.data()), voxelCount * 4 * sizeof(float)); 
        if (!file) throw std::runtime_error("SolidObstacle: SDF file truncated: " + path);

        // Create the Texture3D on the default heap (GPU-local, COPY_DEST initial state)
        D3D12_RESOURCE_DESC texDesc = {};
		texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D; // this is a 3D volume texture, not a buffer
		texDesc.Width = (UINT)nx; // width = x resolution of the voxel grid
		texDesc.Height = (UINT)ny; // height = y resolution of the voxel grid
		texDesc.DepthOrArraySize = (UINT16)nz; // depth = z resolution of the voxel grid
		texDesc.MipLevels = 1; 
        texDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT; // R=distance, GBA=gradient xyz
		texDesc.SampleDesc.Count = 1; // no multisampling for volume texture
		texDesc.SampleDesc.Quality = 0;
        texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN; 
        texDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

        DX_API("Failed to create SDF Texture3D")
            device->CreateCommittedResource(
				&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), // default heap = GPU-local memory, not CPU-writable
				D3D12_HEAP_FLAG_NONE, // no special flags for the heap
                &texDesc, // the description we just filled out
				D3D12_RESOURCE_STATE_COPY_DEST, // the first time we touch this resource will be when we upload the data to it from the upload buffer
                nullptr, 
                IID_PPV_ARGS(sdfTexture.ReleaseAndGetAddressOf())); // save pointer to it
        sdfTexture->SetName(L"SDF Texture3D");

        // Query the required upload buffer size and row/slice pitches
        // GPU textures don't store rorws as tightly as packed arrays, each row has
        // a pitch - the number of bytes from the start of one the next - 
        // which must be aligned (typicall to 256 bytes). GetCopyableFootprint
        // tells us how the GPU expects the data to be laid out in an upload
        // buffer for a CopyTextureRegion call.
        UINT numRows = 0;
        UINT64 rowSizeInBytes = 0;
        UINT64 uploadBufSize = 0;
        device->GetCopyableFootprints(
            &texDesc, // the Texture3D description we want to copy into
            0, // FirstSubresource: start at mip 0
            1, // NumSubresources: just 1 mip level
            0, // BaseOffset: no offset in the upload buffer
            &sdfFootprint, // out: layout description (Offset, Format, Width, Height, Depth, RowPitch)
            &numRows, // out: rows per slice
            &rowSizeInBytes, // out: actual data bytes per row (no padding)
            &uploadBufSize // out: total upload buffer size needed
        );

        // Create the upload buffer (upload heap, CPU-writable) with the size calculated in the previous call
        DX_API("Failed to create SDF upload buffer")
            device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(uploadBufSize),
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(sdfUpload.ReleaseAndGetAddressOf()));

        // Map and copy data row-by-row (D3D12 requires RowPitch alignment padding)
        void* mappedPtr = nullptr; // this is the cpu side pointer that will point to the upload buffer
        // The flow of mapping will be the following: 
        // -we declare the range we want to read, then call map 
        // -do whatever reads/writes we want
        // -declare what range we dirtied by writing to it, and unmap with it
        CD3DX12_RANGE noRead(0, 0); // we don't read
        DX_API("Failed to map SDF upload buffer")
            sdfUpload->Map(0, &noRead, &mappedPtr);

        const UINT64 rowPitch   = sdfFootprint.Footprint.RowPitch; // bytes between starts of rows
        const UINT64 slicePitch = rowPitch * numRows; // bytes between starts of slices (z layers)
        // the z and y loops say: for each row, of each slize, do ...
        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                // first calculate where the destination row resides
                UINT8*  dst = static_cast<UINT8*>(mappedPtr) // start from the base pointer
                    + sdfFootprint.Offset // flat offset if there is any 
                    + (UINT64)z * slicePitch // skip past the first z layers (slizes)
                    + (UINT64)y * rowPitch; // skip past the first y rows in this slice
                // then calculate where the data resides in the tightly packed source
                const float* src = sdfData.data() // start from the base pointer
                    + ((size_t)z * ny * nx // skip past the first z layers (slices) of ny * nx size
                        + (size_t)y * nx) // skip past the first y rows in this slice, all of which are nx size
                    * 4; // each element is 4 floats
                memcpy(dst, src, (size_t)nx * 4 * sizeof(float));
            }
        }
        CD3DX12_RANGE writtenAll(0, uploadBufSize); // we dirty a whole uploadBufferSize's worth
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
              Egg::PsoManager::P psoManager,
              const std::string& meshPath,
              const std::string&  sdfPath,
              Egg::ConstantBuffer<PerFrameCb>& perFrameCb)
    {
        Egg::Mesh::Geometry::P geometry = Egg::Importer::ImportSimpleObj(device, meshPath);

        com_ptr<ID3DBlob> vs = Egg::Shader::LoadCso("Shaders/solidVS.cso");
        com_ptr<ID3DBlob> ps = Egg::Shader::LoadCso("Shaders/solidPS.cso");
        com_ptr<ID3D12RootSignature> rootSig = Egg::Shader::LoadRootSignature(device, vs.Get());

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

    // Allocate one slot from alloc and create a Texture3D SRV for the SDF volume.
    void CreateSdfSrv(ID3D12Device* device, DescriptorAllocator& alloc) {
        UINT slot = alloc.Allocate();
        D3D12_CPU_DESCRIPTOR_HANDLE cpu = alloc.GetCpuHandle(slot);

        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format                        = DXGI_FORMAT_R32G32B32A32_FLOAT;
        srvDesc.ViewDimension                 = D3D12_SRV_DIMENSION_TEXTURE3D;
        srvDesc.Shader4ComponentMapping       = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Texture3D.MipLevels           = 1;
        srvDesc.Texture3D.MostDetailedMip     = 0;
        srvDesc.Texture3D.ResourceMinLODClamp = 0.0f;

        device->CreateShaderResourceView(sdfTexture.Get(), &srvDesc, cpu);
        sdfGpuHandle = alloc.GetGpuHandle(slot);
        sdfCpuHandle = cpu;
    }

    D3D12_GPU_DESCRIPTOR_HANDLE GetSdfGpuHandle() const { return sdfGpuHandle; }
    D3D12_CPU_DESCRIPTOR_HANDLE GetSdfCpuHandle() const { return sdfCpuHandle; }

    // Delegate to the underlying shaded mesh.
    void Draw(ID3D12GraphicsCommandList* commandList) {
        mesh->Draw(commandList);
    }

GG_ENDCLASS
