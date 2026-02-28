#pragma once

#include <Egg/SimpleApp.h>
#include <Egg/Importer.h>
#include <Egg/Math/Math.h>
#include <Egg/Mesh/Prefabs.h>
#include <Egg/ConstantBuffer.hpp>

#include <Egg/Cam/FirstPerson.h>
#include "ConstantBufferTypes.h"

using namespace Egg::Math;

__declspec(align(16)) struct PerBucketCb {
	int bitOffset;
	int dummy1;
	int dummy2;
	int dummy3;
};

class ggl005App : public Egg::SimpleApp {
protected:
	Egg::Mesh::Shaded::P shadedMesh;
	Float4x4 rotation;
	Egg::ConstantBuffer<PerObjectCb> cb;
	com_ptr<ID3D12DescriptorHeap> srvHeap;
	Egg::Texture2D tex;

	Egg::Cam::FirstPerson::P camera;
	Egg::ConstantBuffer<PerFrameCb> perFrameCb;

	Egg::TextureCube env;
	Egg::Mesh::Shaded::P backgroundMesh;
public:
	virtual void Update(float dt, float T) override {
		rotation *= Float4x4::Rotation(Float3::UnitY, dt);
		cb->modelTransform = Float4x4::Translation(Float3{ 0.0f, 0.0f, -0.5f }) * rotation * Float4x4::Translation(Float3{ 0.0f, 0.0f, 0.5f });
		cb.Upload();
		camera->Animate(dt);
		perFrameCb->viewProjMat =
			camera->GetViewMatrix() * 
			camera->GetProjMatrix();
		perFrameCb->rayDirTransform = camera->GetRayDirMatrix();
		perFrameCb.Upload();
	}

	virtual void PopulateCommandList() override {
		commandAllocator->Reset();
		commandList->Reset(commandAllocator.Get(), nullptr);

		commandList->RSSetViewports(1, &viewPort);
		commandList->RSSetScissorRects(1, &scissorRect);

		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(renderTargets[swapChainBackBufferIndex].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));

		CD3DX12_CPU_DESCRIPTOR_HANDLE rHandle(rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart(), swapChainBackBufferIndex, rtvDescriptorHandleIncrementSize);
		CD3DX12_CPU_DESCRIPTOR_HANDLE dsvHandle(dsvHeap->GetCPUDescriptorHandleForHeapStart());
		commandList->OMSetRenderTargets(1, &rHandle, FALSE, &dsvHandle);

		const float clearColor[] = { 0.0f, 0.2f, 0.4f, 1.0f };
		commandList->ClearRenderTargetView(rHandle, clearColor, 0, nullptr);
		commandList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

		ID3D12DescriptorHeap* descriptorHeaps[] = { srvHeap.Get() };
		commandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

		shadedMesh->SetPipelineState(commandList.Get());
		shadedMesh->BindConstantBuffer(commandList.Get(), cb);
		shadedMesh->BindConstantBuffer(commandList.Get(), perFrameCb);
		commandList->SetGraphicsRootDescriptorTable(2, srvHeap->GetGPUDescriptorHandleForHeapStart());
		shadedMesh->Draw(commandList.Get());

		backgroundMesh->SetPipelineState(commandList.Get());
		backgroundMesh->BindConstantBuffer(commandList.Get(), perFrameCb);
		commandList->SetGraphicsRootDescriptorTable(2, srvHeap->GetGPUDescriptorHandleForHeapStart());
		backgroundMesh->Draw(commandList.Get());

		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(renderTargets[swapChainBackBufferIndex].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));

		DX_API("Failed to close command list")
			commandList->Close();
	}

	/*
	Almost a render call
	*/
	void UploadResources() {
		DX_API("Failed to reset command allocator (UploadResources)")
			commandAllocator->Reset();
		DX_API("Failed to reset command list (UploadResources)")
			commandList->Reset(commandAllocator.Get(), nullptr);

		tex.UploadResource(commandList.Get());
		env.UploadResource(commandList.Get());

		DX_API("Failed to close command list (UploadResources)")
			commandList->Close();

		ID3D12CommandList * commandLists[] = { commandList.Get() };
		commandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);

		WaitForPreviousFrame();

		tex.ReleaseUploadResources();
		env.ReleaseUploadResources();
	}

	virtual void CreateResources() override {
		Egg::SimpleApp::CreateResources();

		D3D12_DESCRIPTOR_HEAP_DESC dhd;
		dhd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
		dhd.NodeMask = 0;
		dhd.NumDescriptors = 2;
		dhd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;

		DX_API("Failed to create descriptor heap for texture")
			device->CreateDescriptorHeap(&dhd, IID_PPV_ARGS(srvHeap.GetAddressOf()));
		
		camera = Egg::Cam::FirstPerson::Create();
	}

	virtual void ReleaseResources() override {
		srvHeap.Reset();

		Egg::SimpleApp::ReleaseResources();
	}

	virtual void LoadAssets() override {


		com_ptr<ID3DBlob> vertexShader = Egg::Shader::LoadCso("Shaders/trafoVS.cso");
		com_ptr<ID3DBlob> pixelShader = Egg::Shader::LoadCso("Shaders/DefaultPS.cso");
		com_ptr<ID3D12RootSignature> rootSig = Egg::Shader::LoadRootSignature(device.Get(), vertexShader.Get());

		Egg::Mesh::Material::P material = Egg::Mesh::Material::Create();
		material->SetRootSignature(rootSig);
		material->SetVertexShader(vertexShader);
		material->SetPixelShader(pixelShader);
		material->SetDepthStencilState(CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT));
		material->SetDSVFormat(DXGI_FORMAT_D32_FLOAT);

		Egg::Mesh::Geometry::P geometry = Egg::Importer::ImportSimpleObj(device.Get(), "giraffe.obj");

		shadedMesh = Egg::Mesh::Shaded::Create(psoManager.get(), material, geometry);
		
		com_ptr<ID3DBlob> bgVertexShader = Egg::Shader::LoadCso("Shaders/quadVS.cso");
		com_ptr<ID3DBlob> bgPixelShader = Egg::Shader::LoadCso("Shaders/bgPS.cso");

		Egg::Mesh::Material::P bgMaterial = Egg::Mesh::Material::Create();
		bgMaterial->SetRootSignature(rootSig);
		bgMaterial->SetVertexShader(bgVertexShader);
		bgMaterial->SetPixelShader(bgPixelShader);
		bgMaterial->SetDepthStencilState(CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT));
		bgMaterial->SetDSVFormat(DXGI_FORMAT_D32_FLOAT);
		
		backgroundMesh = Egg::Mesh::Shaded::Create(psoManager.get(), 
			bgMaterial,
			Egg::Mesh::Prefabs::FullScreenQuad(device.Get()));

		tex = Egg::Importer::ImportTexture2D(device.Get(), "giraffe.jpg");
		tex.CreateSRV(device.Get(), srvHeap.Get(), 0);
		env = Egg::Importer::ImportTextureCube(device.Get(), "cloudynoon.dds");
		env.CreateSRV(device.Get(), srvHeap.Get(), 1);

		UploadResources();
	}

	void ProcessMessage(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) override
	{
		camera->ProcessMessage(hWnd, uMsg, wParam, lParam);
	}

};


