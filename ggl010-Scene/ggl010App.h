#pragma once

#include <Egg/SimpleApp.h>
#include <Egg/Importer.h>
#include <Egg/Math/Math.h>
#include <Egg/Mesh/Prefabs.h>
#include <Egg/Mesh/Multi.h>
#include <Egg/ConstantBuffer.hpp>

#include <Egg/Cam/FirstPerson.h>
#include <Egg/Scene/Entity.h>
#include <Egg/Scene/FixedRigidBody.h>
#include "ConstantBufferTypes.h"
#include "Particle.h"
#include <vector>
#include <algorithm>

#include <Egg/Scene/PerObjectData.h>

using namespace Egg::Math;

class ggl010App : public Egg::SimpleApp {
protected:
	Egg::Mesh::Shaded::P shadedMesh;
	Egg::Mesh::Multi::P multiMesh;
	std::vector<Egg::Scene::Entity::P> entities;
	Float4x4 rotation;
	Egg::ConstantBuffer<PerObjectCb> cb;

	com_ptr<ID3D12DescriptorHeap> srvHeap;
	com_ptr<ID3D12DescriptorHeap> particleSrvHeap;
	Egg::Texture2D tex;
	Egg::Texture2D particleTex;

	Egg::Cam::FirstPerson::P camera;
	Egg::ConstantBuffer<PerFrameCb> perFrameCb;

	Egg::TextureCube env;
	Egg::Mesh::Shaded::P backgroundMesh;
	Egg::Mesh::Shaded::P fireBillboardSet;

	std::vector<Particle> particles;
	Egg::Mesh::VertexStreamGeometry::P particlesGeometry;
public:
	virtual void Update(float dt, float T) override {
		rotation *= Float4x4::Rotation(Float3::UnitY, dt);
		//for (int i = 0; i < 100; i++) {
		//	cb->objects[i].modelTransform = Float4x4::Translation(Float3{ -(float)i, 0.0f, -0.5f }) * rotation * Float4x4::Translation(Float3{ (float)i, 0.0f, 0.5f });
		//	cb->objects[i].modelTransformInverse = cb->objects[0].modelTransform.Invert();
		//}
		for (int i = 0; i < entities.size(); i++) {
			entities[i]->Update(dt, T, cb->objects[i]);
		}
		cb.Upload();
		camera->Animate(dt);
		perFrameCb->viewProjMat =
			camera->GetViewMatrix() * 
			camera->GetProjMatrix();
		perFrameCb->rayDirTransform = camera->GetRayDirMatrix();
		perFrameCb->cameraPos = Float4(camera->GetEyePosition(), 1);
		perFrameCb->lightPos = Float4(0, 1, 0, 0);
		perFrameCb->lightPowerDensity = Float4(2, 1, 2, 0);
		perFrameCb->billboardSize = Float4(50.1, 50.1, 0, 0) * camera->GetProjMatrix();
		perFrameCb.Upload();

		for (int i = 0; i < particles.size(); i++)
			particles.at(i).move(dt);
		using namespace Egg::Math;
		struct CameraDepthComparator {
			Float3 ahead;
			bool operator()(const Particle& a,
				const Particle& b) {
				return
					a.position.Dot(ahead) >
					b.position.Dot(ahead) + 0.01;
			}
		} comp = { camera->GetAhead() };
		std::sort(particles.begin(), particles.end(), comp);

		particlesGeometry->SetData(particles.data(), particles.size() * sizeof(particles[0]));
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

//		ID3D12DescriptorHeap* descriptorHeaps[] = { srvHeap.Get()};
//		commandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

//		shadedMesh->SetPipelineState(commandList.Get());
//		shadedMesh->BindConstantBuffer(commandList.Get(), cb);
//		shadedMesh->BindConstantBuffer(commandList.Get(), perFrameCb);
//		commandList->SetGraphicsRootDescriptorTable(2, srvHeap->GetGPUDescriptorHandleForHeapStart());
//		shadedMesh->Draw(commandList.Get());
//		for (int i = 0; i < 100; i++) {
//			multiMesh->Draw(commandList.Get(), 0, i);
//		}

//s		for (int i = 0; i < entities.size(); i++) {
//s			entities[i]->Draw(commandList.Get(), 0, i);
//s		}

//		backgroundMesh->SetPipelineState(commandList.Get());
//		backgroundMesh->BindConstantBuffer(commandList.Get(), perFrameCb);
//		commandList->SetGraphicsRootDescriptorTable(2, srvHeap->GetGPUDescriptorHandleForHeapStart());
//s		backgroundMesh->Draw(commandList.Get());

		////// START
//		ID3D12DescriptorHeap* descriptorHeaps2[] = { particleSrvHeap.Get() };
//		commandList->SetDescriptorHeaps(_countof(descriptorHeaps2), descriptorHeaps2);

//		fireBillboardSet->SetPipelineState(commandList.Get());
//		fireBillboardSet->BindConstantBuffer(commandList.Get(), cb);
//		fireBillboardSet->BindConstantBuffer(commandList.Get(), perFrameCb);
//		commandList->SetGraphicsRootDescriptorTable(2, particleSrvHeap->GetGPUDescriptorHandleForHeapStart());
//s		fireBillboardSet->Draw(commandList.Get());
		///// END
	
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
		particleTex.UploadResource(commandList.Get());

		DX_API("Failed to close command list (UploadResources)")
			commandList->Close();

		ID3D12CommandList * commandLists[] = { commandList.Get() };
		commandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);

		WaitForPreviousFrame();

		tex.ReleaseUploadResources();
		env.ReleaseUploadResources();
		particleTex.ReleaseUploadResources();
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
		
		DX_API("Failed to create descriptor heap for texture")
			device->CreateDescriptorHeap(&dhd, IID_PPV_ARGS(particleSrvHeap.GetAddressOf()));

		camera = Egg::Cam::FirstPerson::Create();
	}

	virtual void ReleaseResources() override {
		srvHeap.Reset();
		particleSrvHeap.Reset();

		Egg::SimpleApp::ReleaseResources();
	}

	virtual void LoadAssets() override {
		cb.CreateResources(device.Get());
		perFrameCb.CreateResources(device.Get());

		com_ptr<ID3DBlob> vertexShader = Egg::Shader::LoadCso("Shaders/trafoVS.cso");
//		com_ptr<ID3DBlob> pixelShader = Egg::Shader::LoadCso("Shaders/DefaultPS.cso");
//		com_ptr<ID3DBlob> pixelShader = Egg::Shader::LoadCso("Shaders/EnvMapPS.cso");
//		com_ptr<ID3DBlob> pixelShader = Egg::Shader::LoadCso("Shaders/DiffusePS.cso");
		com_ptr<ID3DBlob> pixelShader = Egg::Shader::LoadCso("Shaders/MaxBlinnPS.cso");
		com_ptr<ID3D12RootSignature> rootSig = Egg::Shader::LoadRootSignature(device.Get(), vertexShader.Get());

		Egg::Mesh::Material::P material = Egg::Mesh::Material::Create();
		material->SetRootSignature(rootSig);
		material->SetVertexShader(vertexShader);
		material->SetPixelShader(pixelShader);
		material->SetDepthStencilState(CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT));
		material->SetDSVFormat(DXGI_FORMAT_D32_FLOAT);
		material->SetSrvHeap(2, srvHeap);
		material->SetConstantBuffer(cb, sizeof(Egg::Scene::PerObjectData));
		material->SetConstantBuffer(perFrameCb);

		Egg::Mesh::Geometry::P geometry = Egg::Importer::ImportSimpleObj(device.Get(), "giraffe.obj");

		shadedMesh = Egg::Mesh::Shaded::Create(psoManager, material, geometry);
		
		multiMesh = Egg::Mesh::Multi::Create();
		auto flip = Egg::Mesh::Flip::Create();
		flip->Add(0, shadedMesh);
		multiMesh->Add(flip);

		com_ptr<ID3DBlob> bgVertexShader = Egg::Shader::LoadCso("Shaders/quadVS.cso");
		com_ptr<ID3DBlob> bgPixelShader = Egg::Shader::LoadCso("Shaders/bgPS.cso");

		Egg::Mesh::Material::P bgMaterial = Egg::Mesh::Material::Create();
		bgMaterial->SetRootSignature(rootSig);
		bgMaterial->SetVertexShader(bgVertexShader);
		bgMaterial->SetPixelShader(bgPixelShader);
		bgMaterial->SetDepthStencilState(CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT));
		bgMaterial->SetDSVFormat(DXGI_FORMAT_D32_FLOAT);
		bgMaterial->SetSrvHeap(2, srvHeap);
		bgMaterial->SetConstantBuffer(perFrameCb);
		
		backgroundMesh = Egg::Mesh::Shaded::Create(psoManager, 
			bgMaterial,
			Egg::Mesh::Prefabs::FullScreenQuad(device.Get()));

		tex = Egg::Importer::ImportTexture2D(device.Get(), "giraffe.jpg");
		tex.CreateSRV(device.Get(), srvHeap.Get(), 0);
		env = Egg::Importer::ImportTextureCube(device.Get(), "cloudynoon.dds");
		env.CreateSRV(device.Get(), srvHeap.Get(), 1);
		particleTex = Egg::Importer::ImportTexture2D(device.Get(), "particle.png");
		particleTex.CreateSRV(device.Get(), particleSrvHeap.Get(), 0);

		////////////// START
		for (int i = 0; i < 40; i++)
			particles.push_back(Particle());

		particlesGeometry = 
			Egg::Mesh::VertexStreamGeometry::Create(device.Get(),
				particles.data(),
				sizeof(particles[0]) * particles.size(),
				sizeof(particles[0]));
		particlesGeometry->SetTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
		D3D12_INPUT_ELEMENT_DESC particlePositionDesc;
		particlePositionDesc.AlignedByteOffset = offsetof(Particle, position);
		particlePositionDesc.Format = DXGI_FORMAT_R32G32B32_FLOAT;
		particlePositionDesc.InputSlot = 0;
		particlePositionDesc.InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
		particlePositionDesc.InstanceDataStepRate = 0;
		particlePositionDesc.SemanticIndex = 0;
		particlePositionDesc.SemanticName = "POSITION";
		particlesGeometry->AddInputElement(particlePositionDesc);
		D3D12_INPUT_ELEMENT_DESC particleLifespanDesc;
		particleLifespanDesc.AlignedByteOffset = offsetof(Particle, lifespan);
		particleLifespanDesc.Format = DXGI_FORMAT_R32_FLOAT;
		particleLifespanDesc.InputSlot = 0;
		particleLifespanDesc.InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
		particleLifespanDesc.InstanceDataStepRate = 0;
		particleLifespanDesc.SemanticIndex = 0;
		particleLifespanDesc.SemanticName = "LIFESPAN";
		particlesGeometry->AddInputElement(particleLifespanDesc);
		D3D12_INPUT_ELEMENT_DESC particleAgeDesc;
		particleAgeDesc.AlignedByteOffset = offsetof(Particle, age);
		particleAgeDesc.Format = DXGI_FORMAT_R32_FLOAT;
		particleAgeDesc.InputSlot = 0;
		particleAgeDesc.InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
		particleAgeDesc.InstanceDataStepRate = 0;
		particleAgeDesc.SemanticIndex = 0;
		particleAgeDesc.SemanticName = "AGE";
		particlesGeometry->AddInputElement(particleAgeDesc);

		com_ptr<ID3DBlob> billboardVertexShader = Egg::Shader::LoadCso("Shaders/billboardVS.cso");
		com_ptr<ID3DBlob> billboardGeometryShader = Egg::Shader::LoadCso("Shaders/billboardGS.cso");
		com_ptr<ID3DBlob> firePixelShader = Egg::Shader::LoadCso("Shaders/firePS.cso");
		com_ptr<ID3D12RootSignature> billboardRootSig = Egg::Shader::LoadRootSignature(device.Get(), billboardVertexShader.Get());

		Egg::Mesh::Material::P fireMaterial = Egg::Mesh::Material::Create();
		fireMaterial->SetRootSignature(billboardRootSig);
		fireMaterial->SetVertexShader(billboardVertexShader);
		fireMaterial->SetGeometryShader(billboardGeometryShader);
		fireMaterial->SetPixelShader(firePixelShader);
		fireMaterial->SetDepthStencilState(CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT));
		fireMaterial->SetDSVFormat(DXGI_FORMAT_D32_FLOAT);
		fireMaterial->SetSrvHeap(2, particleSrvHeap);
		fireMaterial->SetConstantBuffer(perFrameCb);
		fireMaterial->SetConstantBuffer(cb);

		D3D12_BLEND_DESC transparency;
		transparency.AlphaToCoverageEnable = false;
		transparency.IndependentBlendEnable = false;
		transparency.RenderTarget[0].BlendEnable = true;
		transparency.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
		transparency.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
		transparency.RenderTarget[0].LogicOpEnable = false;
		transparency.RenderTarget[0].LogicOp = D3D12_LOGIC_OP_NOOP;
//		transparency.RenderTarget[0].DestBlend = D3D12_BLEND_ONE;
		transparency.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
		transparency.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
		transparency.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
		transparency.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_SRC_ALPHA;
		transparency.RenderTarget[0].RenderTargetWriteMask = 0xfu;
		fireMaterial->SetBlendState(transparency);

		D3D12_DEPTH_STENCIL_DESC dd;
		dd.DepthEnable = true;
		dd.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
		dd.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
		dd.StencilEnable = false;
		fireMaterial->SetDepthStencilState(dd);

		fireBillboardSet = Egg::Mesh::Shaded::Create(
			psoManager, fireMaterial, particlesGeometry);

		///////// END

		for (int i = 0; i < 100; i++) {
			auto r = Egg::Scene::FixedRigidBody::Create();
			auto e = Egg::Scene::Entity::Create(multiMesh, r);
			r->Translate(Float3(i, 0, 0));
			entities.push_back( e );
		}

		UploadResources();
	}

	void ProcessMessage(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) override
	{
		camera->ProcessMessage(hWnd, uMsg, wParam, lParam);
	}

	void CreateSwapChainResources() override {
		__super::CreateSwapChainResources();
		camera->SetAspect(aspectRatio);
	}
};


