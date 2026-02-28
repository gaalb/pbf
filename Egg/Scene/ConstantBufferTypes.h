#pragma once
#include "../Common.h"
#include "Egg/Math/math.h"
#include "PerObjectData.h"

__declspec(align(16)) struct PerObjectCb {
	Egg::Scene::PerObjectData objects[1024];
};

__declspec(align(16)) struct PerFrameCb {
	Egg::Math::Float4x4 viewProjMat;
	Egg::Math::Float4x4 rayDirTransform;
	Egg::Math::Float4 cameraPos;
	Egg::Math::Float4 lightPos;
	Egg::Math::Float4 lightPowerDensity;
	Egg::Math::Float4x4 lightViewProjTransform;
	Egg::Math::Float4 lightPos2;
	Egg::Math::Float4 lightPowerDensity2;
	Egg::Math::Float4x4 lightViewProjTransform2;
	Egg::Math::Float4 billboardSize;
};