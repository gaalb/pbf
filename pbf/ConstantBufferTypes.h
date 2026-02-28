#pragma once

#include "Egg/Math/Float4x4.h"

using namespace Egg::Math;

// per-frame data sent to shaders every frame - camera matrices etc.
// must be 16-byte aligned because the GPU reads constant buffers in 16byte chunks
__declspec(align(16)) struct PerFrameCb {
	Float4x4 viewProjMat; // combined view * projection matrix
	Float4 cameraPos; // camera position in world space
};