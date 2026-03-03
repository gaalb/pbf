#pragma once

#include "Egg/Math/Float4x4.h"

using namespace Egg::Math;

// per-frame data sent to shaders every frame - camera matrices etc.
// must be 16-byte aligned because the GPU reads constant buffers in 16byte chunks
__declspec(align(16)) struct PerFrameCb {
	Float4x4 viewProjTransform; // combined view * projection matrix
	Float4x4 rayDirTransform; // maps screen-space positions to world-space ray directions
	Float4 cameraPos; // camera position in world space, w=1
	Float4 lightDir; // xyz = direction toward light, w = unused
	Float4 particleParams; // xyz are color, w is radius
};

// per-simulation-step data uploaded to the gravity compute shader before each dispatch
__declspec(align(16)) struct ComputeCb {
	float dt;           // simulation timestep in seconds, set each frame in Update()
	UINT numParticles;  // total particle count; used for bounds check in the compute shader
	float pad[2];       // padding to reach 16 bytes — GPU constant buffers are read in 16-byte chunks
};