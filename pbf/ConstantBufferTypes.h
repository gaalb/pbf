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

// per-simulation-step data uploaded to all PBF compute shaders before each dispatch
// must be 16-byte aligned; total size = 32 bytes (two 16-byte chunks)
__declspec(align(16)) struct ComputeCb {
	float dt; // simulation timestep in seconds, set each frame in Update()
	UINT  numParticles; // total particle count; used for bounds check in every compute shader
	float h; // SPH smoothing radius — particles within this distance are neighbors
	float rho0; // rest density (kg/m^3); constraint target: rho_i / rho0 - 1 = 0

	float epsilon; // constraint force mixing relaxation parameter
	float pad[3]; // padding to reach 32 bytes (next 16-byte boundary)
};