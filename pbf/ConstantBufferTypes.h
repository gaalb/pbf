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
// must be 16-byte aligned, hence the not-so-didactic ordering under here, that avoids padding
__declspec(align(16)) struct ComputeCb {
	float dt; // offset 0 (4 bytes): simulation timestep in seconds
	UINT numParticles; // offset 4 (4 bytes): total particle count; used for bounds check in every compute shader
	float h; // offset 8 (4 bytes): SPH smoothing radius
	float rho0; // offset 12 (4 bytes): rest density, constraint target: rho_i / rho0 - 1 = 0
	Float3 boxMin; // offset 16 (12 bytes): simulation box minimum corner (world space)
	float epsilon; // offset 28 (4 bytes): constraint force mixing relaxation
	Float3 boxMax; // offset 32 (12 bytes): simulation box maximum corner (world space)
	float viscosity; // offset 44 (4 bytes): XSPH viscosity coefficient c
	float sCorrK; // offset 48 (4 bytes): artificial pressure k
	float sCorrDeltaQ; // offset 52 (4 bytes): artificial pressure deltaq
	float sCorrN; // offset 56 (4 bytes): artificial pressure n
	float vorticityEpsilon; // offset 60 (4 bytes): vorticity confinement strength coefficient
	Float3 externalForce; // offset 64 (12 bytes): horizontal force from arrow keys (acceleration, m/s^2)
	UINT fountainEnabled; // offset 76 (4 bytes): 1 = fountain jet active, 0 = off
};