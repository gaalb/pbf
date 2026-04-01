#pragma once

#include "Egg/Math/Float4x4.h"

using namespace Egg::Math;

// note on __declspec(align(16)): HLSL constant buffers are read by the GPU in 16-byte chunks ("rows"),
// every field in a cbuffer must fit neatly withing these rows, i.e. an element must not straddle
// a 16-byte boundary. This means two things:
// 1: the struct must be placed in a memory address that's a multiple of 16: this is what declspec does
// 2: the struct' fields must not pass a 16-byte boundary: assured by paddings


// per-frame data sent to shaders every frame - camera matrices etc.
// must be 16-byte aligned because the GPU reads constant buffers in 16byte chunks
__declspec(align(16)) struct PerFrameCb {
	Float4x4 viewProjTransform; // combined view * projection matrix
	Float4x4 rayDirTransform; // maps screen-space positions to world-space ray directions
	Float4 cameraPos; // camera position in world space, w=1
	Float4 lightDir; // xyz = direction toward light, w = unused
	Float4 particleParams; // xyz are color, w is radius
};

// per-draw data for the solid obstacle rendering shader
__declspec(align(16)) struct SolidCb {
	Float4x4 modelMat; // object-to-world transform for solidVS
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
	float adhesion; // offset 80 (4 bytes): tangential velocity damping on wall contact (0 = frictionless, 1 = full stop)
	float pushRadius; // offset 84 (4 bytes): SDF push-out target distance (particleSpacing * pushRadiusMult, set on CPU)
	float _pad[2]; // offsets 88, 92: padding to align solidInvTransform to 16-byte boundary
	Float4x4 solidInvTransform; // offset 96 (64 bytes): world-to-object transform; updated each frame
	Float4 sdfMin; // offset 160 (16 bytes): object-space SDF AABB min (xyz = min corner, w unused)
	Float4 sdfMax; // offset 176 (16 bytes): object-space SDF AABB max (xyz = max corner, w unused)
	// total: 192 bytes
};
