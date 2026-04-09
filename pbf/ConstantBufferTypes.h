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

// Per-simulation-step data uploaded to all PBF compute shaders before each dispatch.
// Fields that are compile-time constants (h, rho0, gridMin/Max, kernel coefficients,
// push radius, sCorrDeltaQ/N) have been removed and are now #defines in SharedConfig.hlsli.
// Only runtime-variable fields remain here.
// Must be 16-byte aligned; fields ordered to avoid straddling 16-byte boundaries.
__declspec(align(16)) struct ComputeCb {
	float dt;               // offset  0: simulation timestep in seconds
	UINT numParticles;      // offset  4: total particle count; bounds check in every shader
	float sCorrK;           // offset  8: artificial pressure magnitude coefficient
	float vorticityEpsilon; // offset 12: vorticity confinement strength
	Float3 boxMin;          // offset 16: simulation box minimum corner (adjustable, world space)
	float epsilon;          // offset 28: constraint force mixing relaxation
	Float3 boxMax;          // offset 32: simulation box maximum corner (adjustable, world space)
	float viscosity;        // offset 44: XSPH viscosity coefficient
	Float3 externalForce;   // offset 48: horizontal acceleration from arrow keys (m/s^2)
	UINT fountainEnabled;   // offset 60: 1 = fountain jet active, 0 = off
	float adhesion;         // offset 64: tangential velocity damping on wall contact
	float _pad0;            // offset 68
	float _pad1;            // offset 72
	float _pad2;            // offset 76
	Float4x4 solidInvTransform; // offset  80: world-to-object transform for SDF sampling
	Float4 sdfMin;          // offset 144: object-space SDF AABB min (xyz = min corner, w unused)
	Float4 sdfMax;          // offset 160: object-space SDF AABB max (xyz = max corner, w unused)
	// total: 176 bytes
};
