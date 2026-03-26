#pragma once

#include "Egg/Math/Float3.h"

using namespace Egg::Math;

// SoA (Structure of Arrays) field definitions for the particle data.
// Each field is stored in its own GPU buffer
// The field order here matches the UAV register order (u0..u6) and descriptor heap layout.
//
// Minor optimization note: scratch and omega are never live at the same time.
// scratch is written by deltaCS and consumed by positionFromScratchCS during the solver loop,
// then reused by viscosityCS and consumed by velocityFromScratchCS after confinement.
// omega is written by vorticityCS and consumed by confinementCS, which runs between those two uses.
// They could therefore share the same field to save 12 bytes per particle.
enum ParticleField : unsigned int {
	PF_POSITION = 0, // float3: current (committed) position in world space
	PF_VELOCITY, // float3: current velocity, updated at end of each PBF step
	PF_PREDICTED_POSITION, // float3: predicted position used during constraint solving (p*)
	PF_LAMBDA, // float:  Lagrange multiplier computed in lambdaCS, read in deltaCS
	PF_DENSITY, // float:  SPH density estimate written by lambdaCS, read by rendering shaders
	PF_OMEGA, // float3: vorticity vector written by vorticityCS, read by confinementCS
	PF_SCRATCH, // float3: scratch pad buffer used by deltaCS and viscosityCS to avoid Gauss-Seidel races
	PF_COUNT // last entry in enum -> its value equals the number of actual fields
};

// Byte stride of each field per particle, matching the GPU StructuredBuffer element size.
// float3 fields are 12 bytes (tightly packed in structured buffers, unlike cbuffer alignment).
static constexpr unsigned int fieldStrides[PF_COUNT] = {
	sizeof(Float3), // PF_POSITION
	sizeof(Float3), // PF_VELOCITY
	sizeof(Float3), // PF_PREDICTED_POSITION
	sizeof(float), // PF_LAMBDA
	sizeof(float), // PF_DENSITY
	sizeof(Float3), // PF_OMEGA
	sizeof(Float3), // PF_SCRATCH
};

// Debug names for GPU resource labeling
static constexpr const wchar_t* fieldNames[PF_COUNT] = {
	L"Position", L"Velocity", L"PredictedPosition",
	L"Lambda", L"Density", L"Omega", L"Scratch"
};
