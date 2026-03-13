#pragma once

#include "Egg/Math/Float3.h"

using namespace Egg::Math;

// element type for the particle structured buffer on the GPU
// layout must exactly match the Particle struct in Particle.hlsli
// note: Float3 = 3 floats = 12 bytes; no hidden padding between fields in a structured buffer
//
// Optimization note: predictedPosition and omega are never live at the same time.
// predictedPosition is written by predictCS and consumed by finalizeCS; omega is written
// by vorticityCS and consumed by confinementCS, which runs after finalizeCS is done.
// They could therefore share the same field to save 12 bytes per particle.
struct Particle {
	Float3 position;          // current (committed) position in world space
	Float3 velocity;          // current velocity, updated at end of each PBF step
	Float3 predictedPosition; // predicted position used during constraint solving (p*)
	float  lambda;            // Lagrange multiplier computed in lambdaCS, read in deltaCS
	Float3 omega;             // vorticity vector written by vorticityCS, read by confinementCS
};
