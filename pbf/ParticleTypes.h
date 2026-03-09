#pragma once

#include "Egg/Math/Float3.h"

using namespace Egg::Math;

// element type for the particle structured buffer on the GPU
// layout must exactly match the Particle struct in Particle.hlsli
// note: Float3 = 3 floats = 12 bytes; no hidden padding between fields in a structured buffer
struct Particle {
	Float3 position;          // current (committed) position in world space
	Float3 velocity;          // current velocity, updated at end of each PBF step
	Float3 predictedPosition; // predicted position used during constraint solving (p*)
	float  lambda;            // Lagrange multiplier computed in lambdaCS, read in deltaCS
};
