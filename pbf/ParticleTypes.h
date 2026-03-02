#pragma once

#include "Egg/Math/Float3.h"

using namespace Egg::Math;

// element type for the particle structured buffer on the GPU
// layout must exactly match the Particle struct in Particle.hlsli
struct Particle {
	Float3 position; // current position in world space
	Float3 velocity; // current velocity in world space
};
