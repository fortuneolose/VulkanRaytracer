#version 460
#extension GL_EXT_ray_tracing : require

// Shadow payload at location 1.
// Initialised to 0.0 (occluded) before the trace call.
// This shader fires when the shadow ray reaches the sky (not blocked).
layout(location = 1) rayPayloadInEXT float shadowPayload;

void main()
{
    shadowPayload = 1.0; // visible — not in shadow
}
