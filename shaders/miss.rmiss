#version 460
#extension GL_EXT_ray_tracing          : require
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) rayPayloadInEXT RayPayload payload;

void main()
{
    vec3 dir = normalize(gl_WorldRayDirectionEXT);

    // Sky gradient: horizon is light blue, zenith is deeper blue
    float t   = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    vec3  sky = mix(vec3(0.6, 0.75, 0.95), vec3(0.1, 0.3, 0.7), t);

    // Simple sun disc
    const vec3 sunDir   = normalize(vec3(0.5, 1.0, 0.3));
    float sunDot = max(dot(dir, sunDir), 0.0);
    sky += pow(sunDot, 128.0) * vec3(3.5, 3.0, 2.5);   // bright core
    sky += pow(sunDot,  16.0) * vec3(0.5, 0.4, 0.3);   // warm corona

    payload.radiance = sky;
    payload.done     = true;
}
