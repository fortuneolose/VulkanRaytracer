#pragma once
#include <glm/glm.hpp>
#include <cstdint>

// Vertex layout: pos(12) + normal(12) + uv(8) = 32 bytes
// Must match the GLSL struct in common.glsl (scalar layout).
struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;
};

// Material layout (scalar, 48 bytes).
// type: 0 = diffuse, 1 = metal, 2 = glass
struct Material {
    glm::vec3 baseColor;
    float     metallic;
    glm::vec3 emissive;
    float     roughness;
    float     ior;
    int       type;
    float     _pad[2];
};

// Per-mesh data uploaded to the GPU so the closest-hit shader can look up
// vertex/index data and material by instanceCustomIndex.
struct InstanceData {
    uint32_t vertexOffset;   // first vertex in the global vertex buffer
    uint32_t indexOffset;    // first index  in the global index  buffer
    uint32_t materialIndex;
    uint32_t pad;
};

// Camera matrices updated every frame.
struct CameraUBO {
    glm::mat4 invView;
    glm::mat4 invProj;
    uint32_t  sampleCount;   // accumulation counter (0 = first frame after reset)
    uint32_t  frameIndex;
    float     _pad[2];
};

// Small push-constant block (available in both raygen and closest-hit).
struct PushConstants {
    uint32_t maxBounces;
    uint32_t samplesPerFrame;
};
