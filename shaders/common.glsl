// Shared struct definitions — included by all RT shaders.
// Uses scalar layout so memory layout matches the C++ structs exactly.

struct Vertex {
    vec3 pos;
    vec3 normal;
    vec2 uv;
};

struct Material {
    vec3  baseColor;
    float metallic;
    vec3  emissive;
    float roughness;
    float ior;
    int   type;      // 0=diffuse  1=metal  2=glass
    float _pad0;
    float _pad1;
};

struct InstanceData {
    uint vertexOffset;
    uint indexOffset;
    uint materialIndex;
    uint pad;
};

// Path-tracing payload (location 0).
// Produced by closesthit / miss; consumed by raygen.
struct RayPayload {
    vec3  radiance;    // direct + emissive contribution from this hit
    vec3  throughput;  // BRDF × NdotL / pdf for the NEXT bounce
    vec3  origin;      // next ray origin
    vec3  direction;   // next ray direction
    bool  done;        // no further bounces needed
    uint  seed;        // RNG state carried through the bounce chain
};

// PCG-based random number generator ----------------------------------------
uint pcgHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float randFloat(inout uint seed) {
    seed = pcgHash(seed);
    return float(seed) / 4294967295.0;
}

vec2 rand2(inout uint seed) {
    return vec2(randFloat(seed), randFloat(seed));
}
