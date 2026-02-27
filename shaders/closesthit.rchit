#version 460
#extension GL_EXT_ray_tracing          : require
#extension GL_EXT_scalar_block_layout  : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------
layout(binding = 0, set = 0) uniform accelerationStructureEXT tlas;

layout(binding = 3, set = 0, scalar) readonly buffer VertexBuf   { Vertex     vertices[]; };
layout(binding = 4, set = 0, scalar) readonly buffer IndexBuf    { uint       indices[];  };
layout(binding = 5, set = 0, scalar) readonly buffer MaterialBuf { Material   materials[];};
layout(binding = 6, set = 0, scalar) readonly buffer InstBuf     { InstanceData instances[]; };

layout(push_constant) uniform PC {
    uint maxBounces;
    uint samplesPerFrame;
} pc;

layout(location = 0) rayPayloadInEXT RayPayload payload;
layout(location = 1) rayPayloadEXT   float      shadowPayload;

hitAttributeEXT vec2 baryCoords;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const float PI     = 3.14159265358979;
const vec3  SUN_DIR   = normalize(vec3(0.5, 1.0, 0.3));
const vec3  SUN_COLOR = vec3(2.2, 2.0, 1.8);

// ---------------------------------------------------------------------------
// GGX / PBR helper functions
// ---------------------------------------------------------------------------

float D_GGX(float NdotH, float a2) {
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

float G_SchlickGGX(float NdotV, float k) {
    return NdotV / max(NdotV * (1.0 - k) + k, 1e-7);
}

float G_Smith(float NdotV, float NdotL, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return G_SchlickGGX(max(NdotV, 0.0), k)
         * G_SchlickGGX(max(NdotL, 0.0), k);
}

vec3 F_Schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Build a tangent frame around N
void buildFrame(vec3 N, out vec3 T, out vec3 B) {
    vec3 up = abs(N.z) < 0.999 ? vec3(0, 0, 1) : vec3(1, 0, 0);
    T = normalize(cross(up, N));
    B = cross(N, T);
}

vec3 toWorld(vec3 local, vec3 N, vec3 T, vec3 B) {
    return normalize(local.x * T + local.y * B + local.z * N);
}

// GGX importance sampling — returns a HALF vector in world space
vec3 sampleGGX(vec2 xi, vec3 N, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float phi      = 2.0 * PI * xi.x;
    float cosTheta = sqrt((1.0 - xi.y) / max(1.0 + (a2 - 1.0) * xi.y, 1e-7));
    float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));

    vec3 T, B;
    buildFrame(N, T, B);
    return toWorld(vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta), N, T, B);
}

// Cosine-weighted hemisphere sample — returns a world-space direction
vec3 sampleCosineHemi(vec2 xi, vec3 N) {
    float phi = 2.0 * PI * xi.x;
    float r   = sqrt(xi.y);
    vec3 T, B;
    buildFrame(N, T, B);
    return toWorld(vec3(r * cos(phi), r * sin(phi), sqrt(max(1.0 - xi.y, 0.0))), N, T, B);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
void main()
{
    uint seed = payload.seed;

    // -----------------------------------------------------------------------
    // Vertex fetch and interpolation
    // -----------------------------------------------------------------------
    uint instanceIdx = gl_InstanceCustomIndexEXT;
    InstanceData inst = instances[instanceIdx];

    uint i0 = indices[inst.indexOffset + gl_PrimitiveID * 3 + 0];
    uint i1 = indices[inst.indexOffset + gl_PrimitiveID * 3 + 1];
    uint i2 = indices[inst.indexOffset + gl_PrimitiveID * 3 + 2];

    Vertex v0 = vertices[inst.vertexOffset + i0];
    Vertex v1 = vertices[inst.vertexOffset + i1];
    Vertex v2 = vertices[inst.vertexOffset + i2];

    vec3 bary = vec3(1.0 - baryCoords.x - baryCoords.y,
                     baryCoords.x, baryCoords.y);

    vec3 localPos  = v0.pos    * bary.x + v1.pos    * bary.y + v2.pos    * bary.z;
    vec3 localNorm = v0.normal * bary.x + v1.normal * bary.y + v2.normal * bary.z;

    // Transform to world space
    // gl_ObjectToWorldEXT is mat4x3 (4 cols, 3 rows)
    vec3 worldPos  = vec3(gl_ObjectToWorldEXT * vec4(localPos, 1.0));
    // Normal: multiply by transpose(inverse(M)) = localNorm * WorldToObject
    vec3 worldNorm = normalize(localNorm * mat3(gl_WorldToObjectEXT));

    // -----------------------------------------------------------------------
    // Material
    // -----------------------------------------------------------------------
    Material mat = materials[inst.materialIndex];

    vec3 V = -normalize(gl_WorldRayDirectionEXT);

    // Ensure normal faces the incoming ray
    if (dot(worldNorm, V) < 0.0) worldNorm = -worldNorm;

    // -----------------------------------------------------------------------
    // Emissive: terminate and contribute emissive radiance directly
    // -----------------------------------------------------------------------
    if (dot(mat.emissive, mat.emissive) > 0.001) {
        payload.radiance = mat.emissive;
        payload.done     = true;
        payload.seed     = seed;
        return;
    }

    vec3 N      = worldNorm;
    vec3 hitPos = worldPos + N * 1e-3;

    // -----------------------------------------------------------------------
    // Glass (dielectric refraction / reflection)
    // -----------------------------------------------------------------------
    if (mat.type == 2) {
        float cosI = dot(V, N);
        float eta  = (cosI > 0.0) ? (1.0 / mat.ior) : mat.ior;
        vec3  refN = (cosI > 0.0) ? N : -N;

        float r0      = (1.0 - mat.ior) / (1.0 + mat.ior);
        r0           *= r0;
        float fresnel = r0 + (1.0 - r0) * pow(1.0 - abs(cosI), 5.0);

        vec3 nextDir;
        vec3 nextOrig;
        if (randFloat(seed) < fresnel) {
            // Reflect
            nextDir  = reflect(-V, N);
            nextOrig = hitPos;
        } else {
            vec3 refracted = refract(-V, refN, eta);
            if (length(refracted) < 0.001) {       // Total internal reflection
                refracted = reflect(-V, N);
                nextOrig  = hitPos;
            } else {
                nextOrig = worldPos - N * 2e-3;    // offset to the transmitted side
            }
            nextDir = normalize(refracted);
        }
        payload.radiance   = vec3(0.0);
        payload.throughput = mat.baseColor;        // tint for colored glass
        payload.origin     = nextOrig;
        payload.direction  = nextDir;
        payload.done       = false;
        payload.seed       = seed;
        return;
    }

    // -----------------------------------------------------------------------
    // Direct illumination — cast a shadow ray toward the sun
    // -----------------------------------------------------------------------
    float NdotL = max(dot(N, SUN_DIR), 0.0);
    vec3  directLight = vec3(0.0);

    if (NdotL > 0.0) {
        shadowPayload = 0.0;
        traceRayEXT(tlas,
                    gl_RayFlagsTerminateOnFirstHitEXT |
                    gl_RayFlagsSkipClosestHitShaderEXT,
                    0xFF,
                    0, 0,   // SBT offset / stride
                    1,      // miss index 1 → shadow.rmiss
                    hitPos, 1e-3, SUN_DIR, 1e4,
                    1);     // payload location 1

        float vis = shadowPayload;

        if (mat.type == 0) {
            // Lambertian diffuse: f = albedo / PI,  pdf = NdotL / PI  →  weight = albedo
            directLight = vis * SUN_COLOR * NdotL * mat.baseColor / PI;

        } else if (mat.type == 1) {
            // Cook-Torrance specular for metallic surfaces (F0 = albedo)
            vec3  H     = normalize(V + SUN_DIR);
            float NdotV = max(dot(N, V), 1e-4);
            float NdotH = max(dot(N, H), 0.0);
            float a2    = mat.roughness * mat.roughness;
            a2          = a2 * a2;

            float D = D_GGX(NdotH, a2);
            float G = G_Smith(NdotV, NdotL, mat.roughness);
            vec3  F = F_Schlick(max(dot(V, H), 0.0), mat.baseColor);

            directLight = vis * SUN_COLOR * NdotL
                        * (D * G * F) / max(4.0 * NdotV * NdotL, 1e-4);
        }
    }

    // -----------------------------------------------------------------------
    // Indirect — importance-sample the BRDF to pick the next bounce direction
    // -----------------------------------------------------------------------
    vec3 nextDir;
    vec3 brdfWeight;

    if (mat.type == 0) {
        // Diffuse: cosine-weighted hemisphere sampling
        // pdf = NdotL / PI,  f = albedo / PI  →  weight = albedo
        nextDir    = sampleCosineHemi(rand2(seed), N);
        brdfWeight = mat.baseColor;

    } else {
        // Metal: GGX specular importance sampling
        float rough = max(mat.roughness, 0.02);
        vec3  H     = sampleGGX(rand2(seed), N, rough);
        nextDir     = reflect(-V, H);

        if (dot(nextDir, N) <= 0.0) {
            // Sampled direction went below the surface — terminate this path
            payload.radiance   = directLight;
            payload.done       = true;
            payload.seed       = seed;
            return;
        }

        float NdotL2 = max(dot(N, nextDir), 1e-4);
        float NdotV  = max(dot(N, V),       1e-4);
        float NdotH  = max(dot(N, H),       0.0);
        float VdotH  = max(dot(V, H),       0.0);
        float a2     = rough * rough;
        a2           = a2 * a2;

        vec3  F = F_Schlick(VdotH, mat.baseColor);
        float G = G_Smith(NdotV, NdotL2, rough);

        // Simplification of the full GGX weight when using GGX IS:
        //   weight = F * G * VdotH / (NdotH * NdotV)
        brdfWeight = F * G * VdotH / max(NdotH * NdotV, 1e-4);
    }

    payload.radiance   = directLight;
    payload.throughput = brdfWeight;
    payload.origin     = hitPos;
    payload.direction  = nextDir;
    payload.done       = false;
    payload.seed       = seed;
}
