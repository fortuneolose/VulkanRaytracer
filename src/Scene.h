#pragma once
#include "types.h"
#include "VulkanContext.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <GLFW/glfw3.h>

#include <vector>

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

class Camera {
public:
    glm::vec3 position{0.0f, 1.5f, 5.0f};
    glm::vec3 target  {0.0f, 0.0f, 0.0f};
    glm::vec3 up      {0.0f, 1.0f, 0.0f};
    float     fov     = 60.0f;
    bool      moved   = true;   // triggers accumulation reset

    glm::mat4 getView()        const;
    glm::mat4 getProj(float aspect) const;

    void processInput(GLFWwindow* window, float dt);

private:
    float  speed       = 3.0f;
    float  sensitivity = 0.15f;
    float  yaw         = -90.0f;
    float  pitch       = 0.0f;
    double lastMouseX  = 0.0;
    double lastMouseY  = 0.0;
    bool   firstMouse  = true;
};

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

struct MeshData {
    std::vector<Vertex>   vertices;
    std::vector<uint32_t> indices;
    uint32_t              materialIndex = 0;
};

struct SceneInstance {
    uint32_t  meshIndex;
    glm::mat4 transform;
    uint32_t  materialIndex;
};

class Scene {
public:
    std::vector<MeshData>      meshes;
    std::vector<SceneInstance> instances;
    std::vector<Material>      materials;
    Camera                     camera;

    // GPU-side resources (filled by uploadToGPU)
    AllocatedBuffer vertexBuffer;
    AllocatedBuffer indexBuffer;
    AllocatedBuffer materialBuffer;
    AllocatedBuffer instanceDataBuffer;

    void buildScene();
    void uploadToGPU(VulkanContext& ctx);
    void destroy(VulkanContext& ctx);

private:
    // Geometry helpers
    void addSphere(const glm::vec3& center, float radius,
                   uint32_t materialIdx, int stacks = 16, int slices = 32);
    void addPlane(const glm::vec3& center, float halfW, float halfD,
                  uint32_t materialIdx);

    // Upload a CPU buffer to a GPU buffer via a staging buffer
    AllocatedBuffer upload(VulkanContext& ctx, const void* data, VkDeviceSize size,
                           VkBufferUsageFlags usage);
};
