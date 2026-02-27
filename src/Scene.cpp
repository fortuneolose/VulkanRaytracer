#include "Scene.h"

#include <glm/gtc/constants.hpp>
#include <cmath>
#include <cstring>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

glm::mat4 Camera::getView() const
{
    return glm::lookAt(position, target, up);
}

glm::mat4 Camera::getProj(float aspect) const
{
    // Standard GLM perspective — Y NOT flipped here; the raygen shader
    // flips NDC-y to account for Vulkan's inverted Y axis.
    return glm::perspective(glm::radians(fov), aspect, 0.01f, 1000.0f);
}

void Camera::processInput(GLFWwindow* window, float dt)
{
    moved = false;

    glm::vec3 forward = glm::normalize(target - position);
    glm::vec3 right   = glm::normalize(glm::cross(forward, up));

    auto key = [&](int k) { return glfwGetKey(window, k) == GLFW_PRESS; };

    if (key(GLFW_KEY_W)) { position += forward * speed * dt; moved = true; }
    if (key(GLFW_KEY_S)) { position -= forward * speed * dt; moved = true; }
    if (key(GLFW_KEY_A)) { position -= right   * speed * dt; moved = true; }
    if (key(GLFW_KEY_D)) { position += right   * speed * dt; moved = true; }
    if (key(GLFW_KEY_E)) { position += up      * speed * dt; moved = true; }
    if (key(GLFW_KEY_Q)) { position -= up      * speed * dt; moved = true; }

    // Right-click mouse-look
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
        double mx, my;
        glfwGetCursorPos(window, &mx, &my);

        if (firstMouse) { lastMouseX = mx; lastMouseY = my; firstMouse = false; }

        double dx = (mx - lastMouseX) * sensitivity;
        double dy = (lastMouseY - my) * sensitivity;
        lastMouseX = mx;
        lastMouseY = my;

        if (dx != 0.0 || dy != 0.0) {
            yaw   += static_cast<float>(dx);
            pitch += static_cast<float>(dy);
            pitch  = glm::clamp(pitch, -89.0f, 89.0f);

            glm::vec3 dir;
            dir.x  = std::cos(glm::radians(yaw)) * std::cos(glm::radians(pitch));
            dir.y  = std::sin(glm::radians(pitch));
            dir.z  = std::sin(glm::radians(yaw)) * std::cos(glm::radians(pitch));
            target = position + glm::normalize(dir);
            moved  = true;
        }
    } else {
        firstMouse = true;
    }

    if (moved) target = position + glm::normalize(target - position);
}

// ---------------------------------------------------------------------------
// Scene geometry
// ---------------------------------------------------------------------------

void Scene::addSphere(const glm::vec3& center, float radius,
                      uint32_t materialIdx, int stacks, int slices)
{
    MeshData mesh;
    mesh.materialIndex = materialIdx;

    for (int i = 0; i <= stacks; ++i) {
        float phi = glm::pi<float>() * i / stacks;
        for (int j = 0; j <= slices; ++j) {
            float theta = 2.0f * glm::pi<float>() * j / slices;
            glm::vec3 n{
                std::sin(phi) * std::cos(theta),
                std::cos(phi),
                std::sin(phi) * std::sin(theta)
            };
            Vertex v;
            v.pos    = center + n * radius;
            v.normal = n;
            v.uv     = {static_cast<float>(j) / slices,
                        static_cast<float>(i) / stacks};
            mesh.vertices.push_back(v);
        }
    }

    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            int a = i * (slices + 1) + j;
            int b = a + slices + 1;
            mesh.indices.push_back(a);     mesh.indices.push_back(b);     mesh.indices.push_back(a + 1);
            mesh.indices.push_back(b);     mesh.indices.push_back(b + 1); mesh.indices.push_back(a + 1);
        }
    }

    uint32_t meshIdx = static_cast<uint32_t>(meshes.size());
    meshes.push_back(std::move(mesh));

    SceneInstance inst;
    inst.meshIndex     = meshIdx;
    inst.transform     = glm::mat4(1.0f);
    inst.materialIndex = materialIdx;
    instances.push_back(inst);
}

void Scene::addPlane(const glm::vec3& center, float halfW, float halfD,
                     uint32_t materialIdx)
{
    MeshData mesh;
    mesh.materialIndex = materialIdx;

    glm::vec3 n{0.0f, 1.0f, 0.0f};
    mesh.vertices = {
        { center + glm::vec3(-halfW, 0.0f, -halfD), n, {0.0f, 0.0f} },
        { center + glm::vec3( halfW, 0.0f, -halfD), n, {1.0f, 0.0f} },
        { center + glm::vec3( halfW, 0.0f,  halfD), n, {1.0f, 1.0f} },
        { center + glm::vec3(-halfW, 0.0f,  halfD), n, {0.0f, 1.0f} },
    };
    mesh.indices = {0, 1, 2, 0, 2, 3};

    uint32_t meshIdx = static_cast<uint32_t>(meshes.size());
    meshes.push_back(std::move(mesh));

    SceneInstance inst;
    inst.meshIndex     = meshIdx;
    inst.transform     = glm::mat4(1.0f);
    inst.materialIndex = materialIdx;
    instances.push_back(inst);
}

// ---------------------------------------------------------------------------
// buildScene — geometry + material definitions
// ---------------------------------------------------------------------------

void Scene::buildScene()
{
    // Materials
    // 0: white diffuse floor
    materials.push_back({{0.8f, 0.8f, 0.8f}, 0.0f, {0,0,0}, 0.95f, 1.5f, 0, {0,0}});
    // 1: red diffuse
    materials.push_back({{0.8f, 0.15f, 0.1f}, 0.0f, {0,0,0}, 0.9f, 1.5f, 0, {0,0}});
    // 2: gold metal
    materials.push_back({{1.0f, 0.78f, 0.2f}, 1.0f, {0,0,0}, 0.1f, 1.5f, 1, {0,0}});
    // 3: glass
    materials.push_back({{0.95f, 0.98f, 1.0f}, 0.0f, {0,0,0}, 0.0f, 1.5f, 2, {0,0}});
    // 4: emissive area light (warm white)
    materials.push_back({{1.0f, 0.9f, 0.8f}, 0.0f, {6.0f, 5.0f, 4.5f}, 0.9f, 1.5f, 0, {0,0}});
    // 5: blue diffuse
    materials.push_back({{0.2f, 0.3f, 0.9f}, 0.0f, {0,0,0}, 0.85f, 1.5f, 0, {0,0}});

    // Geometry
    addPlane ({0.0f, -1.0f,  0.0f}, 6.0f, 6.0f, 0); // floor
    addSphere({-2.0f, 0.0f,  0.0f}, 1.0f, 1);        // red diffuse
    addSphere({ 0.0f, 0.0f,  0.0f}, 1.0f, 2);        // gold metal
    addSphere({ 2.0f, 0.0f,  0.0f}, 1.0f, 3);        // glass
    addSphere({-2.0f, 0.0f, -3.0f}, 1.0f, 5);        // blue diffuse
    addSphere({ 0.0f, 4.5f,  0.0f}, 0.6f, 4);        // area light
}

// ---------------------------------------------------------------------------
// uploadToGPU
// ---------------------------------------------------------------------------

AllocatedBuffer Scene::upload(VulkanContext& ctx,
                              const void* data, VkDeviceSize size,
                              VkBufferUsageFlags usage)
{
    // Staging buffer (CPU visible)
    AllocatedBuffer staging = ctx.createBuffer(
        size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

    void* mapped;
    vmaMapMemory(ctx.allocator, staging.allocation, &mapped);
    std::memcpy(mapped, data, size);
    vmaUnmapMemory(ctx.allocator, staging.allocation);

    // GPU buffer
    AllocatedBuffer gpu = ctx.createBuffer(
        size,
        usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(cmd, staging.buffer, gpu.buffer, 1, &region);
    ctx.endSingleTimeCommands(cmd);

    ctx.destroyBuffer(staging);
    return gpu;
}

void Scene::uploadToGPU(VulkanContext& ctx)
{
    // Flatten all meshes into contiguous vertex / index arrays
    std::vector<Vertex>       allVerts;
    std::vector<uint32_t>     allIndices;
    std::vector<InstanceData> instData;

    for (size_t i = 0; i < meshes.size(); ++i) {
        InstanceData id{};
        id.vertexOffset   = static_cast<uint32_t>(allVerts.size());
        id.indexOffset    = static_cast<uint32_t>(allIndices.size());
        id.materialIndex  = meshes[i].materialIndex;
        instData.push_back(id);

        for (auto& v : meshes[i].vertices) allVerts.push_back(v);
        for (auto& ix : meshes[i].indices)  allIndices.push_back(ix);
    }

    constexpr VkBufferUsageFlags geoFlags =
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    vertexBuffer = upload(ctx, allVerts.data(),
        allVerts.size() * sizeof(Vertex),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | geoFlags);

    indexBuffer = upload(ctx, allIndices.data(),
        allIndices.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | geoFlags);

    materialBuffer = upload(ctx, materials.data(),
        materials.size() * sizeof(Material),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    instanceDataBuffer = upload(ctx, instData.data(),
        instData.size() * sizeof(InstanceData),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
}

// ---------------------------------------------------------------------------
// destroy
// ---------------------------------------------------------------------------

void Scene::destroy(VulkanContext& ctx)
{
    ctx.destroyBuffer(vertexBuffer);
    ctx.destroyBuffer(indexBuffer);
    ctx.destroyBuffer(materialBuffer);
    ctx.destroyBuffer(instanceDataBuffer);
}
