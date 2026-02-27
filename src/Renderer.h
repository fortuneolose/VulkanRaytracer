#pragma once
#include "VulkanContext.h"
#include "Scene.h"
#include "AccelStructure.h"
#include "RTPipeline.h"
#include "types.h"

#include <array>
#include <vector>

static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

class Renderer {
public:
    void init   (VulkanContext& ctx, Scene& scene,
                 AccelStructure& accel, RTPipeline& pipe);
    void drawFrame(VulkanContext& ctx, Scene& scene,
                   AccelStructure& accel, RTPipeline& pipe, float aspect);
    void destroy(VulkanContext& ctx);

private:
    AllocatedImage storageImage;

    std::array<AllocatedBuffer, MAX_FRAMES_IN_FLIGHT> cameraUBOs;
    std::array<void*,           MAX_FRAMES_IN_FLIGHT> cameraUBOMapped{};

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> descriptorSets{};

    std::array<VkCommandBuffer, MAX_FRAMES_IN_FLIGHT> commandBuffers{};
    // One semaphore per frame-in-flight for image acquisition
    std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> imageAvailableSems{};
    // One semaphore per swapchain image so the presentation engine never
    // races with a re-signal before it has finished consuming the semaphore
    std::vector<VkSemaphore> renderFinishedSems;
    std::array<VkFence, MAX_FRAMES_IN_FLIGHT> inFlightFences{};
    // Tracks which per-frame fence last rendered into each swapchain image
    std::vector<VkFence> imagesInFlight;

    uint32_t currentFrame = 0;
    uint32_t sampleCount  = 0;

    void createStorageImage  (VulkanContext& ctx);
    void createDescriptorPool(VulkanContext& ctx);
    void createDescriptorSets(VulkanContext& ctx, Scene& scene,
                              AccelStructure& accel, RTPipeline& pipe);
    void createCommandBuffers(VulkanContext& ctx);
    void createSyncObjects   (VulkanContext& ctx);

    static void imageBarrier(VkCommandBuffer cmd, VkImage image,
                             VkImageLayout oldLayout, VkImageLayout newLayout,
                             VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                             VkPipelineStageFlags srcStage,
                             VkPipelineStageFlags dstStage);
};
