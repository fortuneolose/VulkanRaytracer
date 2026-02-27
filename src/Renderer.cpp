#include "Renderer.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <array>
#include <stdexcept>
#include <cstring>

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

void Renderer::init(VulkanContext& ctx, Scene& scene,
                    AccelStructure& accel, RTPipeline& pipe)
{
    createStorageImage(ctx);
    createDescriptorPool(ctx);
    createDescriptorSets(ctx, scene, accel, pipe);
    createCommandBuffers(ctx);
    createSyncObjects(ctx);

    // Transition storage image to GENERAL layout for shader read/write
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    imageBarrier(cmd, storageImage.image,
        VK_IMAGE_LAYOUT_UNDEFINED,       VK_IMAGE_LAYOUT_GENERAL,
        0,                               VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);
    ctx.endSingleTimeCommands(cmd);
}

// ---------------------------------------------------------------------------
// createStorageImage
// ---------------------------------------------------------------------------

void Renderer::createStorageImage(VulkanContext& ctx)
{
    storageImage = ctx.createImage(
        ctx.swapchainExtent.width,
        ctx.swapchainExtent.height,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
}

// ---------------------------------------------------------------------------
// createDescriptorPool
// ---------------------------------------------------------------------------

void Renderer::createDescriptorPool(VulkanContext& ctx)
{
    std::array<VkDescriptorPoolSize, 4> poolSizes{{
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, MAX_FRAMES_IN_FLIGHT},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,              MAX_FRAMES_IN_FLIGHT},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,             MAX_FRAMES_IN_FLIGHT},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  4 * MAX_FRAMES_IN_FLIGHT},
    }};

    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    pi.pPoolSizes    = poolSizes.data();
    pi.maxSets       = MAX_FRAMES_IN_FLIGHT;
    vkCreateDescriptorPool(ctx.device, &pi, nullptr, &descriptorPool);
}

// ---------------------------------------------------------------------------
// createDescriptorSets
// ---------------------------------------------------------------------------

void Renderer::createDescriptorSets(VulkanContext& ctx, Scene& scene,
                                     AccelStructure& accel, RTPipeline& pipe)
{
    // Camera UBOs (persistently mapped, updated each frame)
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        cameraUBOs[i] = ctx.createBuffer(
            sizeof(CameraUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);

        VmaAllocationInfo ai{};
        vmaGetAllocationInfo(ctx.allocator, cameraUBOs[i].allocation, &ai);
        cameraUBOMapped[i] = ai.pMappedData;
    }

    // Allocate descriptor sets
    std::array<VkDescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts;
    layouts.fill(pipe.descriptorSetLayout);

    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool     = descriptorPool;
    ai.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    ai.pSetLayouts        = layouts.data();
    vkAllocateDescriptorSets(ctx.device, &ai, descriptorSets.data());

    // Write descriptors for each in-flight frame
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        // Binding 0: TLAS
        VkWriteDescriptorSetAccelerationStructureKHR tlasInfo{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
        tlasInfo.accelerationStructureCount = 1;
        tlasInfo.pAccelerationStructures    = &accel.tlas;

        // Binding 1: storage image
        VkDescriptorImageInfo imgInfo{};
        imgInfo.imageView   = storageImage.view;
        imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        // Binding 2: camera UBO
        VkDescriptorBufferInfo camInfo{cameraUBOs[i].buffer, 0, sizeof(CameraUBO)};

        // Bindings 3-6: geometry / material buffers
        VkDescriptorBufferInfo vtxInfo {scene.vertexBuffer.buffer,       0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo idxInfo {scene.indexBuffer.buffer,        0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo matInfo {scene.materialBuffer.buffer,     0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo instInfo{scene.instanceDataBuffer.buffer, 0, VK_WHOLE_SIZE};

        std::array<VkWriteDescriptorSet, 7> writes{};

        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[0].pNext           = &tlasInfo;
        writes[0].dstSet          = descriptorSets[i];
        writes[0].dstBinding      = 0;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        writes[0].descriptorCount = 1;

        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[1].dstSet          = descriptorSets[i];
        writes[1].dstBinding      = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo      = &imgInfo;

        writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[2].dstSet          = descriptorSets[i];
        writes[2].dstBinding      = 2;
        writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[2].descriptorCount = 1;
        writes[2].pBufferInfo     = &camInfo;

        auto makeSsbo = [&](int binding, VkDescriptorBufferInfo* info) {
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet          = descriptorSets[i];
            w.dstBinding      = static_cast<uint32_t>(binding);
            w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.descriptorCount = 1;
            w.pBufferInfo     = info;
            return w;
        };
        writes[3] = makeSsbo(3, &vtxInfo);
        writes[4] = makeSsbo(4, &idxInfo);
        writes[5] = makeSsbo(5, &matInfo);
        writes[6] = makeSsbo(6, &instInfo);

        vkUpdateDescriptorSets(ctx.device,
            static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

// ---------------------------------------------------------------------------
// createCommandBuffers / createSyncObjects
// ---------------------------------------------------------------------------

void Renderer::createCommandBuffers(VulkanContext& ctx)
{
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool        = ctx.commandPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
    vkAllocateCommandBuffers(ctx.device, &ai, commandBuffers.data());
}

void Renderer::createSyncObjects(VulkanContext& ctx)
{
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo     fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vkCreateSemaphore(ctx.device, &si, nullptr, &imageAvailableSems[i]);
        vkCreateFence    (ctx.device, &fi, nullptr, &inFlightFences[i]);
    }

    // One render-finished semaphore per swapchain image so the presentation
    // engine's consume of the semaphore cannot race with a re-signal from a
    // subsequent frame that happens to land on the same frame-in-flight slot.
    uint32_t imgCount = static_cast<uint32_t>(ctx.swapchainImages.size());
    renderFinishedSems.resize(imgCount);
    imagesInFlight.assign(imgCount, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < imgCount; ++i)
        vkCreateSemaphore(ctx.device, &si, nullptr, &renderFinishedSems[i]);
}

// ---------------------------------------------------------------------------
// imageBarrier helper
// ---------------------------------------------------------------------------

void Renderer::imageBarrier(VkCommandBuffer cmd, VkImage image,
                              VkImageLayout oldLayout, VkImageLayout newLayout,
                              VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                              VkPipelineStageFlags srcStage,
                              VkPipelineStageFlags dstStage)
{
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout           = oldLayout;
    b.newLayout           = newLayout;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image               = image;
    b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.srcAccessMask       = srcAccess;
    b.dstAccessMask       = dstAccess;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0,
                         0, nullptr, 0, nullptr, 1, &b);
}

// ---------------------------------------------------------------------------
// drawFrame
// ---------------------------------------------------------------------------

void Renderer::drawFrame(VulkanContext& ctx, Scene& scene,
                          AccelStructure& /*accel*/, RTPipeline& pipe,
                          float aspect)
{
    int f = static_cast<int>(currentFrame);

    vkWaitForFences(ctx.device, 1, &inFlightFences[f], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    VkResult res = vkAcquireNextImageKHR(ctx.device, ctx.swapchain,
                                          UINT64_MAX, imageAvailableSems[f],
                                          VK_NULL_HANDLE, &imageIndex);
    if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR) return;

    // If a previous frame is still using this swapchain image, wait for it.
    // This prevents re-signalling renderFinishedSems[imageIndex] while the
    // presentation engine may still be consuming it from the previous present.
    if (imagesInFlight[imageIndex] != VK_NULL_HANDLE)
        vkWaitForFences(ctx.device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
    imagesInFlight[imageIndex] = inFlightFences[f];

    vkResetFences(ctx.device, 1, &inFlightFences[f]);

    // ---- Update camera UBO ------------------------------------------------
    CameraUBO cam{};
    cam.invView     = glm::inverse(scene.camera.getView());
    cam.invProj     = glm::inverse(scene.camera.getProj(aspect));
    cam.sampleCount = sampleCount;
    cam.frameIndex  = currentFrame;
    std::memcpy(cameraUBOMapped[f], &cam, sizeof(CameraUBO));

    if (scene.camera.moved)
        sampleCount = 0;
    else
        ++sampleCount;

    // ---- Record command buffer --------------------------------------------
    VkCommandBuffer cmd = commandBuffers[f];
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &bi);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipe.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
        pipe.pipelineLayout, 0, 1, &descriptorSets[f], 0, nullptr);

    PushConstants pc{4, 1}; // 4 max bounces, 1 sample per frame
    vkCmdPushConstants(cmd, pipe.pipelineLayout,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
        0, sizeof(PushConstants), &pc);

    // Trace rays into the storage image
    ctx.rt.cmdTraceRays(cmd,
        &pipe.rgenRegion, &pipe.missRegion,
        &pipe.hitRegion,  &pipe.callRegion,
        ctx.swapchainExtent.width,
        ctx.swapchainExtent.height,
        1);

    // ---- Copy storage image → swapchain image ----------------------------
    VkImage swapImg = ctx.swapchainImages[imageIndex];

    imageBarrier(cmd, storageImage.image,
        VK_IMAGE_LAYOUT_GENERAL,              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_ACCESS_SHADER_WRITE_BIT,           VK_ACCESS_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT);

    imageBarrier(cmd, swapImg,
        VK_IMAGE_LAYOUT_UNDEFINED,            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        0,                                    VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,    VK_PIPELINE_STAGE_TRANSFER_BIT);

    // Blit with NEAREST (storage image and swapchain are the same resolution)
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1]  = {(int32_t)ctx.swapchainExtent.width,
                           (int32_t)ctx.swapchainExtent.height, 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[1]  = {(int32_t)ctx.swapchainExtent.width,
                           (int32_t)ctx.swapchainExtent.height, 1};

    vkCmdBlitImage(cmd,
        storageImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        swapImg,            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &blit, VK_FILTER_NEAREST);

    // Restore storage image to GENERAL for the next frame
    imageBarrier(cmd, storageImage.image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_ACCESS_TRANSFER_READ_BIT,          VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);

    imageBarrier(cmd, swapImg,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_ACCESS_TRANSFER_WRITE_BIT,         0,
        VK_PIPELINE_STAGE_TRANSFER_BIT,       VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

    vkEndCommandBuffer(cmd);

    // ---- Submit -----------------------------------------------------------
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount   = 1;
    si.pWaitSemaphores      = &imageAvailableSems[f];
    si.pWaitDstStageMask    = &waitStage;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &renderFinishedSems[imageIndex];
    vkQueueSubmit(ctx.graphicsQueue, 1, &si, inFlightFences[f]);

    // ---- Present ----------------------------------------------------------
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &renderFinishedSems[imageIndex];
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &ctx.swapchain;
    pi.pImageIndices      = &imageIndex;
    vkQueuePresentKHR(ctx.graphicsQueue, &pi);

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

// ---------------------------------------------------------------------------
// destroy
// ---------------------------------------------------------------------------

void Renderer::destroy(VulkanContext& ctx)
{
    vkDeviceWaitIdle(ctx.device);

    ctx.destroyImage(storageImage);

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        ctx.destroyBuffer(cameraUBOs[i]);
        vkDestroySemaphore(ctx.device, imageAvailableSems[i], nullptr);
        vkDestroyFence    (ctx.device, inFlightFences[i],     nullptr);
    }
    for (VkSemaphore sem : renderFinishedSems)
        vkDestroySemaphore(ctx.device, sem, nullptr);
    vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);
}
