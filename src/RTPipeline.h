#pragma once
#include "VulkanContext.h"
#include "types.h"
#include <string>

class RTPipeline {
public:
    VkPipeline            pipeline            = VK_NULL_HANDLE;
    VkPipelineLayout      pipelineLayout      = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;

    // Shader Binding Table regions — passed directly to vkCmdTraceRaysKHR
    AllocatedBuffer                sbtBuffer;
    VkStridedDeviceAddressRegionKHR rgenRegion{};
    VkStridedDeviceAddressRegionKHR missRegion{};
    VkStridedDeviceAddressRegionKHR hitRegion{};
    VkStridedDeviceAddressRegionKHR callRegion{};

    // shaderDir must end with a path separator ('/')
    void build  (VulkanContext& ctx, const std::string& shaderDir);
    void destroy(VulkanContext& ctx);

private:
    // Shader groups index: 0=rgen  1=miss(sky)  2=miss(shadow)  3=hitGroup
    static constexpr uint32_t GROUP_RGEN        = 0;
    static constexpr uint32_t GROUP_MISS_SKY    = 1;
    static constexpr uint32_t GROUP_MISS_SHADOW = 2;
    static constexpr uint32_t GROUP_HIT         = 3;
    static constexpr uint32_t GROUP_COUNT       = 4;

    void buildSBT(VulkanContext& ctx);

    static uint32_t alignUp(uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); }
};
