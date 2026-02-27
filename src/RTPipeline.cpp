#include "RTPipeline.h"

#include <array>
#include <cstring>
#include <stdexcept>
#include <iostream>

// ---------------------------------------------------------------------------
// build
// ---------------------------------------------------------------------------

void RTPipeline::build(VulkanContext& ctx, const std::string& shaderDir)
{
    // -----------------------------------------------------------------------
    // Descriptor set layout
    //  Binding 0  ACCELERATION_STRUCTURE  — TLAS
    //  Binding 1  STORAGE_IMAGE           — rgba32f accumulation image
    //  Binding 2  UNIFORM_BUFFER          — CameraUBO
    //  Binding 3  STORAGE_BUFFER          — vertex buffer
    //  Binding 4  STORAGE_BUFFER          — index buffer
    //  Binding 5  STORAGE_BUFFER          — material buffer
    //  Binding 6  STORAGE_BUFFER          — per-instance data
    // -----------------------------------------------------------------------
    const VkShaderStageFlags rtAll = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                     VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                     VK_SHADER_STAGE_MISS_BIT_KHR;
    const VkShaderStageFlags hitOnly = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    const VkShaderStageFlags rgenOnly = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    std::array<VkDescriptorSetLayoutBinding, 7> bindings{{
        {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, rtAll,    nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,              1, rgenOnly, nullptr},
        {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,             1, rgenOnly, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             1, hitOnly,  nullptr},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             1, hitOnly,  nullptr},
        {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             1, hitOnly,  nullptr},
        {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             1, hitOnly,  nullptr},
    }};

    VkDescriptorSetLayoutCreateInfo dslCI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslCI.bindingCount = static_cast<uint32_t>(bindings.size());
    dslCI.pBindings    = bindings.data();
    vkCreateDescriptorSetLayout(ctx.device, &dslCI, nullptr, &descriptorSetLayout);

    // -----------------------------------------------------------------------
    // Pipeline layout — push constants available in raygen + closesthit
    // -----------------------------------------------------------------------
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    pcRange.size       = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo layoutCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutCI.setLayoutCount         = 1;
    layoutCI.pSetLayouts            = &descriptorSetLayout;
    layoutCI.pushConstantRangeCount = 1;
    layoutCI.pPushConstantRanges    = &pcRange;
    vkCreatePipelineLayout(ctx.device, &layoutCI, nullptr, &pipelineLayout);

    // -----------------------------------------------------------------------
    // Shader stages
    //  Stage index: 0=rgen  1=miss(sky)  2=miss(shadow)  3=chit
    // -----------------------------------------------------------------------
    VkShaderModule rgenMod   = ctx.loadShaderModule(shaderDir + "raygen.rgen.spv");
    VkShaderModule missMod   = ctx.loadShaderModule(shaderDir + "miss.rmiss.spv");
    VkShaderModule shadowMod = ctx.loadShaderModule(shaderDir + "shadow.rmiss.spv");
    VkShaderModule chitMod   = ctx.loadShaderModule(shaderDir + "closesthit.rchit.spv");

    auto stageCI = [](VkShaderStageFlagBits stage, VkShaderModule mod) {
        VkPipelineShaderStageCreateInfo s{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        s.stage  = stage;
        s.module = mod;
        s.pName  = "main";
        return s;
    };

    std::array<VkPipelineShaderStageCreateInfo, 4> stages{{
        stageCI(VK_SHADER_STAGE_RAYGEN_BIT_KHR,      rgenMod),
        stageCI(VK_SHADER_STAGE_MISS_BIT_KHR,         missMod),
        stageCI(VK_SHADER_STAGE_MISS_BIT_KHR,         shadowMod),
        stageCI(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,  chitMod),
    }};

    // -----------------------------------------------------------------------
    // Shader groups
    //  Group 0: rgen  (general, uses stage 0)
    //  Group 1: miss  (general, uses stage 1)
    //  Group 2: shadow miss (general, uses stage 2)
    //  Group 3: hit group (triangles, uses stage 3 as closestHit)
    // -----------------------------------------------------------------------
    auto generalGroup = [](uint32_t stageIdx) {
        VkRayTracingShaderGroupCreateInfoKHR g{
            VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
        g.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        g.generalShader    = stageIdx;
        g.closestHitShader = VK_SHADER_UNUSED_KHR;
        g.anyHitShader     = VK_SHADER_UNUSED_KHR;
        g.intersectionShader = VK_SHADER_UNUSED_KHR;
        return g;
    };

    VkRayTracingShaderGroupCreateInfoKHR hitGroup{
        VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    hitGroup.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    hitGroup.generalShader      = VK_SHADER_UNUSED_KHR;
    hitGroup.closestHitShader   = 3;
    hitGroup.anyHitShader       = VK_SHADER_UNUSED_KHR;
    hitGroup.intersectionShader = VK_SHADER_UNUSED_KHR;

    std::array<VkRayTracingShaderGroupCreateInfoKHR, 4> groups{{
        generalGroup(0),   // rgen
        generalGroup(1),   // miss sky
        generalGroup(2),   // miss shadow
        hitGroup,
    }};

    // -----------------------------------------------------------------------
    // Ray tracing pipeline
    // -----------------------------------------------------------------------
    VkRayTracingPipelineCreateInfoKHR pipeCI{
        VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    pipeCI.stageCount                   = static_cast<uint32_t>(stages.size());
    pipeCI.pStages                      = stages.data();
    pipeCI.groupCount                   = static_cast<uint32_t>(groups.size());
    pipeCI.pGroups                      = groups.data();
    pipeCI.maxPipelineRayRecursionDepth = 2; // primary + shadow
    pipeCI.layout                       = pipelineLayout;

    if (ctx.rt.createRayTracingPipelines(
            ctx.device, VK_NULL_HANDLE, VK_NULL_HANDLE,
            1, &pipeCI, nullptr, &pipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create ray tracing pipeline");

    // Destroy shader modules — they're baked into the pipeline now
    vkDestroyShaderModule(ctx.device, rgenMod,   nullptr);
    vkDestroyShaderModule(ctx.device, missMod,   nullptr);
    vkDestroyShaderModule(ctx.device, shadowMod, nullptr);
    vkDestroyShaderModule(ctx.device, chitMod,   nullptr);

    buildSBT(ctx);
    std::cout << "[RTPipeline] Pipeline + SBT created\n";
}

// ---------------------------------------------------------------------------
// buildSBT
// ---------------------------------------------------------------------------

void RTPipeline::buildSBT(VulkanContext& ctx)
{
    const uint32_t handleSize      = ctx.rtPipelineProperties.shaderGroupHandleSize;
    const uint32_t handleAlign     = ctx.rtPipelineProperties.shaderGroupHandleAlignment;
    const uint32_t baseAlign       = ctx.rtPipelineProperties.shaderGroupBaseAlignment;
    const uint32_t handleSizeAlgn  = alignUp(handleSize, handleAlign);

    // Layout (each region starts at a multiple of baseAlign):
    //   [rgen region: 1 record, size = baseAlign]
    //   [miss region: 2 records (sky + shadow)]
    //   [hit  region: 1 record]
    const uint32_t rgenSize = baseAlign;
    const uint32_t missSize = alignUp(2 * handleSizeAlgn, baseAlign);
    const uint32_t hitSize  = alignUp(1 * handleSizeAlgn, baseAlign);
    const uint32_t totalSize = rgenSize + missSize + hitSize;

    // Retrieve all group handles from the driver
    std::vector<uint8_t> handles(GROUP_COUNT * handleSize);
    if (ctx.rt.getRayTracingShaderGroupHandles(
            ctx.device, pipeline, 0, GROUP_COUNT,
            handles.size(), handles.data()) != VK_SUCCESS)
        throw std::runtime_error("Failed to get shader group handles");

    // Pack handles into SBT CPU buffer
    std::vector<uint8_t> sbt(totalSize, 0);

    auto copyHandle = [&](uint8_t* dst, uint32_t groupIdx) {
        std::memcpy(dst, handles.data() + groupIdx * handleSize, handleSize);
    };

    copyHandle(sbt.data(),                                     GROUP_RGEN);
    copyHandle(sbt.data() + rgenSize + 0 * handleSizeAlgn,    GROUP_MISS_SKY);
    copyHandle(sbt.data() + rgenSize + 1 * handleSizeAlgn,    GROUP_MISS_SHADOW);
    copyHandle(sbt.data() + rgenSize + missSize,               GROUP_HIT);

    // Upload SBT to GPU
    AllocatedBuffer staging = ctx.createBuffer(
        totalSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

    void* mapped;
    vmaMapMemory(ctx.allocator, staging.allocation, &mapped);
    std::memcpy(mapped, sbt.data(), totalSize);
    vmaUnmapMemory(ctx.allocator, staging.allocation);

    sbtBuffer = ctx.createBuffer(
        totalSize,
        VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT   |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    VkBufferCopy region{0, 0, totalSize};
    vkCmdCopyBuffer(cmd, staging.buffer, sbtBuffer.buffer, 1, &region);
    ctx.endSingleTimeCommands(cmd);
    ctx.destroyBuffer(staging);

    VkDeviceAddress base = sbtBuffer.address;

    rgenRegion.deviceAddress = base;
    rgenRegion.stride        = rgenSize;
    rgenRegion.size          = rgenSize;

    missRegion.deviceAddress = base + rgenSize;
    missRegion.stride        = handleSizeAlgn;
    missRegion.size          = missSize;

    hitRegion.deviceAddress  = base + rgenSize + missSize;
    hitRegion.stride         = handleSizeAlgn;
    hitRegion.size           = hitSize;

    callRegion = {}; // no callable shaders
}

// ---------------------------------------------------------------------------
// destroy
// ---------------------------------------------------------------------------

void RTPipeline::destroy(VulkanContext& ctx)
{
    ctx.destroyBuffer(sbtBuffer);
    if (pipeline       != VK_NULL_HANDLE) vkDestroyPipeline(ctx.device, pipeline, nullptr);
    if (pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx.device, pipelineLayout, nullptr);
    if (descriptorSetLayout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(ctx.device, descriptorSetLayout, nullptr);
}
