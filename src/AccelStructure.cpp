#include "AccelStructure.h"
#include "types.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cstring>
#include <stdexcept>
#include <iostream>

// ---------------------------------------------------------------------------
// buildSingleBLAS
// ---------------------------------------------------------------------------

BLAS AccelStructure::buildSingleBLAS(VulkanContext& ctx,
                                      const MeshData& mesh,
                                      VkDeviceAddress vertexBaseAddress,
                                      VkDeviceAddress indexBaseAddress,
                                      uint32_t        vertexOffset,
                                      uint32_t        indexOffset)
{
    // Triangle geometry description
    VkAccelerationStructureGeometryTrianglesDataKHR triData{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
    triData.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;
    triData.vertexData.deviceAddress = vertexBaseAddress + vertexOffset * sizeof(Vertex);
    triData.vertexStride             = sizeof(Vertex);
    triData.maxVertex                = static_cast<uint32_t>(mesh.vertices.size()) - 1;
    triData.indexType                = VK_INDEX_TYPE_UINT32;
    triData.indexData.deviceAddress  = indexBaseAddress + indexOffset * sizeof(uint32_t);

    VkAccelerationStructureGeometryKHR geometry{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.geometry.triangles = triData;
    geometry.flags              = VK_GEOMETRY_OPAQUE_BIT_KHR;

    // Query sizes
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries   = &geometry;

    uint32_t primitiveCount = static_cast<uint32_t>(mesh.indices.size()) / 3;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    ctx.rt.getAccelerationStructureBuildSizes(
        ctx.device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &primitiveCount, &sizeInfo);

    // Allocate AS storage buffer
    BLAS blas{};
    blas.buffer = ctx.createBuffer(
        sizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    // Create AS handle
    VkAccelerationStructureCreateInfoKHR createInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    createInfo.buffer = blas.buffer.buffer;
    createInfo.size   = sizeInfo.accelerationStructureSize;
    createInfo.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    ctx.rt.createAccelerationStructure(ctx.device, &createInfo, nullptr, &blas.handle);

    // Scratch buffer (aligned to hardware requirement)
    uint32_t scratchAlign = ctx.asProperties.minAccelerationStructureScratchOffsetAlignment;
    VkDeviceSize scratchSize = sizeInfo.buildScratchSize +
                               alignUp(static_cast<uint32_t>(sizeInfo.buildScratchSize),
                                       scratchAlign);

    AllocatedBuffer scratch = ctx.createBuffer(
        scratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    // Align scratch address
    VkDeviceAddress scratchAddr = scratch.address;
    if (scratchAlign > 1)
        scratchAddr = (scratchAddr + scratchAlign - 1) & ~VkDeviceAddress(scratchAlign - 1);

    // Build
    buildInfo.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.dstAccelerationStructure = blas.handle;
    buildInfo.scratchData.deviceAddress = scratchAddr;

    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = primitiveCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;

    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    ctx.rt.cmdBuildAccelerationStructures(cmd, 1, &buildInfo, &pRange);
    ctx.endSingleTimeCommands(cmd);

    ctx.destroyBuffer(scratch);

    // Get device address
    VkAccelerationStructureDeviceAddressInfoKHR addrInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    addrInfo.accelerationStructure = blas.handle;
    blas.address = ctx.rt.getAccelerationStructureDeviceAddress(ctx.device, &addrInfo);

    return blas;
}

// ---------------------------------------------------------------------------
// buildBLASes
// ---------------------------------------------------------------------------

void AccelStructure::buildBLASes(VulkanContext& ctx, const Scene& scene)
{
    uint32_t vertexOffset = 0;
    uint32_t indexOffset  = 0;

    for (size_t i = 0; i < scene.meshes.size(); ++i) {
        const MeshData& mesh = scene.meshes[i];
        blases.push_back(buildSingleBLAS(ctx, mesh,
            scene.vertexBuffer.address,
            scene.indexBuffer.address,
            vertexOffset, indexOffset));

        vertexOffset += static_cast<uint32_t>(mesh.vertices.size());
        indexOffset  += static_cast<uint32_t>(mesh.indices.size());
        std::cout << "  BLAS[" << i << "] built — " << mesh.indices.size() / 3 << " triangles\n";
    }
}

// ---------------------------------------------------------------------------
// buildTLAS
// ---------------------------------------------------------------------------

void AccelStructure::buildTLAS(VulkanContext& ctx, const Scene& scene)
{
    // One VkAccelerationStructureInstanceKHR per scene instance
    std::vector<VkAccelerationStructureInstanceKHR> vkInstances;
    vkInstances.reserve(scene.instances.size());

    for (size_t i = 0; i < scene.instances.size(); ++i) {
        const SceneInstance& si = scene.instances[i];

        VkAccelerationStructureInstanceKHR vkInst{};

        // VkTransformMatrixKHR is row-major 3x4; GLM is column-major 4x4
        glm::mat4 rowMaj = glm::transpose(si.transform);
        std::memcpy(&vkInst.transform, &rowMaj, sizeof(VkTransformMatrixKHR));

        vkInst.instanceCustomIndex                    = si.meshIndex; // used in closesthit
        vkInst.mask                                   = 0xFF;
        vkInst.instanceShaderBindingTableRecordOffset = 0;
        vkInst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        vkInst.accelerationStructureReference         = blases[si.meshIndex].address;

        vkInstances.push_back(vkInst);
    }

    VkDeviceSize instSize = vkInstances.size() * sizeof(VkAccelerationStructureInstanceKHR);

    // Upload instance data
    AllocatedBuffer staging = ctx.createBuffer(
        instSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

    void* mapped;
    vmaMapMemory(ctx.allocator, staging.allocation, &mapped);
    std::memcpy(mapped, vkInstances.data(), instSize);
    vmaUnmapMemory(ctx.allocator, staging.allocation);

    instanceBuffer = ctx.createBuffer(
        instSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

    // Copy and barrier in one command buffer, then build TLAS
    VkAccelerationStructureGeometryInstancesDataKHR instData{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
    instData.data.deviceAddress = instanceBuffer.address;

    VkAccelerationStructureGeometryKHR geometry{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType          = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances    = instData;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries   = &geometry;

    uint32_t primitiveCount = static_cast<uint32_t>(vkInstances.size());

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    ctx.rt.getAccelerationStructureBuildSizes(
        ctx.device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &primitiveCount, &sizeInfo);

    // Allocate TLAS storage
    tlasBuffer = ctx.createBuffer(
        sizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    VkAccelerationStructureCreateInfoKHR createInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    createInfo.buffer = tlasBuffer.buffer;
    createInfo.size   = sizeInfo.accelerationStructureSize;
    createInfo.type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    ctx.rt.createAccelerationStructure(ctx.device, &createInfo, nullptr, &tlas);

    // Scratch
    uint32_t scratchAlign = ctx.asProperties.minAccelerationStructureScratchOffsetAlignment;
    AllocatedBuffer scratch = ctx.createBuffer(
        sizeInfo.buildScratchSize + scratchAlign,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    VkDeviceAddress scratchAddr = scratch.address;
    if (scratchAlign > 1)
        scratchAddr = (scratchAddr + scratchAlign - 1) & ~VkDeviceAddress(scratchAlign - 1);

    buildInfo.mode                      = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.dstAccelerationStructure  = tlas;
    buildInfo.scratchData.deviceAddress = scratchAddr;

    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = primitiveCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;

    // Single command buffer: copy instances → barrier → build TLAS
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

    VkBufferCopy copyRegion{0, 0, instSize};
    vkCmdCopyBuffer(cmd, staging.buffer, instanceBuffer.buffer, 1, &copyRegion);

    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        0, 1, &barrier, 0, nullptr, 0, nullptr);

    ctx.rt.cmdBuildAccelerationStructures(cmd, 1, &buildInfo, &pRange);

    ctx.endSingleTimeCommands(cmd);

    ctx.destroyBuffer(staging);
    ctx.destroyBuffer(scratch);

    std::cout << "  TLAS built — " << primitiveCount << " instances\n";
}

// ---------------------------------------------------------------------------
// destroy
// ---------------------------------------------------------------------------

void AccelStructure::destroy(VulkanContext& ctx)
{
    for (auto& blas : blases) {
        if (blas.handle != VK_NULL_HANDLE)
            ctx.rt.destroyAccelerationStructure(ctx.device, blas.handle, nullptr);
        ctx.destroyBuffer(blas.buffer);
    }
    if (tlas != VK_NULL_HANDLE)
        ctx.rt.destroyAccelerationStructure(ctx.device, tlas, nullptr);
    ctx.destroyBuffer(tlasBuffer);
    ctx.destroyBuffer(instanceBuffer);
}
