#pragma once
#include "VulkanContext.h"
#include "Scene.h"
#include <vector>

// One Bottom-Level Acceleration Structure per mesh
struct BLAS {
    VkAccelerationStructureKHR handle  = VK_NULL_HANDLE;
    AllocatedBuffer            buffer;
    VkDeviceAddress            address = 0;
};

class AccelStructure {
public:
    std::vector<BLAS>          blases;

    VkAccelerationStructureKHR tlas       = VK_NULL_HANDLE;
    AllocatedBuffer            tlasBuffer;

    void buildBLASes(VulkanContext& ctx, const Scene& scene);
    void buildTLAS  (VulkanContext& ctx, const Scene& scene);
    void destroy    (VulkanContext& ctx);

private:
    AllocatedBuffer instanceBuffer; // lives as long as the TLAS

    BLAS buildSingleBLAS(VulkanContext& ctx,
                         const MeshData& mesh,
                         VkDeviceAddress vertexBaseAddress,
                         VkDeviceAddress indexBaseAddress,
                         uint32_t        vertexOffset,
                         uint32_t        indexOffset);

    static uint32_t alignUp(uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); }
};
