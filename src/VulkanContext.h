#pragma once

#include <vulkan/vulkan.h>
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>
#include <GLFW/glfw3.h>

#include <vector>
#include <string>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Resource wrappers
// ---------------------------------------------------------------------------

struct AllocatedBuffer {
    VkBuffer      buffer     = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkDeviceAddress address  = 0;
};

struct AllocatedImage {
    VkImage       image      = VK_NULL_HANDLE;
    VkImageView   view       = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
};

// ---------------------------------------------------------------------------
// Ray-tracing extension function pointers
// ---------------------------------------------------------------------------

struct RTFunctions {
    PFN_vkCreateAccelerationStructureKHR           createAccelerationStructure          = nullptr;
    PFN_vkDestroyAccelerationStructureKHR          destroyAccelerationStructure         = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR    getAccelerationStructureBuildSizes   = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR        cmdBuildAccelerationStructures       = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR getAccelerationStructureDeviceAddress = nullptr;
    PFN_vkCreateRayTracingPipelinesKHR             createRayTracingPipelines            = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR       getRayTracingShaderGroupHandles      = nullptr;
    PFN_vkCmdTraceRaysKHR                          cmdTraceRays                         = nullptr;
};

// ---------------------------------------------------------------------------
// VulkanContext
// ---------------------------------------------------------------------------

class VulkanContext {
public:
    GLFWwindow* window = nullptr;

    VkInstance               instance       = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR             surface        = VK_NULL_HANDLE;
    VkPhysicalDevice         physicalDevice = VK_NULL_HANDLE;
    VkDevice                 device         = VK_NULL_HANDLE;

    VkQueue  graphicsQueue      = VK_NULL_HANDLE;
    uint32_t graphicsQueueFamily = 0;

    VmaAllocator   allocator    = VK_NULL_HANDLE;
    VkCommandPool  commandPool  = VK_NULL_HANDLE;

    // Swapchain
    VkSwapchainKHR           swapchain      = VK_NULL_HANDLE;
    VkFormat                 swapchainFormat{};
    VkExtent2D               swapchainExtent{};
    std::vector<VkImage>     swapchainImages;
    std::vector<VkImageView> swapchainImageViews;

    // Hardware RT properties (pipeline + acceleration structure)
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR   rtPipelineProperties{};
    VkPhysicalDeviceAccelerationStructurePropertiesKHR asProperties{};

    RTFunctions rt;

    // Lifecycle
    void init(GLFWwindow* win, uint32_t width, uint32_t height);
    void destroy();

    // Buffer / image helpers
    AllocatedBuffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                 VmaMemoryUsage memUsage = VMA_MEMORY_USAGE_AUTO,
                                 VmaAllocationCreateFlags flags = 0);
    void            destroyBuffer(AllocatedBuffer& buf);

    AllocatedImage  createImage(uint32_t width, uint32_t height,
                                VkFormat format, VkImageUsageFlags usage);
    void            destroyImage(AllocatedImage& img);

    // Single-use command buffer helpers
    VkCommandBuffer beginSingleTimeCommands();
    void            endSingleTimeCommands(VkCommandBuffer cmd);

    // Shader module from a SPIR-V file
    VkShaderModule loadShaderModule(const std::string& path);

    // Buffer device address (Vulkan 1.2 core)
    VkDeviceAddress getBufferAddress(VkBuffer buffer);

private:
    vkb::Instance   vkbInstance;
    vkb::Device     vkbDevice;
    vkb::Swapchain  vkbSwapchain;

    void createSwapchain(uint32_t width, uint32_t height);
    void loadRTFunctions();
};
