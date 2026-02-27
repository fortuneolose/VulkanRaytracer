// VMA implementation — compiled exactly once here
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include "VulkanContext.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Debug messenger callback
// ---------------------------------------------------------------------------

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT   severity,
    VkDebugUtilsMessageTypeFlagsEXT          /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* /*user*/)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        std::cerr << "[Vulkan] " << data->pMessage << '\n';
    return VK_FALSE;
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

void VulkanContext::init(GLFWwindow* win, uint32_t width, uint32_t height)
{
    window = win;

    // ------------------------------------------------------------------
    // Instance
    // ------------------------------------------------------------------
    vkb::InstanceBuilder builder;
    auto instResult = builder
        .set_app_name("VulkanRaytracer")
        .require_api_version(1, 2, 0)
        .request_validation_layers(true)
        .set_debug_callback(debugCallback)
        .build();
    if (!instResult)
        throw std::runtime_error("Failed to create Vulkan instance: " + instResult.error().message());

    vkbInstance   = instResult.value();
    instance      = vkbInstance.instance;
    debugMessenger = vkbInstance.debug_messenger;

    // ------------------------------------------------------------------
    // Surface
    // ------------------------------------------------------------------
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
        throw std::runtime_error("Failed to create window surface");

    // ------------------------------------------------------------------
    // Physical device — require RT extensions
    // ------------------------------------------------------------------
    vkb::PhysicalDeviceSelector selector(vkbInstance);
    auto physResult = selector
        .set_surface(surface)
        .set_minimum_version(1, 2)
        .add_required_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
        .add_required_extension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
        .add_required_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)
        .select();
    if (!physResult)
        throw std::runtime_error("No suitable GPU found (need VK_KHR_ray_tracing_pipeline): " +
                                 physResult.error().message());

    physicalDevice = physResult.value().physical_device;

    // ------------------------------------------------------------------
    // Query RT + AS properties
    // ------------------------------------------------------------------
    rtPipelineProperties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    asProperties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;

    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext                   = &rtPipelineProperties;
    rtPipelineProperties.pNext     = &asProperties;
    vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

    // ------------------------------------------------------------------
    // Logical device — enable Vulkan 1.2 features + RT features
    // ------------------------------------------------------------------
    VkPhysicalDeviceVulkan12Features features12{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    features12.bufferDeviceAddress                              = VK_TRUE;
    features12.descriptorIndexing                               = VK_TRUE;
    features12.runtimeDescriptorArray                           = VK_TRUE;
    features12.shaderSampledImageArrayNonUniformIndexing        = VK_TRUE;
    features12.scalarBlockLayout                                = VK_TRUE;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    asFeatures.accelerationStructure = VK_TRUE;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
    rtFeatures.rayTracingPipeline = VK_TRUE;

    vkb::DeviceBuilder devBuilder(physResult.value());
    auto devResult = devBuilder
        .add_pNext(&features12)
        .add_pNext(&asFeatures)
        .add_pNext(&rtFeatures)
        .build();
    if (!devResult)
        throw std::runtime_error("Failed to create logical device: " + devResult.error().message());

    vkbDevice = devResult.value();
    device    = vkbDevice.device;

    auto qRes = vkbDevice.get_queue(vkb::QueueType::graphics);
    if (!qRes)
        throw std::runtime_error("Failed to get graphics queue");
    graphicsQueue       = qRes.value();
    graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    // ------------------------------------------------------------------
    // Vulkan Memory Allocator
    // ------------------------------------------------------------------
    VmaVulkanFunctions vkFuncs{};
    vkFuncs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vkFuncs.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo vmaInfo{};
    vmaInfo.physicalDevice   = physicalDevice;
    vmaInfo.device           = device;
    vmaInfo.instance         = instance;
    vmaInfo.vulkanApiVersion = VK_API_VERSION_1_2;
    vmaInfo.flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaInfo.pVulkanFunctions = &vkFuncs;

    if (vmaCreateAllocator(&vmaInfo, &allocator) != VK_SUCCESS)
        throw std::runtime_error("Failed to create VMA allocator");

    // ------------------------------------------------------------------
    // Command pool
    // ------------------------------------------------------------------
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = graphicsQueueFamily;
    poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);

    // ------------------------------------------------------------------
    // Swapchain
    // ------------------------------------------------------------------
    createSwapchain(width, height);

    // ------------------------------------------------------------------
    // Load KHR RT function pointers
    // ------------------------------------------------------------------
    loadRTFunctions();

    std::cout << "[VulkanContext] Initialized on "
              << props2.properties.deviceName << '\n';
}

// ---------------------------------------------------------------------------
// createSwapchain
// ---------------------------------------------------------------------------

void VulkanContext::createSwapchain(uint32_t width, uint32_t height)
{
    vkb::SwapchainBuilder swapBuilder(vkbDevice);
    auto swapResult = swapBuilder
        .set_desired_format({VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
        .set_desired_present_mode(VK_PRESENT_MODE_MAILBOX_KHR)
        .add_fallback_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(width, height)
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .build();
    if (!swapResult)
        throw std::runtime_error("Failed to create swapchain: " + swapResult.error().message());

    vkbSwapchain      = swapResult.value();
    swapchain         = vkbSwapchain.swapchain;
    swapchainFormat   = vkbSwapchain.image_format;
    swapchainExtent   = vkbSwapchain.extent;
    swapchainImages   = vkbSwapchain.get_images().value();
    swapchainImageViews = vkbSwapchain.get_image_views().value();
}

// ---------------------------------------------------------------------------
// loadRTFunctions
// ---------------------------------------------------------------------------

void VulkanContext::loadRTFunctions()
{
    auto load = [&](auto& dest, const char* sym) {
        dest = reinterpret_cast<std::remove_reference_t<decltype(dest)>>(
            vkGetDeviceProcAddr(device, sym));
        if (!dest)
            throw std::runtime_error(std::string("Failed to load ") + sym);
    };

    load(rt.createAccelerationStructure,           "vkCreateAccelerationStructureKHR");
    load(rt.destroyAccelerationStructure,          "vkDestroyAccelerationStructureKHR");
    load(rt.getAccelerationStructureBuildSizes,    "vkGetAccelerationStructureBuildSizesKHR");
    load(rt.cmdBuildAccelerationStructures,        "vkCmdBuildAccelerationStructuresKHR");
    load(rt.getAccelerationStructureDeviceAddress, "vkGetAccelerationStructureDeviceAddressKHR");
    load(rt.createRayTracingPipelines,             "vkCreateRayTracingPipelinesKHR");
    load(rt.getRayTracingShaderGroupHandles,       "vkGetRayTracingShaderGroupHandlesKHR");
    load(rt.cmdTraceRays,                          "vkCmdTraceRaysKHR");
}

// ---------------------------------------------------------------------------
// Buffer / image helpers
// ---------------------------------------------------------------------------

AllocatedBuffer VulkanContext::createBuffer(VkDeviceSize size,
                                            VkBufferUsageFlags usage,
                                            VmaMemoryUsage memUsage,
                                            VmaAllocationCreateFlags flags)
{
    AllocatedBuffer result{};

    VkBufferCreateInfo bufInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufInfo.size  = size;
    bufInfo.usage = usage;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = memUsage;
    allocCI.flags = flags;

    if (vmaCreateBuffer(allocator, &bufInfo, &allocCI,
                        &result.buffer, &result.allocation, nullptr) != VK_SUCCESS)
        throw std::runtime_error("Failed to create buffer");

    if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
        result.address = getBufferAddress(result.buffer);

    return result;
}

void VulkanContext::destroyBuffer(AllocatedBuffer& buf)
{
    if (buf.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator, buf.buffer, buf.allocation);
        buf.buffer     = VK_NULL_HANDLE;
        buf.allocation = VK_NULL_HANDLE;
        buf.address    = 0;
    }
}

AllocatedImage VulkanContext::createImage(uint32_t w, uint32_t h,
                                          VkFormat format, VkImageUsageFlags usage)
{
    AllocatedImage result{};

    VkImageCreateInfo imgInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imgInfo.imageType   = VK_IMAGE_TYPE_2D;
    imgInfo.format      = format;
    imgInfo.extent      = {w, h, 1};
    imgInfo.mipLevels   = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.samples     = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling      = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage       = usage;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO;
    allocCI.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    if (vmaCreateImage(allocator, &imgInfo, &allocCI,
                       &result.image, &result.allocation, nullptr) != VK_SUCCESS)
        throw std::runtime_error("Failed to create image");

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image    = result.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format   = format;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCreateImageView(device, &viewInfo, nullptr, &result.view);

    return result;
}

void VulkanContext::destroyImage(AllocatedImage& img)
{
    if (img.view  != VK_NULL_HANDLE) { vkDestroyImageView(device, img.view,  nullptr); img.view  = VK_NULL_HANDLE; }
    if (img.image != VK_NULL_HANDLE) { vmaDestroyImage(allocator, img.image, img.allocation); img.image = VK_NULL_HANDLE; }
}

// ---------------------------------------------------------------------------
// Single-use command buffers
// ---------------------------------------------------------------------------

VkCommandBuffer VulkanContext::beginSingleTimeCommands()
{
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool        = commandPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &ai, &cmd);

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

void VulkanContext::endSingleTimeCommands(VkCommandBuffer cmd)
{
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;

    vkQueueSubmit(graphicsQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);
    vkFreeCommandBuffers(device, commandPool, 1, &cmd);
}

// ---------------------------------------------------------------------------
// Shader module
// ---------------------------------------------------------------------------

VkShaderModule VulkanContext::loadShaderModule(const std::string& path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open shader: " + path);

    size_t            size = static_cast<size_t>(file.tellg());
    std::vector<char> code(size);
    file.seekg(0);
    file.read(code.data(), size);

    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = size;
    info.pCode    = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule mod;
    if (vkCreateShaderModule(device, &info, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error("Failed to create shader module: " + path);
    return mod;
}

// ---------------------------------------------------------------------------
// Buffer device address
// ---------------------------------------------------------------------------

VkDeviceAddress VulkanContext::getBufferAddress(VkBuffer buffer)
{
    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer = buffer;
    return vkGetBufferDeviceAddress(device, &info);
}

// ---------------------------------------------------------------------------
// Destroy
// ---------------------------------------------------------------------------

void VulkanContext::destroy()
{
    vkDeviceWaitIdle(device);

    for (auto& view : swapchainImageViews)
        vkDestroyImageView(device, view, nullptr);
    vkb::destroy_swapchain(vkbSwapchain);

    vkDestroyCommandPool(device, commandPool, nullptr);
    vmaDestroyAllocator(allocator);
    vkb::destroy_device(vkbDevice);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkb::destroy_instance(vkbInstance);
}
