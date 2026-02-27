cmake_minimum_required(VERSION 3.20)
project(VulkanRaytracer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Vulkan REQUIRED)

include(FetchContent)

# vk-bootstrap: Vulkan instance/device creation boilerplate
FetchContent_Declare(
    vk-bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(vk-bootstrap)

# GLFW: windowing and Vulkan surface
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# GLM: math library
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
)
FetchContent_MakeAvailable(glm)

# STB: image loading (HDR environments)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(stb)

# Vulkan Memory Allocator
FetchContent_Declare(
    VulkanMemoryAllocator
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG        v3.1.0
)
FetchContent_MakeAvailable(VulkanMemoryAllocator)

# ---------------------------------------------------------------------------
# Shader compilation
# ---------------------------------------------------------------------------
find_program(GLSLC glslc
    HINTS "$ENV{VULKAN_SDK}/Bin"
          "$ENV{VULKAN_SDK}/bin"
    REQUIRED
)

set(SHADER_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SPIRV_DIR   ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders)
file(MAKE_DIRECTORY ${SPIRV_DIR})

set(SHADERS
    ${SHADER_DIR}/raygen.rgen
    ${SHADER_DIR}/miss.rmiss
    ${SHADER_DIR}/shadow.rmiss
    ${SHADER_DIR}/closesthit.rchit
)

set(SPIRV_OUTPUTS)
foreach(SHADER ${SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPIRV_OUTPUT ${SPIRV_DIR}/${SHADER_NAME}.spv)
    add_custom_command(
        OUTPUT  ${SPIRV_OUTPUT}
        COMMAND ${GLSLC} --target-env=vulkan1.2 -o ${SPIRV_OUTPUT} ${SHADER}
        DEPENDS ${SHADER}
        COMMENT "Compiling shader: ${SHADER_NAME}"
        VERBATIM
    )
    list(APPEND SPIRV_OUTPUTS ${SPIRV_OUTPUT})
endforeach()

add_custom_target(Shaders ALL DEPENDS ${SPIRV_OUTPUTS})

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
file(GLOB SOURCES src/*.cpp)

add_executable(VulkanRaytracer ${SOURCES})
add_dependencies(VulkanRaytracer Shaders)

target_include_directories(VulkanRaytracer PRIVATE
    src
    ${stb_SOURCE_DIR}
    ${vulkanmemoryallocator_SOURCE_DIR}/include
)

target_link_libraries(VulkanRaytracer PRIVATE
    Vulkan::Vulkan
    vk-bootstrap::vk-bootstrap
    glfw
    glm::glm
)

target_compile_definitions(VulkanRaytracer PRIVATE
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
)

if(MSVC)
    target_compile_options(VulkanRaytracer PRIVATE /W3 /wd4201 /wd4100)
else()
    target_compile_options(VulkanRaytracer PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
