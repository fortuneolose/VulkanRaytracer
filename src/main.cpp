#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "VulkanContext.h"
#include "Scene.h"
#include "AccelStructure.h"
#include "RTPipeline.h"
#include "Renderer.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Window dimensions
// ---------------------------------------------------------------------------
static constexpr uint32_t WIDTH  = 1280;
static constexpr uint32_t HEIGHT = 720;

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return 1;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE,  GLFW_FALSE); // no resize handling needed for now

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT,
                                          "Vulkan RTX Path Tracer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return 1;
    }

    glfwSetKeyCallback(window, [](GLFWwindow* w, int key, int, int action, int) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
            glfwSetWindowShouldClose(w, GLFW_TRUE);
    });

    // Shader SPIR-V files are placed next to the executable in a /shaders/ sub-folder
    // by the CMake build system.  Use the executable's working directory as the base.
    std::filesystem::path shaderPath =
        std::filesystem::current_path() / "shaders" / "";

    // Normalise to forward slashes (works on all platforms)
    std::string shaderDir = shaderPath.generic_string();
    if (!shaderDir.empty() && shaderDir.back() != '/')
        shaderDir += '/';

    VulkanContext  ctx;
    Scene          scene;
    AccelStructure accel;
    RTPipeline     rtPipeline;
    Renderer       renderer;

    try {
        std::cout << "Initialising Vulkan context...\n";
        ctx.init(window, WIDTH, HEIGHT);

        std::cout << "Building scene...\n";
        scene.buildScene();
        scene.uploadToGPU(ctx);

        std::cout << "Building acceleration structures...\n";
        accel.buildBLASes(ctx, scene);
        accel.buildTLAS  (ctx, scene);

        std::cout << "Building RT pipeline (shader dir: " << shaderDir << ")...\n";
        rtPipeline.build(ctx, shaderDir);

        std::cout << "Initialising renderer...\n";
        renderer.init(ctx, scene, accel, rtPipeline);

        std::cout << "Ready.  Controls: WASD/QE = move, RMB-drag = look, ESC = quit\n";

        double lastTime = glfwGetTime();

        while (!glfwWindowShouldClose(window)) {
            double now = glfwGetTime();
            float  dt  = static_cast<float>(now - lastTime);
            lastTime   = now;

            glfwPollEvents();

            int w, h;
            glfwGetFramebufferSize(window, &w, &h);
            if (w == 0 || h == 0) continue; // minimised

            scene.camera.processInput(window, dt);
            renderer.drawFrame(ctx, scene, accel, rtPipeline,
                               static_cast<float>(w) / static_cast<float>(h));
        }

        vkDeviceWaitIdle(ctx.device);

    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << '\n';
    }

    renderer.destroy(ctx);
    rtPipeline.destroy(ctx);
    accel.destroy(ctx);
    scene.destroy(ctx);
    ctx.destroy();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
