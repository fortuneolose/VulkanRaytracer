# Vulkan RTX Path Tracer

A real-time hardware-accelerated path tracer built from scratch in C++ using the Vulkan Ray Tracing extension (`VK_KHR_ray_tracing_pipeline`).

![Vulkan](https://img.shields.io/badge/Vulkan-1.2-red?logo=vulkan) ![C++17](https://img.shields.io/badge/C%2B%2B-17-blue?logo=cplusplus) ![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey?logo=windows)

---

## Features

- **Hardware Ray Tracing** — fully leverages RTX GPU hardware via Vulkan's ray tracing pipeline
- **BVH Acceleration Structures** — BLAS per mesh, single TLAS for the full scene, built on-device
- **Progressive Path Tracing** — accumulates samples over time for noise-free convergence; temporal accumulation resets automatically on camera movement
- **Physically Based Rendering (PBR)**
  - Lambertian diffuse with cosine-weighted hemisphere sampling
  - Cook-Torrance GGX metallic BRDF with importance sampling
  - Dielectric glass with Fresnel reflection/refraction and total internal reflection
- **Multi-bounce Lighting** — configurable bounce depth with Russian roulette path termination (kicks in after bounce 3)
- **Direct Sun Illumination** — shadow rays cast toward a directional sun light
- **Anti-Aliasing** — per-sample sub-pixel jitter
- **Free-fly Camera** — WASD + Q/E for translation, right-mouse-drag for look
- **PCG Random Number Generator** — fast, high-quality per-pixel seeding in shaders

---

## Technology Stack

| Library | Purpose |
|---|---|
| [Vulkan SDK](https://vulkan.lunarg.com/) | GPU API & ray tracing extensions |
| [GLFW 3.4](https://github.com/glfw/glfw) | Window creation & input |
| [GLM 1.0.1](https://github.com/g-truc/glm) | Linear algebra |
| [vk-bootstrap](https://github.com/charles-lunarg/vk-bootstrap) | Vulkan instance/device setup |
| [Vulkan Memory Allocator 3.1](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) | GPU memory management |
| [stb_image](https://github.com/nothings/stb) | HDR image loading |

All dependencies are fetched automatically at configure time via CMake `FetchContent`.

---

## Building

### Prerequisites

- Vulkan SDK 1.3+ (with `glslc` on your `PATH` or `VULKAN_SDK` set)
- CMake 3.20+
- A C++17-capable compiler (MSVC 2022 recommended on Windows)
- An RTX-capable GPU (or any GPU supporting `VK_KHR_ray_tracing_pipeline`)

### Steps

```bash
git clone https://github.com/FortuneMU2025/VulkanRaytracer.git
cd VulkanRaytracer
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The executable and compiled shaders are placed in `build/bin/`.

---

## Project Structure

```
VulkanRaytracer/
├── src/
│   ├── main.cpp            # Entry point, window + render loop
│   ├── VulkanContext.h/cpp # Instance, device, swapchain, memory helpers
│   ├── Scene.h/cpp         # Camera, mesh data, GPU buffer upload
│   ├── AccelStructure.h/cpp# BLAS & TLAS construction
│   ├── RTPipeline.h/cpp    # Ray tracing pipeline, SBT, descriptors
│   ├── Renderer.h/cpp      # Frame loop, sync objects, descriptor sets
│   └── types.h             # Shared CPU/GPU types (Vertex, Material, ...)
└── shaders/
    ├── common.glsl         # Shared structs & PCG RNG
    ├── raygen.rgen         # Primary ray generation & path-trace loop
    ├── closesthit.rchit    # PBR shading, shadow rays, next-bounce sampling
    ├── miss.rmiss          # Sky / environment colour
    └── shadow.rmiss        # Shadow ray miss (light is visible)
```

---

## Controls

| Key / Mouse | Action |
|---|---|
| `W` / `S` | Move forward / backward |
| `A` / `D` | Strafe left / right |
| `Q` / `E` | Move down / up |
| Right-mouse drag | Look around |
| `ESC` | Quit |

---

## License

MIT — see [LICENSE](LICENSE) for details.
