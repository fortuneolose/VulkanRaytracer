@echo off
set VULKAN_SDK=C:\VulkanSDK\1.4.341.1
set PATH=%PATH%;C:\VulkanSDK\1.4.341.1\Bin;C:\Program Files\CMake\bin;C:\TDM-GCC-64\bin

echo VULKAN_SDK=%VULKAN_SDK%
cmake --version
glslc --version

cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_C_COMPILER=C:/TDM-GCC-64/bin/gcc.exe ^
    -DCMAKE_CXX_COMPILER=C:/TDM-GCC-64/bin/g++.exe
