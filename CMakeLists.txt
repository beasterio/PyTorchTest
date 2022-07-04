cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pytorchtest)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Vulkan REQUIRED)
include_directories(${Vulkan_INCLUDE_DIRS})

file(GLOB sources
     "src/*.h"
     "src/*.cpp"
)

add_executable(pytorchtest ${sources})
target_link_libraries(pytorchtest "${TORCH_LIBRARIES}" "${Vulkan_LIBRARIES}")

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET pytorchtest
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:pytorchtest>)
endif (MSVC)