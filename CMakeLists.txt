cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pytorchtest)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Vulkan REQUIRED)
include_directories(${Vulkan_INCLUDE_DIRS})

execute_process(COMMAND python ${CMAKE_CURRENT_LIST_DIR}/pack_data.py)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/output)

file(GLOB SOURCES
     "src/*.h"
     "src/*.cpp"
)

add_executable(pytorchtest ${SOURCES})
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

# compile shader sources into spirv before each build.
file(GLOB GLSL_SOURCE_FILES
    "shaders/*.comp"
)

foreach(GLSL ${GLSL_SOURCE_FILES})
  get_filename_component(FILE_NAME ${GLSL} NAME_WE)
  set(SPIRV "shaders/${FILE_NAME}.spv")
  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND ${CMAKE_COMMAND} -E make_directory "shaders"
    COMMAND glslc ${GLSL} -o ${SPIRV}
    DEPENDS ${GLSL})
  list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

add_custom_target(
    shaders 
    DEPENDS ${SPIRV_BINARY_FILES}
    )
add_dependencies(pytorchtest shaders)
