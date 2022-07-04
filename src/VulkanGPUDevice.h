#pragma once

#include <list>
#include <vulkan/vulkan.hpp>
#include "ComputeShader.h"

class VulkanGPUDevice
{
public:
	VulkanGPUDevice();
	~VulkanGPUDevice();

	ComputeShader& CreateShader(const std::string& filename);
protected:
	vk::Instance instance_;
	vk::PhysicalDevice phys_device_;
	vk::Device device_;
	uint32_t compute_queue_index_;
	std::list<ComputeShader> shaders_;
};