#pragma once

#include <vulkan/vulkan.hpp>

class VulkanGPUDevice
{
public:
	VulkanGPUDevice();
	~VulkanGPUDevice();


protected:
	vk::Instance instance_;
	vk::PhysicalDevice phys_device_;
	vk::Device device_;
	uint32_t compute_queue_index;
};