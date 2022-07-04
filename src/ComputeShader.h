#pragma once

#include <vector>
#include <vulkan/vulkan.hpp>

class ComputeShader
{
public:
	ComputeShader(const vk::Device& device, const std::string& filename);
	~ComputeShader();

	void AddBuffer();
	bool Bind();
protected:
	const vk::Device& device_;
	vk::ShaderModule shader_module_;
	std::vector<vk::Buffer> buffers_;
	vk::Pipeline compute_pipeline_;
};