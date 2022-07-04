#pragma once

#include <vector>
#include <vulkan/vulkan.hpp>

class ComputeShader
{
public:
	ComputeShader(const vk::Device& device, const vk::PhysicalDevice& phys_device, uint32_t compute_queue_index, const std::string& filename);
	~ComputeShader();

	uint32_t AddBuffer(uint32_t size);
	uint32_t AddBuffer(const std::vector<char>& data);
	const std::vector<char> ReadBuffer(uint32_t index);

	bool Bind();
	void Execute(uint32_t group_size);
	void Wait();
protected:
	struct Buffer
	{
		vk::Buffer buffer;
		vk::DeviceMemory memory;
		uint32_t size;
	};

	const vk::Device& device_;
	uint32_t memory_type_index_;
	uint32_t compute_queue_index_;
	vk::ShaderModule shader_module_;
	std::vector<Buffer> buffers_data_;
	vk::DescriptorPool descriptor_pool_;
	vk::Pipeline compute_pipeline_;
	vk::PipelineCache pipeline_cache_;
	vk::DescriptorSet descriptor_set_;
	vk::DescriptorSetLayout descriptor_set_layout_;
	vk::PipelineLayout pipeline_layout_;
	vk::CommandPool command_pool_;
	vk::CommandBuffer cmb_buffer_;
};