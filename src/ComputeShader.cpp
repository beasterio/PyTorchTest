#pragma once

#include "ComputeShader.h"

#include <fstream>

ComputeShader::ComputeShader(const vk::Device& device, const vk::PhysicalDevice& phys_device, uint32_t compute_queue_index, const std::string& filename)
    : device_(device)
    , compute_queue_index_(compute_queue_index)
{
    std::vector<char> shader_contents;
    if (std::ifstream file{ filename, std::ios::binary | std::ios::ate })
    {
        const size_t size = file.tellg();
        file.seekg(0);
        shader_contents.resize(size, '\0');
        file.read(shader_contents.data(), size);
    }

    vk::ShaderModuleCreateInfo create_info(
        vk::ShaderModuleCreateFlags(),
        shader_contents.size(),
        reinterpret_cast<const uint32_t*>(shader_contents.data()));
    shader_module_ = device_.createShaderModule(create_info);

    vk::PhysicalDeviceMemoryProperties memory_properties = phys_device.getMemoryProperties();
    memory_type_index_ = uint32_t(~0);
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i)
    {
        vk::MemoryType MemoryType = memory_properties.memoryTypes[i];
        if ((vk::MemoryPropertyFlagBits::eHostVisible & MemoryType.propertyFlags) &&
            (vk::MemoryPropertyFlagBits::eHostCoherent & MemoryType.propertyFlags))
        {
            memory_type_index_ = i;
            break;
        }
    }
}

ComputeShader::~ComputeShader()
{
    for (const auto& data : buffers_data_)
    {
        device_.freeMemory(data.memory);
        device_.destroyBuffer(data.buffer);
    }

    if (command_pool_)
    {
        device_.resetCommandPool(command_pool_, vk::CommandPoolResetFlags());
        device_.destroyCommandPool(command_pool_);
    }

    if (descriptor_set_layout_)
    {
        device_.destroyDescriptorSetLayout(descriptor_set_layout_);
        device_.destroyPipelineLayout(pipeline_layout_);
        device_.destroyPipelineCache(pipeline_cache_);
    }

    if (descriptor_pool_)
    {
        device_.destroyDescriptorPool(descriptor_pool_);
    }

    if (compute_pipeline_)
    {
        device_.destroy(compute_pipeline_);
    }

    device_.destroy(shader_module_);
}

uint32_t ComputeShader::AddBuffer(uint32_t size)
{
    vk::BufferCreateInfo buffer_create_info{
        vk::BufferCreateFlags(),
        size,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::SharingMode::eExclusive,

        // SharingMode is not eConcurrent => Queue family indices ignored
        1,
        &compute_queue_index_
    };

    vk::Buffer buffer = device_.createBuffer(buffer_create_info);
    vk::MemoryRequirements memory_requirements = device_.getBufferMemoryRequirements(buffer);

    vk::MemoryAllocateInfo buffer_memory_allocate_info(memory_requirements.size, memory_type_index_);
    vk::DeviceMemory memory = device_.allocateMemory(buffer_memory_allocate_info);

    device_.bindBufferMemory(buffer, memory, 0);

    Buffer result{ buffer, memory, size };
    buffers_data_.push_back(result);

    return buffers_data_.size() - 1;
}

uint32_t ComputeShader::AddBuffer(const std::vector<char>& data)
{
    const auto index = AddBuffer(data.size());
    const auto& buffer_data = buffers_data_[index];

    char* buffer_ptr = static_cast<char*>(device_.mapMemory(buffer_data.memory, 0, data.size()));
    for (int32_t i = 0; i < data.size(); ++i)
    {
        buffer_ptr[i] = data[i];
    }
    device_.unmapMemory(buffer_data.memory);

    return index;
}

uint32_t ComputeShader::AddBuffer(const torch::Tensor& tensor)
{
    const auto& storage = tensor.storage();
    const auto data_size_bytes = tensor.numel() * tensor.element_size();
    const auto index = AddBuffer(data_size_bytes);
    const auto& buffer_data = buffers_data_[index];

    char* data_ptr = static_cast<char*>(storage.data());
    char* buffer_ptr = static_cast<char*>(device_.mapMemory(buffer_data.memory, 0, data_size_bytes));
    for (int32_t i = 0; i < data_size_bytes; ++i)
    {
        buffer_ptr[i] = data_ptr[i];
    }
    device_.unmapMemory(buffer_data.memory);

    return index;

}

void ComputeShader::ReadBuffer(uint32_t index, torch::Tensor& tensor)
{
    const auto& buffer_data = buffers_data_[index];
    std::vector<float> result;
    result.reserve(buffer_data.size);

    void* out_buffer_ptr = device_.mapMemory(buffer_data.memory, 0, buffer_data.size);
    float* tmp = static_cast<float*>(out_buffer_ptr);
    for (int i = 0; i < buffer_data.size / sizeof(float); ++i)
    {
        result.push_back(tmp[i]);
    }


    tensor = torch::from_blob(out_buffer_ptr, tensor.sizes(), tensor.dtype());
    //std::cout << tensor << std::endl;
    device_.unmapMemory(buffer_data.memory);
}

bool ComputeShader::Bind()
{
    std::vector<vk::DescriptorSetLayoutBinding> descriptor_set_layout_binding;
    for (uint32_t i = 0; i < buffers_data_.size(); ++i)
    {
        descriptor_set_layout_binding.emplace_back(i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);
    }

    vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_createinfo(
        vk::DescriptorSetLayoutCreateFlags(),
        descriptor_set_layout_binding);
    descriptor_set_layout_ = device_.createDescriptorSetLayout(descriptor_set_layout_createinfo);

    vk::PipelineLayoutCreateInfo pipeline_layout_create_info(vk::PipelineLayoutCreateFlags(), descriptor_set_layout_);
    pipeline_layout_ = device_.createPipelineLayout(pipeline_layout_create_info);
    pipeline_cache_ = device_.createPipelineCache(vk::PipelineCacheCreateInfo());

    vk::PipelineShaderStageCreateInfo pipeline_shader_create_info(
        vk::PipelineShaderStageCreateFlags(),
        vk::ShaderStageFlagBits::eCompute,
        shader_module_,
        "main");
    vk::ComputePipelineCreateInfo compute_pipeline_create_info(
        vk::PipelineCreateFlags(),
        pipeline_shader_create_info,
        pipeline_layout_);
    
    auto compute_pipeline_result = device_.createComputePipeline(pipeline_cache_, compute_pipeline_create_info);
    if (compute_pipeline_result.result != vk::Result::eSuccess)
    {
        return false;
    }

    compute_pipeline_ = compute_pipeline_result.value;

    vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, buffers_data_.size());
    vk::DescriptorPoolCreateInfo descriptor_pool_create_info(vk::DescriptorPoolCreateFlags(), 1, descriptor_pool_size);
    descriptor_pool_ = device_.createDescriptorPool(descriptor_pool_create_info);

    vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(descriptor_pool_, 1, &descriptor_set_layout_);
    const std::vector<vk::DescriptorSet> descriptor_sets = device_.allocateDescriptorSets(descriptor_set_alloc_info);
    descriptor_set_ = descriptor_sets.front();

    for (uint32_t i = 0; i < buffers_data_.size(); ++i)
    {
        vk::DescriptorBufferInfo buffer_info(buffers_data_[i].buffer, 0, buffers_data_[i].size);
        vk::WriteDescriptorSet desc_set = { descriptor_set_, i, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &buffer_info };
        device_.updateDescriptorSets({ desc_set }, {});
    }

    vk::CommandPoolCreateInfo command_pool_create_info(vk::CommandPoolCreateFlags(), compute_queue_index_);
    command_pool_ = device_.createCommandPool(command_pool_create_info);

    vk::CommandBufferAllocateInfo command_buffer_alloc_info(
        command_pool_,                         // Command Pool
        vk::CommandBufferLevel::ePrimary,    // Level
        1);                                  // Num Command Buffers
    const std::vector<vk::CommandBuffer> cmd_buffers = device_.allocateCommandBuffers(command_buffer_alloc_info);
    cmb_buffer_ = cmd_buffers.front();

    return true;
}

void ComputeShader::Execute(uint32_t group_size)
{
    vk::CommandBufferBeginInfo cmd_buffer_begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cmb_buffer_.begin(cmd_buffer_begin_info);
    cmb_buffer_.bindPipeline(vk::PipelineBindPoint::eCompute, compute_pipeline_);
    cmb_buffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
        pipeline_layout_,
        0,
        { descriptor_set_ },
        {});
    cmb_buffer_.dispatch(group_size, 1, 1);
    cmb_buffer_.end();
}

void ComputeShader::Wait()
{
    vk::Queue queue = device_.getQueue(compute_queue_index_, 0);
    vk::Fence fence = device_.createFence(vk::FenceCreateInfo());

    vk::SubmitInfo submit_info(0,
        nullptr,
        nullptr,
        1,
        &cmb_buffer_);
    queue.submit({ submit_info }, fence);
    device_.waitForFences({ fence },
        true,
        uint64_t(-1));

    device_.destroyFence(fence);
}
