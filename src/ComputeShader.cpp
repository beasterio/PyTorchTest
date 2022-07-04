#pragma once

#include "ComputeShader.h";

#include <fstream>

ComputeShader::ComputeShader(const vk::Device& device, const std::string& filename)
    : device_(device)
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
}

ComputeShader::~ComputeShader()
{
    device_.destroy(compute_pipeline_);
    device_.destroy(shader_module_);
}

void ComputeShader::AddBuffer()
{

}


bool ComputeShader::Bind()
{
    std::vector<vk::DescriptorSetLayoutBinding> descriptor_set_layout_binding;
    for (uint32_t i = 0; i < buffers_.size(); ++i)
    {
        descriptor_set_layout_binding.emplace_back(i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);
    }

    vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_createinfo(
        vk::DescriptorSetLayoutCreateFlags(),
        descriptor_set_layout_binding);
    vk::DescriptorSetLayout descriptor_set_layout = device_.createDescriptorSetLayout(descriptor_set_layout_createinfo);


    vk::PipelineLayoutCreateInfo pipeline_layout_create_info(vk::PipelineLayoutCreateFlags(), descriptor_set_layout);
    vk::PipelineLayout pipeline_layout = device_.createPipelineLayout(pipeline_layout_create_info);
    vk::PipelineCache pipeline_cache = device_.createPipelineCache(vk::PipelineCacheCreateInfo());

    vk::PipelineShaderStageCreateInfo pipeline_shader_create_info(
        vk::PipelineShaderStageCreateFlags(),
        vk::ShaderStageFlagBits::eCompute,
        shader_module_,
        "main");
    vk::ComputePipelineCreateInfo compute_pipeline_create_info(
        vk::PipelineCreateFlags(),
        pipeline_shader_create_info,
        pipeline_layout);
    
    auto compute_pipeline_result = device_.createComputePipeline(pipeline_cache, compute_pipeline_create_info);
    if (compute_pipeline_result.result != vk::Result::eSuccess)
    {
        return false;
    }

    compute_pipeline_ = compute_pipeline_result.value;

    vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, buffers_.size());
    vk::DescriptorPoolCreateInfo descriptor_pool_create_info(vk::DescriptorPoolCreateFlags(), 1, descriptor_pool_size);
    vk::DescriptorPool descriptor_pool = device_.createDescriptorPool(descriptor_pool_create_info);

    vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(descriptor_pool, 1, &descriptor_set_layout);
    const std::vector<vk::DescriptorSet> descriptor_sets = device_.allocateDescriptorSets(descriptor_set_alloc_info);
    vk::DescriptorSet descriptor_set = descriptor_sets.front();

    std::vector<vk::WriteDescriptorSet> write_descriptor_sets;
    for (uint32_t i = 0; i < buffers_.size(); ++i)
    {
        //todo set proper buffer size.
        const auto size = 322;
        vk::DescriptorBufferInfo buffer_info(buffers_[i], 0, size);
        write_descriptor_sets.emplace_back(descriptor_set, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &buffer_info);
    }

    device_.updateDescriptorSets(write_descriptor_sets, {});
}