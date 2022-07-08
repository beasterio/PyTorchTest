#include "VulkanGPUDevice.h"

#include <iostream>

namespace {

vk::PhysicalDevice GetPhysDevice(const vk::Instance& instance)
{
    const auto devices = instance.enumeratePhysicalDevices();
    if (devices.empty())
    {
        std::cout << "No devices available!" << std::endl;
        return {};
    }

    return devices.front();
}

uint32_t GetComputeQueueIndex(const vk::PhysicalDevice& device)
{
    std::vector<vk::QueueFamilyProperties> queue_family_props = device.getQueueFamilyProperties();
    auto it = std::find_if(queue_family_props.begin(), queue_family_props.end(), [](const vk::QueueFamilyProperties& prop)
        {
            return prop.queueFlags & vk::QueueFlagBits::eCompute;
        });
    const uint32_t index = std::distance(queue_family_props.begin(), it);

    return index;
}

vk::Device GetComputeDevice(const vk::Instance& instance, const vk::PhysicalDevice& phys_device, const uint32_t queue_family_index)
{
    const float priorities[] = { 1.f };
    const auto device_queue_create_info = vk::DeviceQueueCreateInfo(vk::DeviceQueueCreateFlags(),
        queue_family_index,        // Queue Family Index
        1,                  // Number of Queues
        priorities);        // Priorities, cant be null
    vk::DeviceCreateInfo device_create_info(vk::DeviceCreateFlags(), device_queue_create_info);
    return phys_device.createDevice(device_create_info);
}

} // anonymous namespace

VulkanGPUDevice::VulkanGPUDevice()
{
    vk::ApplicationInfo app_info{
    "app",
    1,
    nullptr,
    0,
    VK_API_VERSION_1_3
    };

    const std::vector<const char*> layers = {/* "VK_LAYER_KHRONOS_validation" */};
    vk::InstanceCreateInfo instance_info(vk::InstanceCreateFlags(),
        &app_info,
        layers.size(),
        layers.data());

    instance_ = vk::createInstance(instance_info);
    phys_device_ = GetPhysDevice(instance_);
    compute_queue_index_ = GetComputeQueueIndex(phys_device_);
    device_ = GetComputeDevice(instance_, phys_device_, compute_queue_index_);
}

VulkanGPUDevice::~VulkanGPUDevice()
{
    shaders_.clear();
    device_.destroy();
    instance_.destroy();
}

ComputeShader& VulkanGPUDevice::CreateShader(const std::string& filename)
{
    shaders_.emplace_back(device_, phys_device_, compute_queue_index_, filename);
    return shaders_.back();
}
