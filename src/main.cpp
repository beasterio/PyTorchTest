#include <vulkan/vulkan.hpp>
#include <torch/torch.h>
#include <torch/script.h>

std::vector<std::string> GetSupportedExtensions() {
    uint32_t count;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);

    std::vector<VkExtensionProperties> extensions(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data());
    std::vector<std::string> results;
    for (auto& extension : extensions) {
        results.push_back(extension.extensionName);
    }
    return results;
}

int main() {
    std::cout << GetSupportedExtensions() << std::endl;

    const std::string filename = "packed_data.pt";

    try
    {
        auto container = torch::jit::load(filename, torch::DeviceType::CPU);
        torch::Tensor tensor = container.attr("bps.pth").toTensor();
        std::cout << "tensor=" << tensor << std::endl;
    }
    catch (std::exception& e)
    {
        std::cerr << "error loading the pickle\n";
        std::cerr << e.what() << "\n";
        return -1;
    }

    std::cout << "ok\n";

    return 0;
}