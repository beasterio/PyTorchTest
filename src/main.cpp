#include <torch/torch.h>
#include <torch/script.h>

#include "VulkanGPUDevice.h"


int main() {
    VulkanGPUDevice device;

    const std::string filename = "packed_data.pt";

    try
    {
        auto container = torch::jit::load(filename, torch::DeviceType::CPU);
        torch::Tensor tensor = container.attr("bps.pth").toTensor();
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