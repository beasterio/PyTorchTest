#include <torch/torch.h>
#include <torch/script.h>

#include "VulkanGPUDevice.h"


int main() {
    VulkanGPUDevice device;
    auto& shader = device.CreateShader("shaders/test.spv");

    const std::string filename = "packed_data.pt";

    //input data
    torch::Tensor geometry_parameters;
    torch::Tensor coords;
    torch::Tensor bps;
    torch::Tensor spheres;
    torch::Tensor sd_ground_truth;
    torch::Tensor grad_gt;

    //result data
    torch::Tensor d_y3_d_coords;
    torch::Tensor loss_l1_deriv;
    torch::Tensor positions;
    torch::Tensor nn_4th_layer;
    try
    {
        auto container = torch::jit::load(filename, torch::DeviceType::CPU);
        geometry_parameters = container.attr("geometry_parameters.pth").toTensor();
        coords = container.attr("coords.pth").toTensor();
        bps = container.attr("bps.pth").toTensor();
        spheres = container.attr("spheres.pth").toTensor();
        sd_ground_truth = container.attr("sd_ground_truth.pth").toTensor();
        grad_gt = container.attr("grad_gt.pth").toTensor();

        spheres = spheres[0];
        bps = bps[0];

        // coords is an array of vec3 values, all output data sizes should be the same
        const auto elements_num = coords.size(0);
        d_y3_d_coords = torch::zeros({ elements_num, 3 }, torch::kFloat32);
        loss_l1_deriv = torch::zeros(elements_num, torch::kFloat32);
        positions = torch::zeros({ elements_num, 16 }, torch::kFloat32);
        nn_4th_layer = torch::zeros(elements_num, torch::kFloat32);
    }
    catch (std::exception& e)
    {
        std::cerr << "error loading the data\n";
        std::cerr << e.what() << "\n";
        return -1;
    }

    //bind input to shader
    shader.AddBuffer(geometry_parameters);
    shader.AddBuffer(coords);
    shader.AddBuffer(bps);
    shader.AddBuffer(spheres);
    shader.AddBuffer(sd_ground_truth);
    shader.AddBuffer(grad_gt);

    //bind output buffers to shader
    shader.AddBuffer(d_y3_d_coords);
    shader.AddBuffer(loss_l1_deriv);
    shader.AddBuffer(positions);
    shader.AddBuffer(nn_4th_layer);

    shader.Bind();

    const auto elements_num = coords.size(0);
    shader.Execute(elements_num);
    shader.Wait();

    std::cout << "ok\n";

    return 0;
}