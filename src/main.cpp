#include <torch/torch.h>
#include <torch/script.h>

#include "VulkanGPUDevice.h"


int main() {
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
        // all input data is packed in packed_data.pt TorchScript model
        // the data available by key == filename
        auto container = torch::jit::load("packed_data.pt", torch::DeviceType::CPU);
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

    VulkanGPUDevice device;
    auto& shader = device.CreateShader("shaders/test.spv");

    //bind input to shader
    shader.AddBuffer(geometry_parameters);
    shader.AddBuffer(coords);
    shader.AddBuffer(bps);
    shader.AddBuffer(spheres);
    shader.AddBuffer(sd_ground_truth);
    shader.AddBuffer(grad_gt);

    //bind output buffers to shader
    const auto d_y3_d_coords_index = shader.AddBuffer(d_y3_d_coords);
    const auto loss_l1_deriv_index = shader.AddBuffer(loss_l1_deriv);
    const auto positions_index = shader.AddBuffer(positions);
    const auto nn_4th_layer_index = shader.AddBuffer(nn_4th_layer);

    shader.Bind();

    const auto elements_num = coords.size(0);
    shader.Execute(elements_num);
    shader.Wait();

    shader.ReadBuffer(d_y3_d_coords_index, d_y3_d_coords);
    shader.ReadBuffer(loss_l1_deriv_index, loss_l1_deriv);
    shader.ReadBuffer(positions_index, positions);
    shader.ReadBuffer(nn_4th_layer_index, nn_4th_layer);

    // finally, save results
    try
    {
        torch::save(d_y3_d_coords, "../output/d_y3_d_coords.pth");
        torch::save(loss_l1_deriv, "../output/loss_l1_deriv.pth");
        torch::save(positions, "../output/positions.pth");
        torch::save(nn_4th_layer, "../output/nn_4th_layer.pth");
    }
    catch (std::exception& e)
    {
        std::cerr << "can't save the data\n";
        std::cerr << e.what() << "\n";
        return -1;
    }

    return 0;
}