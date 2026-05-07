#include <torch/extension.h>

// 声明 CUDA 启动函数
void launch_softmax(float* input, float* output, int batch_size, int num_classes);

torch::Tensor softmax_forward(torch::Tensor input) {
    // 1. 基础检查
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be Float32");

    // 2. 获取维度
    // 假设输入是 [batch_size, num_classes]
    int batch_size = input.size(0);
    int num_classes = input.size(1);

    auto output = torch::empty_like(input);

    // 3. 调用
    launch_softmax(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_classes
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softmax_forward, "Online Softmax forward");
}