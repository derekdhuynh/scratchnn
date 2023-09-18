#include "Layers.hpp"
#include "Losses.hpp"

using namespace scratchnn;

void Linear_with_ReLU_test() {
  std::cout << "Test linear" << std::endl;
  auto& inp = Tensor<float>::full(1, std::vector<size_t>{1, 30});
  auto linear = layers::Linear<float>(30, 50);
  auto linear2 = layers::Linear<float>(30, 50);
  auto relu = layers::ReLU<float>();
  auto& out = relu(linear(inp));

  std::cout << "Linear res: " << out.to_string() << std::endl;
  std::cout << "Linear res shape: " << out.shape[0] << " " << out.shape[1] << std::endl;
}

void sigmoid_activation_test() {
  auto& inp = Tensor<float>::arange(0, 10, std::vector<size_t>{1, 10});
  auto sigmoid = layers::Sigmoid<float>();
  auto& out = sigmoid(inp);

  out.print();
}

void bce_test() {
  auto& y_true = tensor<float>(std::vector<float>{1, 0, 0, 1}, std::vector<int>{4});
  auto& y_pred = tensor<float>(std::vector<float>{0.95, 0.8, 0.1, 0.17}, std::vector<int>{5});
  auto binary_ce = losses::BinaryCrossEntropy<float>();
  auto& loss = binary_ce(y_pred, y_true);

  loss.print();
}
