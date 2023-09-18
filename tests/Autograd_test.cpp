#include "Tensor.hpp"
#include <vector>
#include <iostream>

/* x_1 --> v1 = log(x_1) --> v3 = v1 + v2 --> v5 = v3 - v4
 *    \                       |                |
 * x_2 --> v2 = x_1 * x_2------                |
 *  |                                          |
 *  -----------------------------> v4 = sin(x)--
 */

using namespace scratchnn;

void autograd_forward_mode_scalar_test() {
  auto shape = std::vector<size_t>{1};
  Tensor<float>& x_1 = tensor<float>(std::vector<float>{2}, shape, true);
  x_1.print();

  Tensor<float>& x_2 = tensor<float>(std::vector<float>{5}, shape, true);
  x_2.print();

  // 1/x_1 + deriv(x_1) = 1/0.63... + 1 = 
  auto res = x_1.log() + x_1 * x_2 - x_2.sin();
  std::cout << "Result: " << res.name << ", " << res.to_string() << std::endl;
  auto grads_x1 = x_1.forward();

  std::cout << "x_1 grads: " << std::endl;
  for (const auto& grad : grads_x1) {
    std::cout << "Grad: " << grad.to_string() << std::endl;
  } // last grad should be 5.5

  auto grads_x2 = x_2.forward();

  std::cout << "x_2 grads: " << std::endl;
  for (const auto& grad : grads_x2) {
    std::cout << "Grad: " << grad.to_string() << std::endl;
  } // last grad should be 1.716
  
  grad::get_runtime<Tensor<float>, float>()->clean();
}

void autograd_reverse_mode_scalar_test() {
  auto shape = std::vector<size_t>{1};
  Tensor<float>& x_1 = tensor<float>(std::vector<float>{2}, shape, true);
  std::cout << "x_1 name " << x_1.name << ", " << x_1.to_string() << std::endl;

  Tensor<float>& x_2 = tensor<float>(std::vector<float>{5}, shape, true);
  std::cout << "x_2 name " << x_2.name << ", " << x_2.to_string() << std::endl;

  // 1/x_1 + deriv(x_1) = 1/0.63... + 1 = 
  auto& res = x_1.log() + x_1 * x_2 - x_2.sin();
  std::cout << "Result: " << res.name << ", " << res.to_string() << std::endl;
  auto backgrads = res.backward();

  std::cout << "Grads: " << std::endl;
  for (const auto& grad : backgrads) {
    std::cout << "Grad: " << grad.to_string() << std::endl;
  } // last grad should be 5.5

  /*
  auto grads_x2 = x_2.forward();

  std::cout << "x_2 grads: " << std::endl;
  for (const auto& grad : grads_x2) {
    std::cout << "Grad: " << grad.to_string() << std::endl;
  } // last grad should be 1.716
  
  grad::get_runtime<Tensor<float>, float>()->clean();
  */
}
