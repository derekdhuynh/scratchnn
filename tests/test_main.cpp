#include "Autograd_test.cpp"
#include "Tensor_test.cpp"
#include "Layers_test.cpp"

int main(void) {
  //Tensor_test();
  autograd_forward_mode_scalar_test();
  autograd_reverse_mode_scalar_test();
  Linear_with_ReLU_test();
  sigmoid_activation_test();
  bce_test();
}
