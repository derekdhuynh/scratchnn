#include "Tensor.hpp"
#include <iostream>
#include <numbers>

using std::cout;
using std::endl;
using std::vector;
using namespace scratchnn;

void Tensor_test() {
  // Testing matmul with 1 to 4D tensors
  vector<size_t> shp = {3, 3};
  Tensor<float> arr = Tensor<float>::full(1.f, shp, false);
  Tensor<float> arr2 = Tensor<float>::full(2.f, shp, false);
  arr.print();
  arr2.print();

  arr.assign(3, 4);
  auto product = arr.matmul(arr2);
  arr.print();

  vector<size_t> shp2 = {3, 3, 3};
  auto arr3 = Tensor<float>::full(1.0f, shp2, false);
  auto arr4 = Tensor<float>::full(2.0f, shp2, false);

  arr3.assign(3, 4);
  arr3.print();
  arr3.print();

  auto product2 = arr3.matmul(arr4);
  arr3.print();

  vector<size_t> shp3 = {3, 3, 3, 3};
  auto arr5 = Tensor<float>::full(1.0f, shp3, false);
  auto arr6 = Tensor<float>::full(2.0f, shp3, false);

  arr5.assign(3, 4);
  arr5.print();
  arr6.print();

  auto product3 = arr5.matmul(arr6);
  product3.print();

  /*
   * Test ops - addition, subtraction, multiplication, division
   */
  cout << "Inplace adding tensors" << endl;
  arr += Tensor<float>::full(1.f, shp);
  arr.print();

  cout << "Out of place adding tensors" << endl;
  auto sum = arr + arr2;
  sum.print();

  cout << "sin(pi/2) = 1" << endl;
  auto sin_tensor = Tensor<float>::full(std::numbers::pi_v<float> / 2, {2, 4}).sin();
  sin_tensor.print();
}
