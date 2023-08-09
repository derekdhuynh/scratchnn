#include "Tensor.hpp"
#include <iostream>

using namespace std;
using namespace scratchnn;

void Tensor_test() {
  // Testing matmul with 1 to 4D tensors
  vector<size_t> shp = {3, 3};
  Tensor<float> arr = Tensor<float>::full(1.f, shp);
  Tensor<float> arr2 = Tensor<float>::full(2.f, shp);
  arr.print();
  arr2.print();

  arr.assign(3, 4);
  decltype(arr) product = arr.matmul(arr2);
  arr.print();

  vector<size_t> shp2 = {3, 3, 3};
  decltype(arr) arr3 = Tensor<float>::full(1.0f, shp2);
  decltype(arr) arr4 = Tensor<float>::full(2.0f, shp2);

  arr3.assign(3, 4);
  arr3.print();
  arr3.print();

  decltype(arr) product2 = arr3.matmul(arr4);
  arr3.print();

  vector<size_t> shp3 = {3, 3, 3, 3};
  decltype(arr) arr5 = Tensor<float>::full(1.0f, shp3);
  decltype(arr) arr6 = Tensor<float>::full(2.0f, shp3);

  arr5.assign(3, 4);
  arr5.print();
  arr6.print();

  decltype(arr) product3 = arr5.matmul(arr6);
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
}
