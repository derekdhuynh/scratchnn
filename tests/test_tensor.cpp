#include "tensor_shared.h"
#include <iostream>

using namespace std;
using namespace scratchnn;

int main() {
  // Testing matmul wit 1 to 4D tensors
  initializer_list<size_t> shp = {3, 3};
  Tensor<float>& arr = full(1.f, shp);
  Tensor<float>& arr2 = full(2.f, shp);
  printt(arr);
  printt(arr2);

  arr.data[3] = 4;
  decltype(arr) product = arr.matmul(arr2);
  printt(product);

  initializer_list<size_t> shp2= {3, 3, 3};
  decltype(arr) arr3 = full(1.0f, shp2);
  decltype(arr) arr4 = full(2.0f, shp2);

  arr3.data[3] = 4;
  printt(arr3);
  printt(arr4);

  decltype(arr) product2 = arr3.matmul(arr4);
  printt(product2);

  initializer_list<size_t> shp3 = {3, 3, 3, 3};
  decltype(arr) arr5 = full(1.0f, shp3);
  decltype(arr) arr6 = full(2.0f, shp3);

  arr5.data[3] = 4;
  printt(arr5);
  printt(arr6);

  decltype(arr) product3 = arr5.matmul(arr6);
  printt(product3);

  /*
   * Test ops - addition, subtraction, multiplication, division
   */
}
