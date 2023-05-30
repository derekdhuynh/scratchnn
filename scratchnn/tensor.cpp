#include <iostream>
#include <vector>
#include <stdexcept>
#include <memory>

//using namespace scratchnn;
using namespace std;

template<class T>
scratchnn::Tensor<T>::Tensor(T* const arr, const vector<size_t>& shp, const size_t sz) {
  /*
   Strides - https://numpy.org/devdocs/reference/generated/numpy.ndarray.strides.html
   Numpy ndarrays Paper - https://arxiv.org/pdf/1102.1523.pdf
   */
  data = unique_ptr<T[]>(arr);
  shape = shp;
  size = sz;
  strides = (this->get_strides)(shape);
}

template<class T>
scratchnn::Tensor<T>& scratchnn::full(T val, initializer_list<size_t>& shape) {
  /*
   Initialize a tensor with shape full of val. Returns the Tensor object itself.
   */
  size_t size = 1;
  vector<size_t> shp = shape;

  for (const auto& s: shape) {
    size *= s;
  }

  // Creating a contiguous C-style array
  T* const arr = new T[size];

  for (size_t i = 0; i < size; i++) {
    arr[i] = val;
  }

  Tensor<T>* tens = new Tensor(arr, shp, size);
  return *tens;
}


template<class T>
vector<size_t> scratchnn::Tensor<T>::get_strides(const vector<size_t>& shape) {
  vector<size_t> strides = vector<size_t>(shape.size());
  size_t offset = 1;

  // Look into this, if loop variable is size_t the unsigned ll underflows
  for (int i = strides.size() - 1; i >= 0; i--) {
    strides[i] = offset;
    offset *= shape[i];
  }

  return strides;
}

template<class T>
void scratchnn::printt(Tensor<T>& tensor) {
  cout << "[";
  for (size_t i = 0; i < tensor.size; i++) {
    cout << tensor.data[i] << ", ";
  }
  cout << "\b\b";
  cout << "]";
  cout << endl;
}

template<class T>
T scratchnn::Tensor<T>::get(vector<size_t> inds) {
  int ind = 0;

  for (size_t i = 0; i < (this->strides).size(); i++) {
    ind += inds[i] * this->strides[i];
  }

  return *(this->data + ind);
}

template<class T>
void scratchnn::Tensor<T>::_matmul_2D(const Tensor<T>& B, T* const startP, 
    const unique_ptr<T[]>& startA, const unique_ptr<T[]>& startB, const size_t offset) {
  /*
   i - # of rows of this
   j - # of cols of this/ # of rows of B
   k - # of cols of B

   Must be invoked on Tensors with dimensions (M, N) and (N, P)
   */
  // float* prod = new float[B.shape[0] * (this->shape)[1]];

  size_t row = B.shape.size() - 2;
  size_t col = B.shape.size() - 1;

  // vector<int> shape = {(this->shape)[row], B.shape[col]};

  for (size_t i = 0; i < (this->shape)[row]; i++) {
    for (size_t k = 0; k < B.shape[col]; k++) {
      T prod_ik = T(0);
      // float* ik = startP + (i * (this->strides)[row] + k * B.strides[col]);

      for (size_t j = 0; j < B.shape[row]; j++) {
        // float a_ij = *(startA + (i * (this->strides)[row] + j * (this->strides)[col]));
        // float b_jk = *(startB + (j * B.strides[row] + k * B.strides[col]));

        T a_ij = startA[i * (this->strides)[row] + j * (this->strides)[col] + offset];
        T b_jk = startB[j * B.strides[row] + k * B.strides[col] + offset];
        prod_ik += a_ij * b_jk;
      }
      startP[i * (this->strides)[row] + k * B.strides[col] + offset] = prod_ik;
    }
  }
}

template<class T>
void scratchnn::Tensor<T>::_matmul_3D(const Tensor<T>& B, T* const startP, 
    const unique_ptr<T[]>& startA, const unique_ptr<T[]>& startB, const size_t offset) {
  vector<size_t> shape = {B.shape[0], (this->shape)[1], B.shape[2]};
  vector<size_t> strd = Tensor<T>::get_strides(shape);
  //cout << strd[0] << endl;

  for (int d = 0; d < B.shape[0]; d++) {
    // (this->_matmul_2D)(B, startP + d * strides[0], startA + d * strides[0], 
    //     startB + d * strides[0]);
    (this->_matmul_2D)(B, startP, startA, startB, offset + d * strd[0]);
  }

  // Tensor product = Tensor(prod, shape);
  // return product;
}

template<class T>
void scratchnn::Tensor<T>::_matmul_4D(const Tensor<T>& B, T* const startP,
    const unique_ptr<T[]>& startA, const unique_ptr<T[]>& startB) {
  vector<size_t> shape = {B.shape[0], B.shape[1], (this->shape)[2], B.shape[3]};
  vector<size_t> strd = scratchnn::Tensor<T>::get_strides(shape);
  cout << strd[0] << endl;

  for (int d = 0; d < B.shape[0]; d++) {
    (this->_matmul_3D)(B, startP, startA, startB, d * strd[0]);
  }
}

template<class T>
scratchnn::Tensor<T>& scratchnn::Tensor<T>::matmul(const Tensor<T>& B) {
  size_t total_size = 1;

  vector<size_t> shape = vector<size_t>(B.shape.size());
  for (int i = 0; i < B.shape.size() - 2; i++) {
    shape[i] = B.shape[i];
    total_size *= B.shape[i];
  }

  size_t rows = B.shape[B.shape.size() - 2];
  size_t cols = B.shape[B.shape.size() - 1];
  total_size *= rows * cols;

  shape[rows] = B.shape[rows];
  shape[cols] = (this->shape)[cols];
  vector<size_t> strides = Tensor::get_strides(shape);

  T* const prod = new T[total_size];

  if (B.shape.size() == 2) {
    _matmul_2D(B, prod, this->data, B.data, 0);
  } else if (B.shape.size() == 3) {
    _matmul_3D(B, prod, this->data, B.data, 0);
  } else if (B.shape.size() == 4) {
    _matmul_4D(B, prod, this->data, B.data);
  } else {
    throw runtime_error("Dimensions don't match.");
  }

  Tensor<T>* product = new Tensor(prod, shape, total_size);
  return *product;
}
