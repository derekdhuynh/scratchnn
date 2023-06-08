#include <cassert>
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
  data = shared_ptr<T[]>(arr);
  shape = shp;
  size = sz;
  strides = (this->get_strides)(shape);
}

template<class T>
scratchnn::Tensor<T>::Tensor(const shared_ptr<T[]>& ptr, const vector<size_t>& shp, const size_t sz) {
  data = ptr;
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
vector<size_t> scratchnn::Tensor<T>::get_strides(const vector<size_t>& shp) {
  vector<size_t> strd = vector<size_t>(shp.size());
  size_t offset = 1;

  // Look into this, if loop variable is size_t the unsigned ll underflows
  for (int i = strd.size() - 1; i >= 0; i--) {
    strd[i] = offset;
    offset *= shp[i];
  }
  return strd;
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
void scratchnn::Tensor<T>::_matmul_2D(const Tensor<T>& B, const shared_ptr<T[]>& startP, 
    const shared_ptr<T[]>& startA, const shared_ptr<T[]>& startB, const size_t offset) {
  /*
   i - # of rows of this
   j - # of cols of this/ # of rows of B
   k - # of cols of B

   Must be invoked on Tensors with dimensions (M, N) and (N, P)
   */

  size_t row = B.shape.size() - 2;
  size_t col = B.shape.size() - 1;

  for (size_t i = 0; i < (this->shape)[row]; i++) {
    for (size_t k = 0; k < B.shape[col]; k++) {
      T prod_ik = T(0);

      for (size_t j = 0; j < B.shape[row]; j++) {
        T a_ij = startA[i * (this->strides)[row] + j * (this->strides)[col] + offset];
        T b_jk = startB[j * B.strides[row] + k * B.strides[col] + offset];
        // T a_ij = startA[i * (this->strides)[row] + j * (this->strides)[col]];
        // T b_jk = startB[j * B.strides[row] + k * B.strides[col]];
        prod_ik += a_ij * b_jk;
      }
      startP[i * (this->strides)[row] + k * B.strides[col] + offset] = prod_ik;
      // startP[i * (this->strides)[row] + k * B.strides[col]] = prod_ik;
    }
  }
}

template<class T>
void scratchnn::Tensor<T>::_matmul_3D(const Tensor<T>& B, const shared_ptr<T[]>& startP, 
    const shared_ptr<T[]>& startA, const shared_ptr<T[]>& startB, 
    const vector<size_t>& shp, const std::vector<size_t>& strd, const size_t offset) {
  /*
   * Helper for performing a matmul between 3D tensors. 
   */
  //vector<size_t> shp = {B.shape[0], (this->shape)[1], B.shape[2]};
  //vector<size_t> strd = Tensor<T>::get_strides(shp);
  size_t off;
  size_t dim;
  if (B.shape.size() == 4) {
    off = strd[1];
    dim = shp[1];
  } else {
    off = strd[0];
    dim = shp[0];
  }

  for (int d = 0; d < dim; d++) {
    (this->_matmul_2D)(B, startP, startA, startB, offset + d * off);
  }
}

template<class T>
void scratchnn::Tensor<T>::_matmul_4D(const Tensor<T>& B, const shared_ptr<T[]>& startP,
    const shared_ptr<T[]>& startA, const shared_ptr<T[]>& startB,
    const vector<size_t>& shp, const vector<size_t>& strd) {
  /*
   * Helper for performing a matmul between 4D tensors. 
   */
  // vector<size_t> shp = {B.shape[0], B.shape[1], (this->shape)[2], B.shape[3]};
  // vector<size_t> strd = scratchnn::Tensor<T>::get_strides(shp);

  for (int d = 0; d < shp[0]; d++) {
    (this->_matmul_3D)(B, startP, startA, startB, shp, strd, d * strd[0]);
  }
}

template<class T>
scratchnn::Tensor<T>& scratchnn::Tensor<T>::matmul(const Tensor<T>& B) {
  /*
   * Performs a standard matrix multiply between two tensors. Returns a reference to
   * a newly allocated tensor.
   *
   * Note: The use of make_shared for an array only gained support in C++20
   *
   * Strassen's algorithm for tensor contraction - https://arxiv.org/pdf/1704.03092.pdf
   *
   */
  size_t total_size = 1;

  vector<size_t> new_shape = vector<size_t>(B.shape.size());
  for (int i = 0; i < B.shape.size() - 2; i++) {
    new_shape[i] = B.shape[i];
    total_size *= B.shape[i];
  }

  size_t num_mats = total_size;

  size_t rows = B.shape[B.shape.size() - 2];
  size_t cols = B.shape[B.shape.size() - 1];
  size_t mat_size = rows * cols;

  total_size *= rows * cols;

  new_shape[B.shape.size() - 2] = rows;
  new_shape[B.shape.size() - 1] = cols;

  vector<size_t> new_strides = Tensor::get_strides(new_shape);

  const shared_ptr<T[]> prod = make_shared<T[]>(total_size);

  for (int i = 0; i < num_mats; i++) {
    _matmul_2D(B, prod, this->data, B.data, i * mat_size);
  }

  /*
  if (B.shape.size() == 2) {
    _matmul_2D(B, prod, this->data, B.data, 0);
  } else if (B.shape.size() == 3) {
    _matmul_3D(B, prod, this->data, B.data, new_shape, new_strides, 0);
  } else if (B.shape.size() == 4) {
    _matmul_4D(B, prod, this->data, B.data, new_shape, new_strides);
  } else {
    throw runtime_error("Dimensions don't match.");
  }
  */

  Tensor<T>* product = new Tensor(prod, new_shape, total_size);
  return *product;
}

template<class T>
template<class C>
void scratchnn::Tensor<T>::reshape(C shp) {
  /*
   * Numpy-like reshaping method. Add support for the -1 syntax, infer the shape
   * based on the rest of the entries
   */
  size_t new_size = 0;
  vector<size_t> new_shape = vector<size_t>(shp.size());

  int infer = -1;
  int ind = 0;
  for (auto &i: shp) {
    // No missing dim so far
    if (i == -1 && infer == -1) {
      infer = ind;
    } else {
      new_size += i;
    }
    ind++;
  }

  if (infer != -1) {
    new_shape[ind] = size - new_size;
    new_size += new_shape[ind];
  }

  // Make sure the new shape is valid
  assert(new_size == size);

  shape = new_shape;
  strides = get_strides(shape);
}

template<class T>
scratchnn::Tensor<T>& scratchnn::Tensor<T>::operator+=(const Tensor<T>& rhs) {
  for (size_t i = 0; i < size; i++) {
    data[i] += rhs[i];
  }
  return *this;
}

template<class T>
scratchnn::Tensor<T> scratchnn::Tensor<T>::operator+(const Tensor<T>& rhs) {
  assert(size == rhs.size);
  Tensor<T>* res = full<T>(T(0), shape);

  for (size_t i = 0; i < size; i++) {
    res[i] = data[i] + rhs[i];
  }
  return *res;
}

template<class T>
scratchnn::Tensor<T>& scratchnn::Tensor<T>::operator-=(const Tensor<T>& rhs) {
  assert(size == rhs.size);

  for (size_t i = 0; i < size; i++) {
    data[i] -= rhs[i];
  }
  return *this;
}

template<class T>
scratchnn::Tensor<T> scratchnn::Tensor<T>::operator-(const Tensor<T>& rhs) {
  assert(size == rhs.size);

  Tensor<T>* res = full<T>(T(0), shape);

  for (size_t i = 0; i < size; i++) {
    res[i] = data[i] - rhs[i];
  }
  return *res;
}

template<class T>
scratchnn::Tensor<T>& scratchnn::Tensor<T>::operator/=(const Tensor<T>& rhs) {
  assert(size == rhs.size);

  for (size_t i = 0; i < size; i++) {
    data[i] /= rhs[i];
  }
  return *this;
}

template<class T>
scratchnn::Tensor<T> scratchnn::Tensor<T>::operator/(const Tensor<T>& rhs) {
  assert(size == rhs.size);

  Tensor<T>* res = full<T>(T(0), shape);

  for (size_t i = 0; i < size; i++) {
    res[i] = data[i] / rhs[i];
  }
  return *res;
}

template<class T>
scratchnn::Tensor<T>& scratchnn::Tensor<T>::operator*=(const Tensor<T>& rhs) {
  assert(size == rhs.size);

  for (size_t i = 0; i < size; i++) {
    data[i] *= rhs[i];
  }
  return *this;
}

template<class T>
scratchnn::Tensor<T> scratchnn::Tensor<T>::operator*(const Tensor<T>& rhs) {
  assert(size == rhs.size);

  Tensor<T> res = full<T>(T(0), shape);

  for (size_t i = 0; i < size; i++) {
    res[i] = data[i] * rhs[i];
  }
  return *res;
}

template<class T>
T scratchnn::Tensor<T>::max() {
  T mx = T(data[0]);

  for (int i = 0; i < size; i++) {
    mx = (data[i] > mx) ? data[i] : mx;
  }
  return mx;
}

template<class T>
T scratchnn::Tensor<T>::min() {
  T mn = T(data[0]);

  for (int i = 0; i < size; i++) {
    mn = (data[i] < mn) ? data[i] : mn;
  }
  return mn;
}
