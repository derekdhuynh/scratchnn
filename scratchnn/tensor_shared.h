/*
Tensor header.
 */
#include <initializer_list>
#include <vector>
#include <memory>

namespace scratchnn {
  template<class T>
  class Tensor {
    /*
     * Numpy/Pytorch/Tensorflow-like ndarray class.
     * For now, only supports arrays of up to 4 dimensions.
     */
    public:
      std::vector<size_t> shape;
      std::vector<size_t> strides;
      size_t size;
      std::shared_ptr<T[]> data;

      Tensor() = default;
      ~Tensor() = default;
      Tensor(T* const arr, const std::vector<size_t>& shp, const size_t sz);
      Tensor(const std::shared_ptr<T[]>& arr, const std::vector<size_t>& shp, const size_t sz);

      T get(std::vector<size_t> s);

      void _matmul_2D(const Tensor<T>& B, const std::shared_ptr<T[]>& startP, 
          const std::shared_ptr<T[]>& startA, const std::shared_ptr<T[]>& startB, 
          const size_t offset);

      void _matmul_3D(const Tensor<T>& B, const std::shared_ptr<T[]>& startP, 
          const std::shared_ptr<T[]>& startA, const std::shared_ptr<T[]>& startB,
          const std::vector<size_t>& shp, const std::vector<size_t>& strd, const size_t offset);

      void _matmul_4D(const Tensor<T>& B, const std::shared_ptr<T[]>& startP, 
          const std::shared_ptr<T[]>& startA, const std::shared_ptr<T[]>& startB,
          const std::vector<size_t>& shp, const std::vector<size_t>& strd);

      Tensor<T>& matmul(const Tensor<T>& B);
      Tensor<T> dot(const Tensor& B);
      static std::vector<size_t> get_strides(const std::vector<size_t>& shape);

      // Probably cast C to a tensorshape class or something in the future
      template<class C>
      void reshape(C shape);

      /*
      Implement ufuncs + overload artithmetic operators:
        - add (+)
        - subtract (-)
        - divide (/)
        - max
        - min
        - a generic u func template that is performed on each element in the array
       */
      Tensor<T>& operator+=(const Tensor<T>& rhs);
      Tensor<T> operator+(const Tensor<T>& rhs);

      Tensor<T>& operator-=(const Tensor<T>& rhs);
      Tensor<T> operator-(const Tensor<T>& rhs);

      Tensor<T>& operator/=(const Tensor<T>& rhs);
      Tensor<T> operator/(const Tensor<T>& rhs);

      Tensor<T>& operator*=(const Tensor<T>& rhs);
      Tensor<T> operator*(const Tensor<T>& rhs);

      T max();
      T min();
  };

  template<class T>
  Tensor<T>& full(T val, std::initializer_list<size_t>& shape);

  template<class T>
  void printt(Tensor<T>& tensor);
}

#include "tensor_shared.cpp"
