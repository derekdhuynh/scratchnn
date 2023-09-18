/*
 * A high level functional neural net API
 */
#pragma once

#include "Tensor.hpp"

namespace scratchnn { namespace layers {

template<typename T, typename Container>
scratchnn::Tensor<T>& he_normal_init(const Container& shape) {
  auto& weight_tensor = scratchnn::Tensor<T>::he_normal(shape, true);
  return weight_tensor;
}

/*
 * A layer class that represents a fully connected hidden layer. Note that the initialization
 * of the weight matrix of the layer is transposed to how it normally is (i.e in_features x out_features,
 * where we perform a matmul like W.matmul(x), where x is a column vector. Instead we want to do a
 * matmul with a row vector and the weight matrix, like x.matmul(W), which produces another row vector.
 * This makes it easier to compute the shapes for the gradient updates.
 *
 * Pytorch weight layout reasons - https://stackoverflow.com/questions/53465608/pytorch-shape-of-nn-linear-weights
 *
 * This implementation does not transpose the internal weight matrix like pytorch does.
 */
template<typename T>
class Linear {
  public:
    size_t in_features;
    size_t out_features;

    scratchnn::Tensor<T>& weight;
    scratchnn::Tensor<T>& bias;
    std::string initializer;

    Linear(const size_t in, const size_t out, const std::string& init = "he_normal") :
      in_features(in),
      out_features(out),
      weight(scratchnn::Tensor<T>::he_normal(std::vector<size_t>{in, out})),
      bias(scratchnn::Tensor<T>::full(0, std::vector<size_t>{1, out})),
      initializer(init) {}

    scratchnn::Tensor<T>& operator()(scratchnn::Tensor<T>& input) {
      // Need input to be num_examples x in_features, each row is an individual example
      return input.matmul(weight) + bias;
    }
};

template<typename T>
class ReLU {
  public:
    ReLU() = default;

    scratchnn::Tensor<T>& operator()(scratchnn::Tensor<T>& input) {
      return input.relu();
    }
};

template<typename T>
class Sigmoid {
  public:
    Sigmoid() = default;

    scratchnn::Tensor<T>& operator()(scratchnn::Tensor<T>& input) {
      return input.sigmoid();
    }
};

}} // namespace scratchnn::layers
