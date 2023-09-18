/*
 * Loss functions.
 */
#include "Tensor.hpp"

namespace scratchnn { namespace losses {

template<typename T>
class BinaryCrossEntropy {
  public:
    std::string reduction;

    BinaryCrossEntropy(const std::string& reduce="mean") : reduction(reduce) {}

    scratchnn::Tensor<T>& operator()(scratchnn::Tensor<T>& y_pred, scratchnn::Tensor<T>& y_true) {
      return scratchnn::bce(y_pred, y_true, reduction);
    }
};

}} // namespace scratchnn::losses
