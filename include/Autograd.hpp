#include <unordered_map>
#include <unordered_set>
#include <string>
#include <memory>
#include <initializer_list> // temp
#include "Tensor.hpp"
/*
 * Template provides autograd (reverse/forward mode automatic differentiation)
 * functionality, with arbitrarily composed expressions involving the Tensor class.
 *
 * The Theory
 * ---
 *  - In deep learning, we would like to have a convienent method for computing the gradients
 *    of scalar loss/objective functions w.r.t large matrices of parameters in a DNN or even
 *    parametric ML models like and SVM or regression methods.
 *  - In otherwords, we want to know what happens to the loss mathematically if we made a slight
 *    tweak to the parameters (this is given to us via the gradient matrices).
 *  - 
 *
 * Implementation - Static Computation Graph
 * ---
 *  - Each tensor like in torch will have the status of requiring gradients or not. If
 *    gradients are required, then it will need to be registered with the runtime when constructed,
 *    and the computation graph will be built as operations are defined.
 *  - When a tensor is created, it must be recorded in this runtime. And then when 
 *    operations are peformed between tensors, a callback should be triggered at the end
 *    of the operation that forms an edge between the tensors. 
 *  - Subsequent calls to that should then be performed through this graph for 
 *    efficiency purposes. It is likely possible to reduce temporaries and optimize specific 
 *    operations as much as possible if we know the full graph of an expression.
 *  - Then on any tensor in the graph, you can simply call backward() to perform reverse
 *    mode autodiff, the result of which will simply be a composition of functions that
 *    we constructed through our computation graph.
 */

namespace scratchnn {

enum OpType {
  Add,
  Mult,
  Div,
  None,
};

/*
 * A struct representing an arithmetic operation performed on a tensor. Any
 * given node in the computation graph may or may not contain a tensor, depending
 * on whether or not the 
 */
template<class T>
struct OpNode {
  Tensor<T>* tensor; // may or may not be empty
  std::unordered_map<OpNode*, OpType> next_nodes;
};

class ExecutionEngine {

public:
  std::unordered_map<std::string, OpNode*> op_nodes;
  size_t num_tensors = 0;
  
  template<class T>
  Tensor<T> backward(const std::string& rhs_name, const std::string& lhs_name) {
    return Tensor(new T[1], {1}, 1);
  }

  template<class T>
  Tensor<T> forward(const std::string& tensor_name) {
    return Tensor(new T[1], {1}, 1);
  }

  template<class T>
  std::string register_tensor(const std::string& tensor_name, const Tensor<T>* tensor_ptr) {
    std::string name;
    if (tensor_name != "" && op_nodes.find(tensor_name) != op_nodes.end()) {
      name = name;
    } else {
      name = "tensor_" + std::string(runtime_engine->num_tensors);
    }
    OpNode node;
    node.tensor = tensor_ptr;

    op_nodes.emplace(name, node);
    return name;
  }

  /*
   * Adding an edge between two tensors in our computation graph. Requiree pointers
   * to the "lhs" and "rhs" of the expression. If an edge is not already present,
   * add it to both of theier adjacency lists.
   *
   * It is possible for the rhs to be an OpNode that does not itself contain a pointer
   * to a tensor (i.e a unary operation).
   */
  template<class T>
  void add_grad_edge(const std::string& lhs, const std::string& rhs, OpType op_type) {
    auto lhs_it = op_nodes.find(lhs)
    auto rhs_it = op_nodes.find(rhs)

    if (lhs_it->second.next_nodes.find(&rhs_it->second) != lhs_it->second.next_nodes.end()) {
      lhs_it->second.next_nodes.emplace(&rhs_it->second, op_type);
      rhs_it->second.next_nodes.emplace(&lhs_it->second, op_type);
    }
  }

};

/* 
 * Class that is responsible for determining the proper gradient operation
 * given the "forward" op.
 */
class GradOpResolver {
  static void resolve_op(OpNode* op1, OpNode* op2) { 
    OpType op_type = op1.next_nodes.find(op2)->second;

    switch (op_type) {
      default:
        // Do something to the graph
        break;
    }
  }
}

} // namespace scratchnn
