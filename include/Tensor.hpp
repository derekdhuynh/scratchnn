/*
Tensor header.
 */
#pragma once

#include <cmath>
#include <random>
#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include <string>
#include <sstream>
#include <initializer_list>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>
#include <deque>
#include <iostream>
#include <ranges>

namespace scratchnn {


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
 *    operations are peformed between tensors, a callback should be triggered to the 
 *    runtime engine to form an edge between the tensors to signify the gradient
 *    dependency.
 *  - Subsequent calls to that should then be performed through this graph for 
 *    efficiency purposes. It is likely possible to reduce temporaries and optimize specific 
 *    operations as much as possible if we know the full graph of an expression.
 *  - Then on any tensor in the graph, you can simply call backward() to perform reverse
 *    mode autodiff, the result of which will simply be a composition of functions that
 *    we constructed through our computation graph.
 *  - For a binop, the DAG will have two nodes pointing to the result.
 */

namespace grad {

/*
template<class T>
class Tensor;
*/

enum OpType {
  Add,
  Subtract,
  Mult,
  Div,
  Log,
  Exp,
  Sin,
  Cos,
  ReLU,
  Sigmoid,
  Matmul,
  BinaryCrossEntropy,
  Sum,
  None,
};


/*
 * A class representing an arithmetic operation performed on a tensor. Any
 * given node in the computation graph may or may not contain a tensor, depending
 * on whether or not the 
 */
template<typename Tensor>
class OpNode {
  public:
    OpNode() = default;
    OpNode(std::unique_ptr<Tensor>&& tensor_ptr, const OpType op) {
      tensor = std::move(tensor_ptr);
      op_type = op;
    }

    OpNode(OpNode<Tensor>&& other) {
      tensor = std::move(other.tensor);
      op_type = other.op_type;
    }

    OpNode<Tensor> operator=(OpNode<Tensor>&& other) {
      tensor = std::move(other.tensor);
      op_type = other.op_type;
    }

    ~OpNode() = default;

    std::unique_ptr<Tensor> tensor;
    // Tensor* tensor; // may or may not be empty

    // Need map since node could make a bunch of other different independent ones,
    // no limit to how many new nodes it could create via operations
    // std::unordered_map<OpNode*, OpType> next_nodes;
    std::vector<OpNode<Tensor>*> next_nodes;

    // Only need a vector since we know exactly how this node was made
    std::vector<OpNode<Tensor>*> prev_nodes;

    // What op this node was created with
    OpType op_type; 
    
    Tensor& get_tensor() { return *tensor; }
};


template<typename Tensor>
class RuntimeEngine{

public:
  RuntimeEngine() = default;
  ~RuntimeEngine() = default;

  std::unordered_map<std::string, OpNode<Tensor>> op_nodes;
  size_t num_tensors = 0;

  bool contains_name(const std::string& tensor_name) const {
    return tensor_name.empty() || op_nodes.find(tensor_name) != op_nodes.end();
  }

  std::string validate_name(const std::string& tensor_name) {
    std::string name;
    if (contains_name(tensor_name)) {
      std::stringstream name_stream;
      name_stream << "tensor_" << num_tensors;
      name = name_stream.str();
      num_tensors++;
    } else {
      std::cout << "Yeah I'm stupid" << std::endl;
      name = tensor_name;
    }
    if (name.empty()) {
      std::cout << "Yeah I'm stupid" << std::endl;
    }
    return name;
  }

  template<typename T, typename Container, typename Shape>
  Tensor& create_tensor(const Container& container, const Shape& shp, 
                        const bool requires_grad = false, const std::string& tensor_name = "") {
    std::string name = validate_name(tensor_name);
    auto op_node = OpNode(std::make_unique<Tensor>(container, shp, name, requires_grad), OpType::None);
    op_nodes.emplace(name, std::move(op_node));
    return *(op_nodes.at(name).tensor);
  }

  template<typename T>
  Tensor& create_tensor(std::unique_ptr<T[]>& ptr, const std::vector<size_t>& shp, const size_t sz,
                        const bool requires_grad = false, const std::string& tensor_name = "") {
    std::string name = validate_name(tensor_name);
    auto op_node = OpNode(std::make_unique<Tensor>(ptr, shp, sz, requires_grad, name), OpType::None);
    op_nodes.emplace(name, std::move(op_node));
    return *(op_nodes.at(name).tensor);
  }

  void clean() {
    op_nodes.clear();
    num_tensors = 0;
  }

};

static void* runtime_fp32 = nullptr;

template<typename Tensor, typename T>
RuntimeEngine<Tensor>* get_runtime() {
  if constexpr (std::is_same<T, float>::value) {
    if (runtime_fp32 == nullptr) {
      runtime_fp32 = new grad::RuntimeEngine<Tensor>;
      return static_cast<RuntimeEngine<Tensor>*>(runtime_fp32);
    } else {
      return static_cast<RuntimeEngine<Tensor>*>(runtime_fp32);
    }
  } else {
    return nullptr;
  }
}


/*
 * Compute the gradient of some tensor w.r.t to another tensor via reverse
 * mode auto-differentiation.
 *
 * Compute the gradient of the tensor at with a given name w.r.t all of the preceeding ops.
 *
 * Should return a vector of tensors
 */
template<typename Tensor, typename T>
std::vector<Tensor> backward(const std::string& name, RuntimeEngine<Tensor>* runtime) {
  // Compute the gradients for each step that produced the resulting tensor
  auto root = runtime->op_nodes.find(name);
  std::cout << "Found root tensor " << root->second.tensor->to_string() << std::endl;

  // TODO: Construct first grad in place
  std::vector<Tensor> grads;
  auto init_grad = Tensor::full(static_cast<T>(1), root->second.tensor->shape, false);
  grads.push_back(init_grad);
  std::unordered_map<OpNode<Tensor>*, size_t> inds;
  inds.emplace(&root->second, 0);

  // Search the graph in a way that preserves topological ordering, iterative DFS
  std::deque<OpNode<Tensor>*> queue;
  queue.push_back(&root->second);

  std::deque<OpNode<Tensor>*> sorted;
  std::unordered_map<OpNode<Tensor>*, int> has_dependency; // tracks all vars and their dependencies (edges in next_nodes)
  while (!queue.empty()) {
    auto cur_node = queue.back();

    if (cur_node->prev_nodes.size() == 0 && !has_dependency.contains(cur_node)) {
      // std::cout << "Base Case Name: " << cur_node->tensor->name << ", Value: " << cur_node->tensor->at(std::vector<size_t>{0}) << ", Op Type: " << cur_node->op_type << ", Num Prev: " << cur_node->prev_nodes.size() << ", Num Next: " << cur_node->next_nodes.size() << std::endl;
      sorted.push_front(cur_node); // prepend the current node
      has_dependency.insert({cur_node, -1}); 
      queue.pop_back();
    // To get rid of duplicates
    } else if (has_dependency.contains(cur_node)) {
      // TODO: Consider noexcept
      if (has_dependency.at(cur_node) > 0) {
        has_dependency.at(cur_node)--; // keep track of how many dependencies have been searched
      }
      if (has_dependency.at(cur_node) == 0) {
        // std::cout << "Non Base Case Name: " << cur_node->tensor->name << ", Value: " << cur_node->tensor->at(std::vector<size_t>{0}) << ", Op Type: " << cur_node->op_type << ", Num Prev: " << cur_node->prev_nodes.size() << ", Num Next: " << cur_node->next_nodes.size() << std::endl;
        sorted.push_front(cur_node); // prepend the current node
        has_dependency.at(cur_node)--;
      }
      queue.pop_back();

    } else {
      has_dependency.insert({cur_node, cur_node->next_nodes.size()}); 
      for (const auto& op : cur_node->prev_nodes) {
        queue.push_back(op);
      }
    }
  }
  sorted.pop_front(); // remove the output node

  for (const auto& cur_node : sorted) {
    std::cout << "Name: " << cur_node->tensor->name << ", Value: " << cur_node->tensor->at(std::vector<size_t>{0}) << ", Op Type: " << cur_node->op_type << std::endl;
  }
  std::cout << "\n";

  // Propagating gradients in reverse topo order
  // TODO: Figure out how to properly compute shapes
  size_t i = 1;
  for (const auto& cur_node : sorted) {
    std::cout << "Name: " << cur_node->tensor->name << ", Value: " << cur_node->tensor->at(std::vector<size_t>{0}) << ", Op Type: " << cur_node->op_type << std::endl;
    Tensor cur_grad;

    for (const auto& next_node : cur_node->next_nodes) {
      std::cout << "Child node name: " << next_node->tensor->name << ", Value: " << next_node->tensor->at(std::vector<size_t>{0}) << ", Op Type: " << next_node->op_type << std::endl;
      OpType next_op = next_node->op_type;
      Tensor local_grad;

      // Computing the local grad
      if (next_op == OpType::Add || next_op == OpType::Subtract) {
        // This means op_node created next_node via op_node + some_other_node = dx/dx + dy/dx
        // = 1 + 0 (even if y depends on x, it will be accounted for in isolation)

        if (next_op == OpType::Subtract && next_node->prev_nodes[1] == cur_node) {
          local_grad = Tensor::full(static_cast<T>(-1), next_node->tensor->shape, false);
        } else {
          local_grad = Tensor::full(static_cast<T>(1), next_node->tensor->shape, false);
        }
        // Multiplying by upstream grad via chain rule
        local_grad *= grads[inds.at(next_node)];

        // std::cout << "Add grad " << local_grad.to_string() << std::endl;

      } else if (next_op == OpType::Mult) {
        // For created next_nodes of the form cur_node * some_node = next_node, d(xy)/dx = y,
        // we assume that y is some constant
        // std::cout << "Computing mult grad" << std::endl;
        auto* other_node = next_node->prev_nodes[0] == cur_node ? next_node->prev_nodes[1] : next_node->prev_nodes[0];
        local_grad = grads[inds.at(next_node)] * other_node->get_tensor();
        //std::cout << "Finished computing mult grad" << std::endl;
      } else if (next_op == OpType::Log) {
        local_grad = grads[inds.at(next_node)] / cur_node->get_tensor();
        // std::cout << "Log local grad: " << local_grad.to_string() << std::endl;
      } else if (next_op == OpType::Sin) {
        local_grad = grads[inds.at(next_node)] * cur_node->get_tensor().cos();
        // std::cout << "Computing sin grad, cur: " << cur_node->tensor->to_string() << std::endl;
        // std::cout << inds.at(next_node) << std::endl;
        // std::cout << grads.size() << std::endl;
        // std::cout << grads[inds.at(next_node)].to_string() << std::endl;
        //Tensor local_grad = cur_node->get_tensor().cos();
        //std::cout << "Finished computing sin grad" << std::endl;
        //cur_grad *= local_grad;
        //std::cout << "Finished computing sin grad" << std::endl;
      } else {}

      if (cur_grad.is_empty()) {
        cur_grad = local_grad;
      } else {
        cur_grad += local_grad;
      }
      std::cout << "Computed grad: " << cur_grad.to_string() << std::endl;
    }
    grads.push_back(cur_grad);
    inds.emplace(cur_node, i);
    i++;
  }
  return grads;
}

/*
 * Compute the gradient of a tensor w.r.t to another tensor via forward mode
 * auto-differentiation.
 */
template<typename Tensor, typename T>
std::vector<Tensor> forward(const std::string& name, RuntimeEngine<Tensor>* runtime) {
  // Iterate through all the results from this op onwards, computing the gradients
  // at each step. 
  auto root = runtime->op_nodes.find(name);

  // TODO: Construct first grad in place
  std::vector<Tensor> grads;
  auto init_grad = Tensor::full(static_cast<T>(1), root->second.tensor->shape, false);
  grads.push_back(init_grad);

  std::unordered_map<OpNode<Tensor>*, size_t> inds;

  // Search the graph in a way that preserves topological ordering.
  // i.e x_1 --> x_2 --> x_3
  //      |               |
  //      -----------------
  // The above dependency graph can be resolved by computing the values in this
  // order: x_1, (x_2, x_3) --> since there is a hard dependency on x_2 and for x_3
  //
  // Otherwise without keeping this dependency order in mind, we could have simply went x_1, x_3, x_2,
  // evaluting x_1's children normally.
  //
  // The algorithm will greedily add a node's children in order.
  //
  // A child will only pop off the queue if it's dependencies have been resolved, otherwise it gets moved to the back.
  //
  // To determine the shape of the grad matrices/vectors, we assume that root is
  // simply some scalar (for now)
  std::deque<OpNode<Tensor>*> queue;
  queue.push_back(&root->second);

  // First, perform a DFS-based topo sort of the DAG
  // Some problems: We don't have complete set of all the "seed" nodes, which means we
  // can't resolve all of the dependencies. We would need to know which nodes are independent
  // of our root node of interest (i.e there exists no path between x_1 and x_2).
  // Thus it is necessary to compute all elements of the graph where there exists a path to
  // the root node, and then we observe the rest of elements of the expression and determine
  // if they fall into the following categories:
  // 1. All operands that produced the node depend on the input; proceed normally.
  // 2. None of the operands depend on the input, the grad is automatically 0 (represents 
  //    another input node in the graph (ex. f(x1, x2)).
  // 3. Atleast one of the operands depend on the input, then the others are 0
  //
  // This can be achieved by performing a lookup in a has_dependency set.
  std::deque<OpNode<Tensor>*> sorted;
  std::unordered_map<OpNode<Tensor>*, bool> has_dependency; // tracks all vars that depend on input
  while (!queue.empty()) {
    auto cur_node = queue.back();

    if (cur_node->next_nodes.size() == 0) {
      sorted.push_front(cur_node); // prepend the current node
      has_dependency.insert({cur_node, true}); 
      queue.pop_back();
    // To get rid of duplicates
    } else if (has_dependency.contains(cur_node)) {
      // TODO: Consider noexcept
      if (auto it = has_dependency.find(cur_node); it != has_dependency.end() && !it->second) {
        sorted.push_front(cur_node); // prepend the current node
        has_dependency.insert({cur_node, true});
      }
      queue.pop_back();

    } else {
      for (const auto& op : queue.back()->next_nodes) {
        // If this is not true, I know I have already added to sorted
        if (!has_dependency.contains(op)) {
          queue.push_back(op);
        }
      }
      if (has_dependency.contains(cur_node)) {
        queue.pop_back();
      } else {
        has_dependency.insert({cur_node, false});
      }
    }
  }

  // TODO: Use ranges::zip
  // Performs the gradient computation, starting from an input tensot
  size_t i = 0;
  for (const auto& cur_node : sorted) {
    std::cout << "Name: " << cur_node->tensor->name << " Value " << cur_node->tensor->at(std::vector<size_t>{0}) << std::endl;

    if (cur_node->op_type == OpType::None && cur_node != &root->second) {
      grads.emplace_back(Tensor::full(static_cast<T>(0), cur_node->tensor->shape));
    } else if (cur_node->op_type == OpType::Add || cur_node->op_type == OpType::Subtract) {
      // Since derivative is linear operator, d(v_1 + v_2)/dx = d(v_1)/dx + d(v_2)/dx -> sum of recorded grads
      OpNode<Tensor>* lhs = cur_node->prev_nodes[0];
      OpNode<Tensor>* rhs = cur_node->prev_nodes[1];
      std::cout << "Computing add grad, lhs is: " << lhs->tensor->to_string() << ", rhs is " << rhs->tensor->to_string() << std::endl;

      // Check if rhs has dependency on input, if not the rhs will be 0
      if (inds.find(rhs) == inds.end() || !has_dependency.contains(rhs)) {
        grads.push_back(grads[inds.find(lhs)->second]);
      }
      // Check if lhs has dependency on input, if not the lhs will be 0
      else if (inds.find(lhs) == inds.end() || !has_dependency.contains(lhs)) {
        grads.push_back(grads[inds.find(rhs)->second]);
      } 
      // Check if both don't have dependency, result is 0
      else if (!has_dependency.contains(lhs) && !has_dependency.contains(rhs)) {
        grads.push_back(Tensor::full(static_cast<T>(0), cur_node->tensor->shape, false));
      }
      // Check propagate grads normally
      else if (cur_node->op_type == OpType::Add){
        grads.emplace_back(grads[inds.find(lhs)->second] + grads[inds.find(rhs)->second]);
      } 
      else {
        grads.emplace_back(grads[inds.find(lhs)->second] - grads[inds.find(rhs)->second]);
      }

    } else if (cur_node->op_type == OpType::Log) {
      // d log(v_i)/dx = 1/v_1 * dv_i/dx --> second term already recorded, perform lookup
      std::cout << "Computing log grad, cur_node is: " << cur_node->tensor->to_string() << std::endl;
      std::cout << "Computing log grad, prev_node is: " << cur_node->prev_nodes[0]->tensor->to_string() << std::endl;
      Tensor grad = (1 / cur_node->prev_nodes[0]->get_tensor()) * grads[inds.at(cur_node->prev_nodes[0])];
      std::cout << "Log grad: " << grad.to_string() << std::endl;
      grads.push_back(grad);

    } else if (cur_node->op_type == OpType::Mult) {
      // Apply product rule, d v_1v_2/dx = dv_1/dx * v_2 + dv_2/dx * v3, dv_1 and dv_2 already recorded
      std::cout << "Computing mult grad, cur_node is: " << cur_node->tensor->to_string() << std::endl;
      OpNode<Tensor>* lhs = cur_node->prev_nodes[0];
      OpNode<Tensor>* rhs = cur_node->prev_nodes[1];

      // TODO: Reduce copies
      Tensor grad_lhs;
      Tensor grad_rhs;

      if (has_dependency.contains(lhs)) {
        grad_lhs = grads[inds.at(lhs)];
      } else {
        grad_lhs = Tensor::full(static_cast<T>(0), lhs->tensor->shape);
      }
      if (has_dependency.contains(rhs)) {
        grad_rhs = grads[inds.at(rhs)];
      } else {
        grad_rhs = Tensor::full(static_cast<T>(0), rhs->tensor->shape);
      }
      // std::cout << "Mult grad_lhs " << grad_lhs.to_string() << " " << grad_lhs.requires_grad << std::endl;
      // std::cout << "Mult grad_rhs " << grad_rhs.to_string() << " " << grad_rhs.requires_grad << std::endl;
      Tensor grad = grad_lhs * rhs->get_tensor() + grad_rhs * lhs->get_tensor();
      grads.push_back(grad);

    } else if (cur_node->op_type == OpType::Sin) {
      // d(sin(v_1))/dx = cos(x) * d(v_1)/dx
      OpNode<Tensor>* v1 = cur_node->prev_nodes[0];
      Tensor& grad_v1 = grads[inds.at(v1)];
      Tensor grad = v1->get_tensor().cos() * grad_v1;
      grads.push_back(grad);

    } else {
    }
    inds.emplace(cur_node, i);
    i++;
  }
  return grads;
}

template<typename Tensor>
static std::string register_tensor(const std::string& tensor_name, Tensor* tensor_ptr, RuntimeEngine<Tensor>* runtime) {
  std::string name;
  if (runtime->contains_name(tensor_name)) {
    std::stringstream name_stream;
    name_stream << "tensor_" << runtime->num_tensors;
    name = name_stream.str();
    runtime->num_tensors++;
  } else {
    std::cout << "Yeah I'm stupid" << std::endl;
    name = tensor_name;
  }
  if (name == "") {
    std::cout << "Yeah I'm stupid" << std::endl;
  }
  runtime->op_nodes.emplace(name, OpNode<Tensor>(tensor_ptr, OpType::None));
  return name;
}

/*
 * Adding a the result of an operation into our computation graph.
 * The nodes specified in tensor_names will now point to a new node that represents the result of the op.
 */
template<typename Tensor>
static void add_grad_edge(const std::vector<std::string>& tensor_names, OpType op_type, const std::string& res_name, RuntimeEngine<Tensor>* runtime) {
  OpNode<Tensor>& res_node = runtime->op_nodes.find(res_name)->second;
  // std::cout << "Adding edges to res_node: " << res_name << " " << res_node.tensor->to_string() << std::endl;
  res_node.op_type = op_type;

  for (const auto& name : tensor_names) {
    auto it = runtime->op_nodes.find(name);
    // std::cout << name << ", " << it->second.tensor->to_string() << " "<< "Is registered " << (it != runtime->op_nodes.end()) << std::endl;

    // Edge pointing from operand to result
    it->second.next_nodes.push_back(&res_node);

    // Edge pointing from result to operand node
    res_node.prev_nodes.push_back(&it->second);
  }
}


/* 
 * Class that is responsible for determining the proper gradient operation
 * given the "forward" op.
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
*/
} // namespace grad

template<typename T>
class Tensor {
/*
 * Numpy/Pytorch/Tensorflow-like ndarray class.
 */
public:
  std::vector<size_t> shape;
  size_t size;
  std::string name;
  bool requires_grad;
  grad::RuntimeEngine<Tensor<T>>* runtime;

protected:
  std::vector<size_t> strides;
  std::unique_ptr<T[]> data;

public:
  using dtype = T;

  ~Tensor() {
    //std::cout << "Destructed " << name << std::endl;
    //print();
  }

  Tensor() {
  }
  
  // Move constructor
  Tensor(Tensor<T>&& other) {
    shape = std::move(other.shape);
    strides = std::move(other.strides);
    size = other.size;
    data = std::move(other.data);
    requires_grad = other.requires_grad;
    runtime = other.runtime;
  }

  // Copy constructor
  Tensor(const Tensor<T>& other) {
    data = std::make_unique<T[]>(other.size);
    std::copy(other.data.get(), other.data.get() + other.size, data.get());
    strides = other.strides;
    size = other.size;
    requires_grad = other.requires_grad;
    runtime = grad::get_runtime<Tensor<T>, T>();
    //this->name = grad::register_tensor<Tensor<T>>("", this, runtime);
  }

  // Regular constructor
  Tensor(T* const arr, const std::vector<size_t>& shp, const size_t sz,
      const bool requires_grad = false, const std::string& name = "") {
    /*
     Strides - https://numpy.org/devdocs/reference/generated/numpy.ndarray.strides.html
     Numpy ndarrays Paper - https://arxiv.org/pdf/1102.1523.pdf
     */
    data = std::unique_ptr<T[]>(arr);
    shape = shp;
    size = sz;
    strides = get_strides(shape);
    runtime = grad::get_runtime<Tensor<T>, T>();
    this->requires_grad = requires_grad;

    // Setting name based on the current runtime's state
    // Potentially need to add a mutex for this as we don't want data race for
    // incrementing the num_tensors attr of the runtime engine
    //this->name = grad::register_tensor<Tensor<T>>(name, this, runtime);
  }

  Tensor(std::unique_ptr<T[]>& ptr, const std::vector<size_t>& shp, const size_t sz,
         const bool requires_grad = false, const std::string& name = "") {
    data = nullptr;
    data = std::move(ptr);
    shape = shp;
    size = sz;
    strides = get_strides(shape);
    this->requires_grad = requires_grad;
    runtime = grad::get_runtime<Tensor<T>, T>();
    this->name = name;
    //this->name = grad::register_tensor<Tensor<T>>(name, this, runtime);
  }

  template<typename Container, typename Shape>
  Tensor(const Container& container, const Shape& shp, 
         const std::string& name, const bool requires_grad = false) {
    data = std::make_unique<T[]>(container.size());
    std::copy(container.begin(), container.end(), data.get());

    if (shp.empty()) {
      size = 0;
    } else {
      size = 1;
    }

    for (const auto& i : shp) {
      size *= i;
    }
    shape.resize(size);

    // Copying shape list
    std::copy(shp.begin(), shp.end(), shape.begin());
    strides = get_strides(shape);
    this->requires_grad = requires_grad;
    runtime = grad::get_runtime<Tensor<T>, T>();
    this->name = name;
    //this->name = grad::register_tensor<Tensor<T>>(name, this, runtime);
  }

  template<class Container>
  static Tensor<T>& arange(int start, int stop, const Container& new_shape, const bool requires_grad = false, const std::string& tensor_name = "") {
    /*
     Initialize a tensor with shape full of val. Returns the Tensor object itself.
     */
    size_t new_size = stop - start;
    std::vector<size_t> shp(new_shape.size(), 0);

    std::transform(new_shape.begin(), new_shape.end(), shp.begin(),
        [&new_size](auto shape_entry) {
          return static_cast<size_t>(shape_entry); 
        }
    );

    // Creating a contiguous C-style array
    auto arr = std::make_unique<T[]>(new_size);

    int val = start;
    for (size_t i = 0; i < new_size; i++) {
      arr[i] = val++;
    }


    return grad::get_runtime<Tensor<T>, T>()->template create_tensor(arr, shp, new_size, requires_grad, tensor_name);
  }

  template<class Container>
  static Tensor<T>& full(T val, const Container& new_shape, const bool requires_grad = false, const std::string& tensor_name = "") {
    /*
     Initialize a tensor with shape full of val. Returns the Tensor object itself.
     */
    size_t new_size = 1;
    std::vector<size_t> shp(new_shape.size(), 0);

    std::transform(new_shape.begin(), new_shape.end(), shp.begin(),
        [&new_size](auto shape_entry) {
          new_size *= shape_entry; 
          return static_cast<size_t>(shape_entry); 
        }
    );

    // Creating a contiguous C-style array
    auto arr = std::make_unique<T[]>(new_size);
    std::fill_n(arr.get(), new_size, val);


    return grad::get_runtime<Tensor<T>, T>()->template create_tensor(arr, shp, new_size, requires_grad, tensor_name);
  }

  static Tensor<T>& full(T val, const std::initializer_list<size_t>& new_shape, const bool requires_grad = false, const std::string& tensor_name = "") {
    /*
     Initialize a tensor with shape full of val. Returns the Tensor object itself.
     */
    size_t new_size = 1;
    std::vector<size_t> shp(new_shape.size(), 0);

    std::transform(new_shape.begin(), new_shape.end(), shp.begin(),
        [&new_size](auto shape_entry) {
          new_size *= shape_entry; 
          return static_cast<size_t>(shape_entry); 
        }
    );

    // Creating a contiguous C-style array
    auto arr = std::make_unique<T[]>(new_size);
    std::fill_n(arr.get(), new_size, val);

    return grad::get_runtime<Tensor<T>, T>()->template create_tensor(arr, shp, new_size, requires_grad, tensor_name);
  }

  template<class Container>
  static Tensor<T>& empty(const Container& new_shape, const bool requires_grad = false, const std::string& tensor_name = "") {
    /*
     * Default initialize an empty tensor with shape.
     */
    size_t new_size = 1;
    std::vector<size_t> shp(new_shape.size(), 0);

    // Writing the shape into out shape vector
    std::transform(new_shape.begin(), new_shape.end(), shp.begin(),
        [&new_size](auto shape_entry) {
          new_size *= shape_entry; 
          return static_cast<size_t>(shape_entry); 
        }
    );

    // Creating a contiguous C-style array
    auto arr = std::make_unique<T[]>(new_size);

    return grad::get_runtime<Tensor<T>, T>()->template create_tensor(arr, shp, new_size, requires_grad, tensor_name);
  }

  static Tensor<T> empty(const std::initializer_list<size_t>& new_shape, const bool requires_grad = false) {
    /*
     * Default initialize an empty tensor with shape.
     */
    size_t new_size = 1;
    std::vector<size_t> shp(new_shape.size(), 0);

    // Writing the shape into out shape vector
    std::transform(new_shape.begin(), new_shape.end(), shp.begin(),
        [&new_size](auto shape_entry) {
          new_size *= shape_entry; 
          return static_cast<size_t>(shape_entry); 
        }
    );

    // Creating a contiguous C-style array
    auto arr = std::make_unique<T[]>(new_size);

    Tensor<T> tensor = Tensor<T>(arr, shp, new_size, requires_grad);
    return tensor;
  }

  Tensor<T>& copy() const {
    /*
     * Default initialize an empty tensor with shape.
     */
    auto new_data = std::make_unique<T[]>(size);
    std::copy(data.get(), data.get() + size, new_data.get());
    return grad::get_runtime<Tensor<T>, T>()->template create_tensor(new_data, shape, size, requires_grad);
  }

  template<class Container>
  static Tensor<T>& he_normal(const Container& new_shape, const bool requires_grad = false, const std::string& tensor_name = "") {
    /*
     Randomly initialize a tensor according to "Delving Deep into Rectifiers:
     Surpassing Human-Level Performance on ImageNet Classification" (He et. al 2015).
     Returns reference to the Tensor object.
     */
    size_t new_size = 1;
    std::vector<size_t> shp(new_shape.size(), 0);

    std::transform(new_shape.begin(), new_shape.end(), shp.begin(),
        [&new_size](auto shape_entry) {
          new_size *= shape_entry; 
          return static_cast<size_t>(shape_entry); 
        }
    );

    // Initialize a normal distribution with mean 0 and stddev sqrt(1 / fan_in),
    // where fan_in is the number of input units to the weight matrix
    std::random_device rd;
    std::normal_distribution<T> normal(0.0, std::sqrt(2.L / new_shape[0]));
    auto arr = std::make_unique<T[]>(new_size);
    std::generate_n(arr.get(), new_size, [&normal, &rd]() {
        return normal(rd);
    });

    return grad::get_runtime<Tensor<T>, T>()->template create_tensor(arr, shp, new_size, requires_grad, tensor_name);
  }

  std::vector<size_t> get_strides(const std::vector<size_t>& shp) {
    /*
     * Strides describe how to traverse through the internal array in the proper order
     * described by the shape of the tensor.
     *
     * This is expecially important with reshaping operations like transposes, where the left-right
     * order of the entries as they appear in something like a matrix changes (i.e A[i, j] = A.transpose()[j, i])
     * but the internal array does not. This way we can properly keep track of changing shapes and
     * traverse our tensors correctly.
     *
     * All that should happen to strides when a reshape happens is the strides should get shuffled
     * to their new spots.
     *
     * Ex. a 9 x 9 matrix will have strides (24, 8) while its transpose will have strides (8, 24)
     */
    std::vector<size_t> strd = std::vector<size_t>(shp.size());
    size_t offset = 1;

    // Look into this, if loop variable is size_t the unsigned ll underflows
    for (int i = strd.size() - 1; i >= 0; i--) {
      strd[i] = offset;
      offset *= shp[i];
    }
    return strd;
  }

  template<typename Container>
  T at(const Container& inds) {
    int ind = 0;

    for (size_t i = 0; i < strides.size(); i++) {
      ind += inds[i] * strides[i];
    }

    return *(data.get() + ind);
  }

  void assign(size_t ind, T val) {
    data[ind] = val;
  }

  bool is_empty() const {
    return data == nullptr;
  }

  void _matmul_2D(const Tensor<T>& B, const std::unique_ptr<T[]>& startP, 
      const std::unique_ptr<T[]>& startA, const std::unique_ptr<T[]>& startB, const size_t offset) {
    /*
     i - # of rows of this
     j - # of cols of this/ # of rows of B
     k - # of cols of B

     Must be invoked on Tensors with dimensions (M, N) and (N, P)
     to create a matrix with dimensions (M, P)
     */

    size_t row = B.shape.size() - 2;
    size_t col = B.shape.size() - 1;

    for (size_t i = 0; i < (this->shape)[row]; i++) {
      for (size_t k = 0; k < B.shape[col]; k++) {
        T prod_ik = T(0);

        for (size_t j = 0; j < B.shape[row]; j++) {
          T a_ij = startA[i * (this->strides)[row] + j * (this->strides)[col] + offset];
          T b_jk = startB[j * B.strides[row] + k * B.strides[col] + offset];
          prod_ik += a_ij * b_jk;
        }
        startP[i * (this->strides)[row] + k * B.strides[col] + offset] = prod_ik;
      }
    }
  }

  void _matmul_3D(const Tensor<T>& B, const std::unique_ptr<T[]>& startP, 
      const std::unique_ptr<T[]>& startA, const std::unique_ptr<T[]>& startB, 
      const std::vector<size_t>& shp, const std::vector<size_t>& strd, const size_t offset) {
    /*
     * Helper for performing a matmul between 3D tensors. 
     */
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

  void _matmul_4D(const Tensor<T>& B, const std::unique_ptr<T[]>& startP,
      const std::unique_ptr<T[]>& startA, const std::unique_ptr<T[]>& startB,
      const std::vector<size_t>& shp, const std::vector<size_t>& strd) {
    /*
     * Helper for performing a matmul between 4D tensors. 
     */
    for (int d = 0; d < shp[0]; d++) {
      (this->_matmul_3D)(B, startP, startA, startB, shp, strd, d * strd[0]);
    }
  }

  Tensor<T>& matmul(const Tensor<T>& B) {
    /*
     * Performs a standard matrix multiply between two tensors. Returns a reference to
     * a newly allocated tensor.
     *
     * Note: The use of make_unique for an array only gained support in C++20
     *
     * Treats the last two trailing dimensions as the shape of the matrices, and
     * the rest of the dimensions as how many of stacks of them we have.
     *
     * Strassen's algorithm for tensor contraction - https://arxiv.org/pdf/1704.03092.pdf
     *
     */
    size_t total_size = 1;

    std::vector<size_t> new_shape = std::vector<size_t>(B.shape.size());
    for (size_t i = 0; i < B.shape.size() - 2; i++) {
      new_shape[i] = B.shape[i];
      total_size *= B.shape[i];
    }

    size_t num_mats = total_size;

    // Rows of result is rows of lhs, cols is rhs
    std::cout << "A num dims: " << shape.size() << " " << "B dims: " << B.shape.size() << std::endl;
    size_t rows = this->shape[B.shape.size() - 2];
    size_t cols = B.shape[B.shape.size() - 1];
    size_t mat_size = rows * cols;
    std::cout << "Res rows: " << rows << " Res cols: " << cols << std::endl;

    total_size *= rows * cols;

    new_shape[B.shape.size() - 2] = rows;
    new_shape[B.shape.size() - 1] = cols;

    std::vector<size_t> new_strides = Tensor::get_strides(new_shape);

    auto prod = std::make_unique<T[]>(total_size);

    for (size_t i = 0; i < num_mats; i++) {
      _matmul_2D(B, prod, this->data, B.data, i * mat_size);
    }

    return grad::get_runtime<Tensor<T>, T>()->template create_tensor(prod, new_shape, total_size);;
  }

  template<class Container>
  Tensor<T>& reshape(const Container& shp) {
    /*
     * Numpy-like reshaping method. Add support for the -1 syntax, infer the shape
     * based on the rest of the entries
     */
    size_t new_size = 0;
    std::vector<size_t> new_shape = std::vector<size_t>(shp.size());

    int infer = -1;
    int ind = 0;
    for (const auto &i: shp) {
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
    //assert(new_size == size);

    shape = new_shape;
    strides = get_strides(shape);

    return *this;
  }

  template<class C>
  Tensor<T>& transpose() {
    /*
     * Perform a transpose on a 2 dimensional matrix, swapping the two trailing dimensions
     * along with their corresponding strides.
     */
    // Swapping dimensions
    size_t temp = shape[shape.size() - 1];
    shape[shape.size() - 1] = shape[shape.size() - 2];
    shape[shape.size() - 2] = temp;

    // Swapping strides
    temp = strides[strides.size() - 1];
    strides[strides.size() - 1] = strides[strides.size() - 2];
    strides[strides.size() - 2] = temp;

    return *this;
  }

  std::vector<Tensor<T>> backward() {
    return grad::backward<Tensor<T>, T>(name, runtime);
  }

  std::vector<Tensor<T>> forward() {
    return grad::forward<Tensor<T>, T>(name, runtime);
  }

  /*
   * Overloaded arithmetic operators. Future work would implement broadcasting here
   * but frankly that seems unnecessary at this point.
   *
   * TODO: 
   *    - Exponentiation
   *    - Rvalue operators (big perf boost)
   */
  Tensor<T>& operator+=(const Tensor<T>& rhs) {
    for (size_t i = 0; i < size; i++) {
      data[i] += rhs.data[i];
    }
    return *this;
  }

  friend Tensor<T>& operator+(const Tensor<T>& lhs, const Tensor<T>& rhs) {
    Tensor<T>& res = full<std::vector<size_t>>(static_cast<T>(0), lhs.shape, lhs.requires_grad && rhs.requires_grad);
    for (size_t i = 0; i < lhs.size; i++) {
      res.data[i] = lhs.data[i] + rhs.data[i];
    }

    /*
    // Find quickest changing dimension
    int change_dim;
    for (int i = lhs.strides.size() - 1; i >= 0; i--) {
      if (lhs.strides[i] == 1) {
        change_dim = i;
        break;
      }
    }
    */

    if (lhs.requires_grad && rhs.requires_grad) {
      grad::add_grad_edge(std::vector<std::string>{lhs.name, rhs.name}, grad::OpType::Add, res.name, lhs.runtime);
    }
    return res;
  }

  Tensor<T>& operator-=(const Tensor<T>& rhs) {
    //assert(size == rhs.size);

    for (size_t i = 0; i < size; i++) {
      data[i] -= rhs.data[i];
    }
    return *this;
  }

  Tensor<T>& operator-(const Tensor<T>& rhs) {
    //assert(size == rhs.size);

    Tensor<T>& res = full(static_cast<T>(0), shape);

    for (size_t i = 0; i < size; i++) {
      res.data[i] = data[i] - rhs.data[i];
    }

    if (requires_grad && rhs.requires_grad) {
      grad::add_grad_edge(std::vector<std::string>{name, rhs.name}, grad::OpType::Subtract, res.name, runtime);
    }
    return res;
  }

  friend Tensor<T>& operator-(T lhs, const Tensor<T>& rhs) {
    Tensor<T>& lhs_tensor = full(static_cast<T>(0), std::vector<size_t>{1, 1});
    Tensor<T>& res = full(static_cast<T>(0), rhs.shape);

    for (size_t i = 0; i < rhs.size; i++) {
      res.data[i] = lhs - rhs.data[i];
    }

    if (rhs.requires_grad) {
      grad::add_grad_edge(std::vector<std::string>{lhs_tensor.name, rhs.name}, grad::OpType::Subtract, res.name, rhs.runtime);
    }
    return res;
  }

  Tensor<T>& operator/=(const Tensor<T>& rhs) {
    //assert(size == rhs.size);

    for (size_t i = 0; i < size; i++) {
      data[i] /= rhs.data[i];
    }
    return *this;
  }

  Tensor<T> operator/(const Tensor<T>& rhs) {
    //assert(size == rhs.size);

    Tensor<T>& res = full(static_cast<T>(0), shape);

    for (size_t i = 0; i < size; i++) {
      res.data[i] = data[i] / rhs.data[i];
    }

    if (requires_grad && rhs.requires_grad) {
      grad::add_grad_edge(std::vector<std::string>{name, rhs.name}, grad::OpType::Div, res.name, runtime);
    }
    return res;
  }

  friend Tensor<T> operator/(const T lhs, const Tensor<T>& rhs) {
    //assert(size == rhs.size);

    Tensor<T> res = full(static_cast<T>(0), rhs.shape);

    for (size_t i = 0; i < rhs.size; i++) {
      res.data[i] = lhs / rhs.data[i];
    }

    if (rhs.requires_grad) {
      //grad::add_grad_edge(std::vector<std::string>{res.name, rhs.name}, grad::OpType::Div, res.name, rhs.runtime);
    }
    return res;
  }

  Tensor<T>& operator*=(const Tensor<T>& rhs) {
    //assert(size == rhs.size);

    for (size_t i = 0; i < size; i++) {
      data[i] *= rhs.data[i];
    }
    return *this;
  }

  // Tensor<T> operator*(const Tensor<T>& rhs) {
  friend Tensor<T>& operator*(const Tensor<T>& lhs, const Tensor<T>& rhs) {
    //assert(size == rhs.size);

    // std::cout << "lhs " << lhs.to_string() <<  " requires_grad " << lhs.requires_grad << std::endl;
    // std::cout << "rhs " << rhs.to_string() << " requires_grad " << rhs.requires_grad << std::endl;
    Tensor<T>& res = full(static_cast<T>(0), lhs.shape, lhs.requires_grad && rhs.requires_grad);
    std::cout << "Mult operator res name: " << res.name << std::endl;

    for (size_t i = 0; i < rhs.size; i++) {
      res.data[i] = lhs.data[i] * rhs.data[i];
    }

    if (lhs.requires_grad && rhs.requires_grad) {
      grad::add_grad_edge(std::vector<std::string>{lhs.name, rhs.name}, grad::OpType::Mult, res.name, rhs.runtime);
    }
    return res;
  }

  T max() {
    T mx = data[0];

    for (int i = 0; i < size; i++) {
      mx = (data[i] > mx) ? data[i] : mx;
    }
    return mx;
  }

  T min() {
    T mn = data[0];

    for (int i = 0; i < size; i++) {
      mn = (data[i] < mn) ? data[i] : mn;
    }
    return mn;
  }

  Tensor<T>& sin() {
    Tensor<T>& result = Tensor::empty(shape, requires_grad);

    std::transform(data.get(), data.get() + size, result.data.get(),
        [](T item) { return static_cast<T>(std::sin(item)); }
    );

    if (requires_grad) {
      // &res should be ok because of RVO
      grad::add_grad_edge(std::vector<std::string>{name}, grad::OpType::Sin, result.name, runtime);
    }
    return result;
  }

  Tensor<T>& cos() {
    Tensor<T>& result = Tensor::empty(shape, requires_grad);

    std::transform(data.get(), data.get() + size, result.data.get(),
        [](T item) { return static_cast<T>(std::cos(item)); }
    );

    if (requires_grad) {
      //grad::add_grad_edge(std::vector<std::string>{name}, grad::OpType::Cos, result.name, runtime);
    }
    return result;
  }

  Tensor<T>& log(double base = std::numbers::e_v<T>) {
    Tensor<T>& result = Tensor::empty(shape, requires_grad);
    // std::cout << "Log res requires_grad: " << result.requires_grad << std::endl;

    std::transform(data.get(), data.get() + size, result.data.get(),
        [base](T item) { return static_cast<T>(std::log(item) / std::log(base)); }
    );

    if (requires_grad) {
      grad::add_grad_edge(std::vector<std::string>{name}, grad::OpType::Log, result.name, runtime);
    }

    return result;
  }

  Tensor<T>& relu() {
    Tensor<T>& result = Tensor::empty(shape, requires_grad);

    std::transform(data.get(), data.get() + size, result.data.get(),
        [](T item) { return item > 0 ? item : static_cast<T>(0); }
    );

    if (requires_grad) {
      grad::add_grad_edge(std::vector<std::string>{name}, grad::OpType::ReLU, result.name, runtime);
    }

    return result;
  }

  Tensor<T>& sigmoid() {
    Tensor<T>& result = Tensor::empty(shape, requires_grad);

    std::transform(data.get(), data.get() + size, result.data.get(),
        [](T item) { return 1 / (1 + std::exp(-item)); }
    );

    if (requires_grad) {
      grad::add_grad_edge(std::vector<std::string>{name}, grad::OpType::Sigmoid, result.name, runtime);
    }

    return result;
  }


  Tensor<T>& sum() {
    Tensor<T>& result = Tensor::empty(std::vector<size_t>{1}, requires_grad);

    T val = std::accumulate(data.get(), data.get() + size, static_cast<T>(0));

    result.assign(0, val);

    if (requires_grad) {
      grad::add_grad_edge(std::vector<std::string>{name}, grad::OpType::Sum, result.name, runtime);
    }

    return result;
  }

  /*
   * Move assignment operator
   */
  Tensor<T>& operator=(Tensor<T>&& rhs) {
    shape = std::move(rhs.shape);
    strides = std::move(rhs.strides);
    size = rhs.size;
    data = std::move(rhs.data);
    name = std::move(rhs.name);
    requires_grad = rhs.requires_grad;
    return *this;
  }

  /*
   * Copy assignment operator
   */
  Tensor<T>& operator=(const Tensor<T>& rhs) {
    shape = rhs.shape;
    strides = rhs.strides;
    size = rhs.size;
    data = std::make_unique<T[]>(rhs.size);
    std::copy(rhs.data.get(), rhs.data.get() + rhs.size, data.get());
    requires_grad = rhs.requires_grad;
    name = rhs.name;
    return *this;
  }

  std::string to_string() const {
    std::ostringstream out_string;
    out_string << "[";
    for (size_t i = 0; i < size; i++) {
      out_string << data[i] << ", ";
    }
    out_string << "\b\b"; // get rid of extra whitespace
    out_string << "]";

    return out_string.str();
  }

  void print() const {
    std::cout << to_string() << std::endl;
  }

};

template<typename T, typename Container, typename Shape>
Tensor<T>& tensor(const Container& container, const Shape& shp, 
                 const bool requires_grad = false, const std::string& name = "") {
  auto runtime = grad::get_runtime<Tensor<T>, T>();
  return runtime->template create_tensor<T>(container, shp, requires_grad, name);
}

template<typename T>
Tensor<T>& bce(Tensor<T>& y_pred, const Tensor<T>& y_true, const std::string& reduction = "sum") {
  /*
   * BCE(y', y) = sum(y * ln(y') + (1 - y) * ln(1 - y'))
   */
  bool y_pred_grad = y_pred.requires_grad;

  if (y_pred_grad) {
    y_pred.requires_grad = false;
  }

  Tensor<T> result = y_pred.log(); //+ (1 - y_true) * (1 - y_pred).log();
  std::cout << "BCE : " << result.to_string() << std::endl;
  if (!reduction.empty()) {
    result = result.sum();
  }
  //result *= y_pred.log();

  //Tensor<T>& prob_false = (1 - y_true);
  //prob_false *= (1 - y_pred).log();
  //result += prob_false;

  //result = result.sum();

  if (y_pred_grad) {
    y_pred.requires_grad = true;
  }

  if (y_pred.requires_grad) {
    grad::add_grad_edge(std::vector<std::string>{y_pred.name, y_true.name}, grad::OpType::BinaryCrossEntropy, result.name, y_pred.runtime);
  }

  return result;
}

template<typename T>
constexpr std::vector<size_t> broadcasted_shape(const Tensor<T>& lhs, const Tensor<T>& rhs) {
  static_assert(lhs.size == rhs.size);

  // Just need to satisfy two rules as we iterate through the dimensions
  // 1. The dimensions match
  // 2. Atleast one of them is 1
  int lhs_ind = lhs.shape.size();
  int rhs_ind = rhs.shape.size();

  std::vector<size_t> lhs_ones_inds;
  std::vector<size_t> rhs_ones_inds;

  std::vector<size_t> new_shape;

  while (lhs_ind >= 0 && rhs_ind >= 0) {
    if (lhs.shape[lhs_ind] != rhs.shape[lhs_ind] && (lhs.shape[lhs_ind] != 1 || rhs.shape != 1)) {
      return std::vector<size_t>{};
    } else if (lhs.shape[lhs_ind] != 1 && rhs.shape[rhs_ind] == 1) {
      rhs_ones_inds.push_back(rhs_ind);
      new_shape.push_back(lhs.shape[lhs_ind]);
    } else if (lhs.shape[lhs_ind] == 1 && rhs.shape[rhs_ind] != 1) {
      lhs_ones_inds.push_back(lhs_ind);
      new_shape.push_back(rhs.shape[rhs_ind]);
    }

    rhs_ind--; 
    lhs_ind--;
  }

  return new_shape;
}

} // namespace scratchnn
