# scratchnn

### Purpose
scratchnn is a DL framework written in C++ that tries to simplify things as much as possible,
while also trying its best to be mildly performant.

It has full implementations of:

* Autodiff
* Tensors
* High-level Neural Net API including:
    * Dense layers
    * Attention layers (for transformers)
    * Convolutional layers
    * Arithmetic layers (i.e add)
    * Concatentation layers
* Gradient-based optimization methods
* Data parallelism
* Traditional ML algorithms (i.e SVMs)

I found that using Tensorflow and Keras wasn't enough to get the concepts of DL
into my thick skull, so I decided to try and make a DL framework myself. The
main goals are to educate myself and also provide a useful resource for
other people who share my position. I encourage you to follow along in the
```
./notebooks
```
directory for explanations, extra resources and to implement everything
here yourself! Or simply read the C++ code.

IB: [karpathy/micrograd](https://github.com/karpathy/micrograd), [geohot/tinygrad](https://github.com/geohot/tinygrad)


### TODO
- [X] Fix matmul code
    - [X] Calculating the shapes at each level
    - [X] Calculating the strides at each level
- [X] Overload basic arithmetic operators
    - [X] Add support for ufuncs
- [X] Views of tensors
    - [X] Reshaping
    - [ ] Shape permuting (like torch), use variadic template
    - 
- [X] Look into overloading the assignment operator
- [ ] Template for sizes? Very annoying to have to make it size_t everytime (backlog this)
    - [ ] Maybe make a wrapper over vector and just cast it to size_t or something
- [ ] Basic activation functions
- [ ] Autograd engine + integrate grad functions with tensors
- [ ] FC Layer class
    - [ ] Weights + biases
    - [ ] Ability to easily adjust weights
- [ ] Optimizers
    - [ ] SGD
    - [ ] Adam
- [ ] Transformers
- [ ] Convolutional layers
- [ ] VAEs
- [ ] RNNs
- [ ] Probabilistic Stuff
- [ ] Logistic Regression
- [ ] Linear Regression
- [ ] SVM
- [ ] Tensor Optimizations
    - [ ] Parallel ufuncs for large tensors. Add benchmarks against single threaded
    - [ ] Multithreaded matmuls
- [ ] Make RuntimeEngine thread-safe
- [ ] Network parallelism (computing paths of the graph in parallel, asynchronously)

### Questions You'll Probably Ask
1. Why don't I just use Tensorflow/Pytorch/JAX/whatever?
    Yes.
