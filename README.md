# scratchnn

### Purpose
scratchnn is a DL framework written in C++ that tries to simplify things as much as possible,
while also trying to keep performance in mind.

It has full implementations of:

* Autodiff
* Tensors
* High-level Neural Net API including:
    * Dense layers
    * Convolutional layers
    * Arithmetic layers (i.e add)
    * Concatentation layers
* Gradient-based optimization methods
* Traditional ML algorithms (i.e SVMs)

I found that using Tensorflow and Keras wasn't enough to get the concepts of DL
into my thick skull, so I decided to try and make a DL framework myself. The
main goals are to educate myself and also provide a useful resource for
other people in who share my position. I encourage you to follow along in the
```
./notebooks
```
directory for explanations, extra resources and to implement everything
here yourself! 

IB: [karpathy/micrograd](https://github.com/karpathy/micrograd), [geohot/tinygrad](https://github.com/geohot/tinygrad)


### TODO
- [X] Fix matmul code
    - [X] Calculating the shapes at each level
    - [X] Calculating the strides at each level
- [ ] Overload basic arithmetic operators
    - [ ] Add support for ufuncs
- [ ] Views of tensors
    - [ ] Reshaping
- [ ] Look into overloading the assignment operator
- [ ] Template for sizes? Very annoying to have to make it size_t everytime
    - [ ] Maybe make a wrapper over vector and just cast it to size_t or something
- [ ] Basic activation functions
- [ ] Autograd engine + integrate grad functions with tensors
- [ ] FC Layer class
    - [ ] Weights + biases
    - [ ] Ability to easily adjust weights
- [ ] Optimizers
    - [ ] SGD
    - [ ] Adam
    - [ ] RMSProp
- [ ] Convolutional layers
- [ ] RNNs
- [ ] VAEs
- [ ] Transformers
- [ ] Logistic Regression
- [ ] Linear Regression
- [ ] SVM

### Questions You'll Probably Ask
1. Why don't I just use Tensorflow/Pytorch/JAX/whatever?
    Yes.
