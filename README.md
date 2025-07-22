# MLX

[**Quickstart**](#quickstart) | [**Installation**](#installation) |
[**Documentation**](https://ml-explore.github.io/mlx/build/html/index.html) |
[**Examples**](#examples) 

[![CircleCI](https://circleci.com/gh/ml-explore/mlx.svg?style=svg)](https://circleci.com/gh/ml-explore/mlx)

MLX is an array framework for machine learning on Apple silicon,
brought to you by Apple machine learning research.

Some key features of MLX include:

 - **Familiar APIs**: MLX has a Python API that closely follows NumPy.  MLX
   also has fully featured C++, [C](https://github.com/ml-explore/mlx-c), and
   [Swift](https://github.com/ml-explore/mlx-swift/) APIs, which closely mirror
   the Python API.  MLX has higher-level packages like `mlx.nn` and
   `mlx.optimizers` with APIs that closely follow PyTorch to simplify building
   more complex models.

 - **Composable function transformations**: MLX supports composable function
   transformations for automatic differentiation, automatic vectorization,
   and computation graph optimization.

 - **Lazy computation**: Computations in MLX are lazy. Arrays are only
   materialized when needed.

 - **Dynamic graph construction**: Computation graphs in MLX are constructed
   dynamically. Changing the shapes of function arguments does not trigger
   slow compilations, and debugging is simple and intuitive.

 - **Multi-device**: Operations can run on any of the supported devices
   (currently the CPU and the GPU).

 - **Unified memory**: A notable difference from MLX and other frameworks
   is the *unified memory model*. Arrays in MLX live in shared memory.
   Operations on MLX arrays can be performed on any of the supported
   device types without transferring data.

MLX is designed by machine learning researchers for machine learning
researchers. The framework is intended to be user-friendly, but still efficient
to train and deploy models. The design of the framework itself is also
conceptually simple. We intend to make it easy for researchers to extend and
improve MLX with the goal of quickly exploring new ideas. 

The design of MLX is inspired by frameworks like
[NumPy](https://numpy.org/doc/stable/index.html),
[PyTorch](https://pytorch.org/), [Jax](https://github.com/google/jax), and
[ArrayFire](https://arrayfire.org/).

## Examples

The [MLX examples repo](https://github.com/ml-explore/mlx-examples) has a
variety of examples, including:

- [Transformer language model](https://github.com/ml-explore/mlx-examples/tree/main/transformer_lm) training.
- Large-scale text generation with
  [LLaMA](https://github.com/ml-explore/mlx-examples/tree/main/llms/llama) and
  finetuning with [LoRA](https://github.com/ml-explore/mlx-examples/tree/main/lora).
- Generating images with [Stable Diffusion](https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion).
- Speech recognition with [OpenAI's Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper).

## Quickstart

See the [quick start
guide](https://ml-explore.github.io/mlx/build/html/usage/quick_start.html)
in the documentation.

## Installation

MLX is available on [PyPI](https://pypi.org/project/mlx/). To install the Python API, run:

**With `pip`**:

```
pip install mlx
```

**With `conda`**:

```
conda install -c conda-forge mlx
```

Checkout the
[documentation](https://ml-explore.github.io/mlx/build/html/install.html#)
for more information on building the C++ and Python APIs from source.

## Contributing 

Check out the [contribution guidelines](https://github.com/ml-explore/mlx/tree/main/CONTRIBUTING.md) for more information
on contributing to MLX. See the
[docs](https://ml-explore.github.io/mlx/build/html/install.html) for more
information on building from source, and running tests.

We are grateful for all of [our
contributors](https://github.com/ml-explore/mlx/tree/main/ACKNOWLEDGMENTS.md#Individual-Contributors). If you contribute
to MLX and wish to be acknowledged, please add your name to the list in your
pull request.

## Citing MLX

The MLX software suite was initially developed with equal contribution by Awni
Hannun, Jagrit Digani, Angelos Katharopoulos, and Ronan Collobert. If you find
MLX useful in your research and wish to cite it, please use the following
BibTex entry:

```
@software{mlx2023,
  author = {Awni Hannun and Jagrit Digani and Angelos Katharopoulos and Ronan Collobert},
  title = {{MLX}: Efficient and flexible machine learning on Apple silicon},
  url = {https://github.com/ml-explore},
  version = {0.0},
  year = {2023},
}
```


---

## MLX-CUDA Backend Testing Notes

**Note:** The CUDA backend is currently experimental and not yet stable. Many features are still under active development, and you may encounter issues or unimplemented operations.

This document summarizes the steps taken to build and test the MLX library with the experimental CUDA backend.

### Machine Specifications

*   **Operating System:** Ubuntu 22.04
*   **CPU:** Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz
*   **GPU:** Tesla V100-DGXS-16GB
*   **CUDA Version:** 12.8

### Build Process

The following steps were taken to successfully build the MLX library with CUDA support:

1.  **Initial Configuration:**
    ```bash
    cmake . -Bbuild -DMLX_BUILD_CUDA=ON -DMLX_BUILD_EXAMPLES=ON
    ```
    This failed due to a missing CUDA architecture definition.

2.  **Specify CUDA Compiler:**
    ```bash
    cmake . -Bbuild -DMLX_BUILD_CUDA=ON -DMLX_BUILD_EXAMPLES=ON -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
    ```
    This failed due to a missing BLAS library.

3.  **Build BLAS from Source (Successful Configuration):**
    ```bash
    cmake . -Bbuild -DMLX_BUILD_CUDA=ON -DMLX_BUILD_EXAMPLES=ON -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DMLX_BUILD_BLAS_FROM_SOURCE=ON
    ```

4.  **Compilation:**
    ```bash
    cmake --build build -j 16
    ```

### Testing

The following tests were performed to validate the build:

1.  **Tutorial Example:**
    *   The `examples/cpp/tutorial.cpp` file was modified to set the default device to the GPU.
    *   The example was run, and it produced the expected output from the pull request, confirming that the CUDA backend was being used.
    *   The changes to the tutorial file were reverted after the test.



---

### Simple C++ Example for CUDA

A new, simplified C++ example (`simple_example.cpp`) has been added to demonstrate a basic MLX operation running on the CUDA backend.

**How to Run the Simple Example:**

1.  **Build the project:** Ensure you have built the project with the CUDA backend enabled using the instructions in the "MLX-CUDA Backend Testing Notes" section. If you have already built it, you can just build the new example:
    ```bash
    cmake --build build --target simple_example -j 16
    ```

2.  **Run the executable:**
    ```bash
    ./build/examples/cpp/simple_example
    ```

**Expected Output:**
The program will print the two initial arrays and their sum, like this:
```
Array a:
array([[1, 2],
       [3, 4]], dtype=int32)

Array b:
array([[5, 6],
       [7, 8]], dtype=int32)

Result of a + b:
array([[6, 8],
       [10, 12]], dtype=int32)
```
