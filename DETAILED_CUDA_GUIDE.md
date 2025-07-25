# Detailed Guide: Building and Verifying MLX with CUDA on a DGX Machine

This guide provides a detailed, step-by-step walkthrough of the process used to build the MLX library with its experimental CUDA backend, and to verify the build on a DGX machine.

## 1. Environment and System Specifications

The first step in any build process is to understand the environment. This work was performed on a DGX machine with the following configuration:

*   **Operating System:** Ubuntu 22.04
*   **CPU:** Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz
*   **GPU:** Tesla V100-DGXS-16GB
*   **CUDA Version:** 12.8

## 2. Building the MLX C++ Library with CUDA Support

The core of the task was to compile the MLX library in a way that enabled the CUDA backend. This involved several steps of configuration and troubleshooting.

### Step 2.1: Install System-Level Dependencies

A critical dependency for building MLX is the LAPACK library, which provides routines for linear algebra. The development headers for this library were missing, which caused initial build failures.

This was resolved by installing the `liblapacke-dev` package:

```bash
sudo apt-get install -y liblapacke-dev
```

### Step 2.2: Configuring the Build with CMake

With the system dependencies in place, the next step was to configure the project using `cmake`. This is where we tell the build system what we want to build and how.

The final, successful configuration command was:

```bash
cmake . -Bbuild -DMLX_BUILD_CUDA=ON -DMLX_BUILD_EXAMPLES=ON -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
```

Let's break down what each of these flags does:
*   `cmake . -Bbuild`: This tells CMake to configure the project in the current directory (`.`) and to put the build files in a new directory named `build`.
*   `-DMLX_BUILD_CUDA=ON`: This is the key flag that enables the experimental MLX-CUDA backend.
*   `-DMLX_BUILD_EXAMPLES=ON`: This tells CMake to also build the C++ examples, which we need for verification.
*   `-DCMAKE_CUDA_ARCHITECTURES=native`: This flag tells the CUDA compiler to build for the specific architecture of the GPU in the machine, which is important for performance and compatibility.
*   `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc`: This explicitly tells CMake where to find the NVIDIA CUDA compiler.

### Step 2.3: Compiling the Project

Once the project was configured, the compilation was performed with the following command:

```bash
cmake --build build -j 16
```
*   `cmake --build build`: This command tells CMake to execute the build process using the configuration in the `build` directory.
*   `-j 16`: This is a performance optimization that tells the build system to use 16 parallel jobs, which significantly speeds up the compilation.

## 3. Verifying the CUDA Backend with a Simple C++ Example

To ensure that the CUDA backend was working correctly, we created a simple, self-contained C++ example.

### Step 3.1: Creating the `simple_example.cpp` File

A new file was created at `examples/cpp/simple_example.cpp`. This file contains a minimal MLX program that:
1.  Sets the default device to the GPU.
2.  Creates two 2x2 arrays.
3.  Adds them together.
4.  Prints the inputs and the result to the console.

This provides a clear and unambiguous test of the CUDA backend.

### Step 3.2: Integrating the Example into the Build System

To compile the new example, the `examples/cpp/CMakeLists.txt` file was modified to include it in the build process. The following lines were added:

```cmake
add_executable(simple_example simple_example.cpp)
target_link_libraries(simple_example PRIVATE mlx)
```

### Step 3.3: Building and Running the Verification Test

The project was rebuilt to include the new example. The test was then executed with:

```bash
./build/examples/cpp/simple_example
```

The program produced the correct output, confirming that the MLX library was successfully built with the CUDA backend and that it can be used to perform computations on the DGX machine's GPU.

## 4. Troubleshooting and Problem Solving

During the build process, we encountered several issues that required specific solutions. This section documents those problems and how they were resolved.

### 4.1. Missing CUDA Architecture

*   **Problem:** The initial CMake configuration failed with an error indicating that the CUDA architecture could not be detected.
*   **Solution:** We resolved this by adding the `-DCMAKE_CUDA_ARCHITECTURES=native` flag to the `cmake` command. This tells the CUDA compiler to build for the specific architecture of the GPU in the machine.

### 4.2. Missing CUDA Compiler

*   **Problem:** CMake was unable to find the NVIDIA CUDA compiler (`nvcc`).
*   **Solution:** We located the compiler at `/usr/local/cuda-12.8/bin/nvcc` and provided this path to CMake using the `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc` flag.

### 4.3. Missing BLAS and LAPACK Dependencies

*   **Problem:** The build process failed because it could not find the required BLAS and LAPACK libraries.
*   **Solution:** We first attempted to build these from source by adding the `-DMLX_BUILD_BLAS_FROM_SOURCE=ON` flag. While this worked for the C++ examples, it caused issues with the Python build. The final and most robust solution was to install the system-level development libraries with `sudo apt-get install -y liblapacke-dev`.

### 4.4. Python Build and Dependency Issues

*   **Problem:** When attempting to install the `mlx` Python package, we faced numerous issues related to proxy settings, missing build dependencies like `nanobind`, and the build process not respecting the CMake flags.
*   **Solution:** This was a multi-step process:
    1.  We created a dedicated Python virtual environment to isolate dependencies.
    2.  We installed all required packages (`build`, `nanobind`, `torch`, etc.) into this virtual environment, using the correct proxy settings.
    3.  We ultimately resolved the build failures by installing the `liblapacke-dev` system dependency, which allowed the Python build to succeed.

*   **Solution:** We first attempted to build these from source by adding the `-DMLX_BUILD_BLAS_FROM_SOURCE=ON` flag. While this worked for the C++ examples, it caused issues with the Python build. The final and most robust solution, as you pointed out in [this GitHub issue comment](https://github.com/ml-explore/mlx/issues/2384#issuecomment-3091227272), was to install the system-level development libraries with `sudo apt-get install -y liblapacke-dev`.

### 4.5. Handling Proxy Errors During Python Package Installation

*   **Problem:** When using `pip` to install Python packages, we encountered persistent proxy errors that prevented us from downloading the necessary dependencies.
*   **Solution:** We resolved this by setting the `http_proxy` and `https_proxy` environment variables before running the `pip` commands. The following is an example of the command we used, with placeholders for your credentials:

    ```bash
    export https_proxy=http://ipn:pwd@138.21.88.176:3128
    export http_proxy=http://ipn:pwd@138.21.88.176:3128
    export HTTPS_PROXY=http://ipn:pwd@138.21.88.176:3128
    export HTTP_PROXY=http://ipn:pwd@138.21.88.176:3128

    # Example of using the proxy with pip:
    /home/heythem/mlx-venv/bin/python3 -m pip install build
    ```
    This ensured that `pip` could connect to the internet and download the required packages.