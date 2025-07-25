# Guide: Running MLX with CUDA on a DGX Machine

This guide documents the complete process of building the MLX library with the experimental CUDA backend and running examples on a DGX machine.

## 1. System Specifications

The environment for this build was a DGX machine with the following specifications:

*   **Operating System:** Ubuntu 22.04
*   **CPU:** Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz
*   **GPU:** Tesla V100-DGXS-16GB
*   **CUDA Version:** 12.8

## 2. Building the MLX Library with CUDA

The following steps were taken to compile the MLX C++ library and examples with CUDA support.

### Step 2.1: Install System Dependencies

The build process requires the LAPACK development libraries. You installed these yourself using the following command:

```bash
sudo apt-get install -y liblapacke-dev
```

### Step 2.2: Configure the Build with CMake

We used `cmake` to configure the build. The final, successful command included flags to enable CUDA, build the examples, and specify the CUDA architecture and compiler:

```bash
cmake . -Bbuild -DMLX_BUILD_CUDA=ON -DMLX_BUILD_EXAMPLES=ON -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
```

### Step 2.3: Compile the Project

After a successful configuration, the project was compiled with:

```bash
cmake --build build -j 16
```

## 3. Running a C++ Example

To verify the build, we created and ran a simple C++ example.

### Step 3.1: Create `simple_example.cpp`

We created a new file at `examples/cpp/simple_example.cpp` with code to add two arrays on the GPU and print the result.

### Step 3.2: Update `CMakeLists.txt`

We added the following lines to `examples/cpp/CMakeLists.txt` to include the new example in the build:

```cmake
add_executable(simple_example simple_example.cpp)
target_link_libraries(simple_example PRIVATE mlx)
```

### Step 3.3: Build and Run the Example

The project was rebuilt, and the example was run with:

```bash
./build/examples/cpp/simple_example
```

This produced the correct output, confirming that the MLX-CUDA backend was working at the C++ level.

## 4. Setting up the Python Environment for LLM Inference

To run a Python-based LLM example, we set up a dedicated virtual environment.

### Step 4.1: Create and Activate Virtual Environment

A virtual environment was created to isolate Python dependencies:

```bash
python3 -m venv /home/heythem/mlx-venv
```

### Step 4.2: Install Python Dependencies

The necessary Python packages were installed into the virtual environment using your proxy:

```bash
export https_proxy=http://p130460:RealmiHpSat2131@138.21.88.176:3128
export http_proxy=http://p130460:RealmiHpSat2131@138.21.88.176:3128
export HTTPS_PROXY=http://p130460:RealmiHpSat2131@138.21.88.176:3128
export HTTP_PROXY=http://p130460:RealmiHpSat2131@138.21.88.176:3128

/home/heythem/mlx-venv/bin/python3 -m pip install build nanobind torch sentencepiece numpy huggingface-hub
```

### Step 4.3: Install the MLX Python Package

Finally, the MLX Python package itself was installed from the local source into the virtual environment:

```bash
# (with proxy variables still set)
/home/heythem/mlx-venv/bin/python3 -m pip install .
```

## 5. Running an LLM with MLX-CUDA

With the environment fully configured, you can now run LLM inference on the GPU.

### Step 5.1: Download the Model

We downloaded the TinyLlama model, which is compatible with the version of the example scripts:

```bash
# (with proxy variables still set)
/home/heythem/mlx-venv/bin/huggingface-cli download mlx-community/TinyLlama-1.1B-Chat-v1.0-mlx --local-dir /home/heythem/mlx-examples/llms/llama/tiny_llama
```

### Step 5.2: Run the Inference Script

The Llama example script is executed using the Python interpreter from our virtual environment. The `MLX_DEVICE` environment variable is used to instruct MLX to use the GPU.

```bash
export MLX_DEVICE=gpu
/home/heythem/mlx-venv/bin/python3 /home/heythem/mlx-examples/llms/llama/llama.py --model-path /home/heythem/mlx-examples/llms/llama/tiny_llama --prompt "hello"
```

This command will load the TinyLlama model and run inference on your DGX machine's GPU, demonstrating the successful use of the MLX-CUDA backend.