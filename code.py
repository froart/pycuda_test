import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.compiler as compiler
import numpy as np

# CUDA kernel function
source_code = """
__global__ void my_kernel(float *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] *= 2.0; // Example: double each element
}
"""

# Compile the CUDA module
module = compiler.SourceModule(source_code)
my_kernel = module.get_function("my_kernel")

# Data on the CPU
data_cpu = np.random.rand(100).astype(np.float32)

# print initial data
print(data_cpu)

# Allocate GPU memory
data_gpu = cuda.mem_alloc(data_cpu.nbytes)

# Copy data from CPU to GPU
cuda.memcpy_htod(data_gpu, data_cpu)

# Launch the CUDA kernel
my_kernel(data_gpu, block=(100, 1, 1), grid=(1, 1))

# Copy data back from GPU to CPU
cuda.memcpy_dtoh(data_cpu, data_gpu)

# Data is now modified on the CPU
print(data_cpu)
