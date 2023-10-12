import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.compiler as compiler
import numpy as np
import datetime as dt

# CUDA kernel function
source_code = """
__global__ void my_kernel(float *data_in, float *data_out) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data_out[idx] = 2.0 * data_in[idx]; // Example: double each element
}
"""

# Compile the CUDA module
module = compiler.SourceModule(source_code)
my_kernel = module.get_function("my_kernel")

# Data on the CPU
data_cpu_in  = np.random.rand(100).astype(np.float32)
print("Generated random array... ", data_cpu_in)
data_cpu_out = [] 

# CPU version
print("Beginning the execution of the CPU version...")
time_start   = dt.datetime.now()

for element in data_cpu_in:
    data_cpu_out.append(2 * element)

time_end     = dt.datetime.now()
time_elapsed = time_end - time_start
print("Done...")
print("Output array is: ", data_cpu_out)
print("Execution time: ", time_elapsed.microseconds, " mcs")

print("Beginning the execution of the GPU version...")
time_start   = dt.datetime.now()
# Allocate GPU memory
data_gpu_in  = cuda.mem_alloc(data_cpu_in.nbytes)
data_gpu_out = cuda.mem_alloc(data_cpu_in.nbytes)

# Copy data from CPU to GPU
cuda.memcpy_htod(data_gpu_in, data_cpu_in)

# Launch the CUDA kernel
my_kernel(data_gpu_in, data_gpu_out, block=(100, 1, 1), grid=(1, 1))

# Copy data back from GPU to CPU
cuda.memcpy_dtoh(data_cpu_out, data_gpu_out)

time_end     = dt.datetime.now()
time_elapsed = time_end - time_start
print("Done...")
print("Output array is: ", data_cpu_out)
print("Execution time: ", time_elapsed.microseconds, " mcs")

print("Exiting...")
