import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

#CUDA kernel code to return data
cuda_code = """
__global__ void return_data(char *output) {
    const char msg[] = "Hello,World!";
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < sizeof(msg)) {
        output[idx] = msg[idx];
    }
}
"""
#Compile the CUDA module
cuda_module = SourceModule(cuda_code)
return_data_kernel = cuda_module.get_function("return_data")

#Allocate memory on the GPU for the output
output_size = 15 #Length of the "Hello,world!" string + null terminator
output_gpu = cuda.mem_alloc(output_size)

#Set up block and grid dimensions
block_dim = (output_size,1,1)
grid_dim = (1,1)

#Launch the kernel
return_data_kernel(output_gpu, block=block_dim, grid=grid_dim)

#Copy the result back to the host
output_host = np.empty(output_size, dtype=np.uint8)
cuda.memcpy_dtoh(output_host, output_gpu)

#Convert the result to a string and print it
output_str = ''.join(chr(c) for c in output_host)
print(output_str)