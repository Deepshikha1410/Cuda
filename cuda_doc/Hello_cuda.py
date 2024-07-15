import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import Sourcemodule

#CUDA KERNEL CODE FOR PRINTING "Hello,World!"
cuda_code = """
#include<stdio.h>

__global__ void hello_world(){
    printf("Hello,World!\\n");
}
"""

#Load the CUDA module
cuda_module = Sourcemodule(cuda_cuda)
hello_world_kernel = cuda_module.get_function("hello world")

#Set upm block and grid dimensions
block_dim = (1,1,1)
grid_dim = (1,1)

#Launch the CUDA kernel
hello_world_kernel(block=block_dim, grid=grid_dim)