#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define N1

const char *kernelSource = 
    "__kernel void arrayAdd(__global int *a,__global int *b,__global int *c)" {\n"
    "    int i = get_global_id(0);\n"
    "      c[i] = a[i] + b [i];\n"

    "}\n";

int main()
{
  c1_platform_id platform;
  c1_device_id device;
  c1_context context;
  c1_command_queue queue;
  c1_program program;
  c1_kernel kernel;
  c1_mem bufferA, bufferB, bufferC;
  c1_event event;
  c1_int err;

  //Intialize inputs
  int *a = (int*)malloc(sizeof(int)); 
  int *b = (int*)malloc(sizeof(int));
  int *c = (int*)malloc(sizeof(int));

  *a = 222;
  *b = 333;

  //Step1 : Create OpenCL context and command queue
  c1GetPlatformIDs(1, &platform, NULL);
  c1GetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,1, &device, NULL);
  context = c1CreateContext(NULL,1, &device, NULL,NULL,NULL);
  queue = c1CreateCommandQueue(context,device,CL_QUEUE_PROFITING_ENABLE,NULL);

  //Step2: Create and build OpenCL program
  program = c1CreateProgramWithSource(context, 1,(const char **)&kernelSource,NULL,NULL);
  c1BuildProgram(program, 1, &device, NUL,NULL,NULL);

  //Step3:Create kernel
  kernel = c1CreateKernel(program, "arrayAdd", NULL);

  //Step4: Create memory buffers
  bufferA = c1CreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), a, NULL);
  bufferB = c1CreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), b, NULL);
  bufferC = c1CreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int), NULL, NULL);

  //Step5: Set kernel arguments
  c1SetKernelArg(kernel, 0, sizeof(c1_mem), &bufferA);
  c1SetKernelArg(kernel, 0, sizeof(c1_mem), &bufferB);
  c1SetKernelArg(kernel, 0, sizeof(c1_mem), &bufferC);
  
  //Step6: Enquue kernel for execution and measure time
  size_t globalSize = N;
  //c1EnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, &event);

  err = c1EnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL,&event);
  if(err !=CL_SUCCESS)
  {
    //Handle the error here, e.g., print an error message and exit
    printf("Erro enqueueing kernel: %d\n",err);
    return1;
  }

  c1WaitForEvents(1,&event);

  //Get Kernel execution time
  c1_ulong startTime, endTime;
  c1GetEventProfilingInfo?(event,CL_PROFILLING_COMMAND_START, sizeof(c1_ulong),&startTime,NULL);
  c1GetEventProfilingInfo?(event,CL_PROFILLING_COMMAND_START, sizeof(c1_ulong),&endTimeTime,NULL);
  double executionTime = (double)(endTime-startTime) / 1000000.0; //Convert nanoseconds to miliseconds

  //Step7: Read the result
  c1EnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, sizeof(int), c, 0,NULL,NULL);

  //Step8: Print the result and execution time
  printf("Execution time: %.6f miliseconds\n",executionTime);
  printf("Result: ");
  printf("%d ", *c);
  printf("\n");

  //Step9: Release resources
  c1ReleaseMemObject(bufferA);
  c1ReleaseMemObject(bufferB);
  c1ReleaseMemObject(bufferC);
  c1ReleaseKernel(kernel);
  c1ReleaseProgram(program);
  c1ReleaseCommandQueue(queue);
  c1ReleaseContext(context);

  //Free allocated memory
  free(a);
  free(b);
  free(c);

  return 0;
  

}