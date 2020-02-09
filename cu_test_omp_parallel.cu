// Using OpenMP run the kernel code on two GPUs in parallel.
// Data is divided between the two GPUs and aggregated in the end
// by the master thread.
//
// Mehdi Paak

#include<iostream>
#include<stdio.h>
#include <sys/timeb.h>

#include<cuda.h>
#include<cuda_runtime.h>
#include "device_launch_parameters.h"
#include <helper_cuda.h>

// Multiple GPU using omp
#include<omp.h>


//#define PINNED_MEM
#define DATA_SIZE  1024*1024*1024 // 16*4MB int
#define NUM_BLOCK  32
#define NUM_THREAD 512
#define NUM_OMP_THREAD 2
#define nPerThrdDataSize  DATA_SIZE /NUM_OMP_THREAD


void DeviceQuery();
void GenerateData(int *panNumber, const int i_nSize);
void SumSqaurCPU(const int *id_panNum, int &nSum);

double EvalTimeFrom( double i_dStartTime);  

__global__ void SumSquar5(const int *id_panNum,int *od_panResult)
{
  const int nTid = threadIdx.x;
  const int nBid = blockIdx.x;
  int nSum = 0;

  extern __shared__ int s_panBuff[];

  for(int i = nBid * NUM_THREAD + nTid; i < nPerThrdDataSize; i += NUM_BLOCK * NUM_THREAD)
    nSum += id_panNum[i] * id_panNum[i];

  s_panBuff[nTid] = nSum;

  __syncthreads();

  int nS;


 for(nS = blockDim.x/2; nS > 0; nS /= 2)
  {
    if(nTid < nS)
      s_panBuff[nTid] += s_panBuff[nTid + nS];
    
    __syncthreads();
  }

 
  if(nTid == 0)
    od_panResult[nBid] = s_panBuff[0]; 
}


int main(int argc, char* argv[])
{

  #define SumArrSize NUM_BLOCK              // for kernel 5
  #define SHARED_MEM_SIZE NUM_THREAD *sizeof(int)

  //Get number of CPU and GPU
  int nNumGPU = 0;
  int nNumCPU = 0;

  checkCudaErrors(cudaGetDeviceCount(&nNumGPU));
  nNumCPU = omp_get_num_procs();
  
  if (nNumGPU < 1)
  {
    printf("no CUDA capable devices were detected\n");
    return 1;
  }

  // display CPU and GPU configuration
  printf("number of host CPUs:\t%d\n", nNumCPU);
  printf("number of CUDA devices:\t%d\n", nNumGPU);

  for (int i = 0; i < nNumGPU; i++)
  {
    cudaDeviceProp dprop;
    cudaGetDeviceProperties(&dprop, i);
    printf("   %d: %s\n", i, dprop.name);
  }
 // nNumGPU = 1;
  printf("**---------------------------**\n");


  // Host data.
  int *panData, *panSumCPU;
  //const int nPerThrdDataSize = DATA_SIZE /NUM_OMP_THREAD;
  panData = new int [DATA_SIZE];
  panSumCPU = new int [nNumGPU];

  // Initialize.
  GenerateData(panData, DATA_SIZE);

  // Run on CPU.
  int nCPUSum = 0;
  
  double dT0 = EvalTimeFrom(0.0);
  SumSqaurCPU(panData,nCPUSum);
  double dT1 = EvalTimeFrom(dT0);

  omp_set_dynamic(0);
  omp_set_num_threads(NUM_OMP_THREAD);

#pragma omp parallel 
  {
    //printf("Hello\n");
    int *panSum = new int[SumArrSize]();
    const int nCPU_Thrd_ID = omp_get_thread_num();
    const int nNum_CPU_Thrds = omp_get_num_threads();

    // Choose which GPU to run on, change this on a multi-GPU system.
    int nGPU_ID = -1;
    checkCudaErrors(cudaSetDevice(nCPU_Thrd_ID % NUM_OMP_THREAD));
    checkCudaErrors(cudaGetDevice(&nGPU_ID));
    printf("CPU thread %d (of %d) uses CUDA device %d\n", nCPU_Thrd_ID, nNum_CPU_Thrds, nGPU_ID);
    
    
    // Allocate data on device.
    int * d_panData, *d_panResult;
    checkCudaErrors(cudaMalloc((void**) &d_panData,sizeof(*d_panData) * nPerThrdDataSize));
    checkCudaErrors(cudaMalloc((void**) &d_panResult,sizeof(*d_panData) * SumArrSize));
    
    // Transfer data to device.
    checkCudaErrors(cudaMemcpy(d_panData, panData + nGPU_ID * nPerThrdDataSize, sizeof(*panData) * nPerThrdDataSize, cudaMemcpyHostToDevice));
    
    
    // Run Kernel.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    SumSquar5<<<NUM_BLOCK,NUM_THREAD,SHARED_MEM_SIZE>>>(d_panData,d_panResult);
  
    cudaError_t err  = cudaGetLastError();
    if(err != cudaSuccess)
       printf("CUDA error --> %s\n", cudaGetErrorString(err));
    
    cudaEventRecord(stop);
    
    // Get results from device.
    checkCudaErrors(cudaMemcpy(panSum, d_panResult, sizeof(*panSum) * SumArrSize, cudaMemcpyDeviceToHost));
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
  
    // Print Results;
    int nSum = 0;
    for(int i = 0; i < SumArrSize; i++)
      nSum += panSum[i];
  
    panSumCPU[nCPU_Thrd_ID] = nSum;
  
    printf("GPU %d time (ms) = %f\n", nGPU_ID, milliseconds);

    cudaFree(d_panData);
    cudaFree(d_panResult);
    delete [] panSum;
  }// End omp parallel.

  // aggregate the result from GPUs
  int nSumTotGPU = 0;
  for(int i = 0; i < NUM_OMP_THREAD; i++)
    nSumTotGPU += panSumCPU[i];

  printf("CPUSum = %d\n",nCPUSum);
  printf("GPUSum = %d\n\n",nSumTotGPU);
  printf("CPU time (ms) = %f\n", dT1 * 1000.0);




  // Free Mem.
  delete [] panData;
  delete [] panSumCPU;


  // Return.
  char ch;
  std::cin>>ch;
  //return EXIT_SUCCESS;
  return 1;
}

//====================================================
void GenerateData(
  int *panNumber, 
  const int i_nSize)
{

  for(int i = 0; i < i_nSize; i++)
    panNumber[i] = rand() % 10;
}

void DeviceQuery()
{
  int nDevices;
  int driverVersion = 0;
  int runtimeVersion = 0;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, i);
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  Device Number: %d\n", i);
    printf("  Device name: %s\n", deviceProp.name);

    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

    printf("  Memory Clock Rate (KHz): %d\n", deviceProp.memoryClockRate/1000);
    printf("  Memory Bus Width (bits): %d\n", deviceProp.memoryBusWidth);

    char msg[256];
    sprintf(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
                (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
    printf("%s", msg);

    printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
               deviceProp.multiProcessorCount,
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
    printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

    if (deviceProp.l2CacheSize)
    {
       printf("  L2 Cache Size:                                 %d KB\n", deviceProp.l2CacheSize/1024);
    }

    printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
    printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
    printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
    printf("  Warp size:                                     %d\n", deviceProp.warpSize);
    printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
    printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
    printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*deviceProp.memoryClockRate*(deviceProp.memoryBusWidth/8)/1.0e6);
  }
}

void SumSqaurCPU(const int *id_panNum, int &nSum)
{
  nSum = 0;
  for(int i = 0; i < DATA_SIZE; i++)
    nSum += id_panNum[i]*id_panNum[i];
}

// Return time in seconds.
double EvalTimeFrom(double i_dStartTime)  
{
  // Get current time.
  timeb sTimeBuffer;
  ftime(&sTimeBuffer);

  // Calculate time from start.
  double dTimeFromStart = sTimeBuffer.time + sTimeBuffer.millitm / 1000.0 - i_dStartTime;

  // Check if day has been changed.
  if(dTimeFromStart < 0.0) 
    dTimeFromStart += 86400.0; // 1 day = 86400 sec

  // Resulting time from start.
  return dTimeFromStart;
}