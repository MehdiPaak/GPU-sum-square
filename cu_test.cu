// Testing the performance of several GPU algorithms 
// for calculating the sum of squares against a serial CPU
// algorithm. 
//
// Mehdi Paak

#include<iostream>
#include<stdio.h>
#include <sys/timeb.h>

#include<cuda.h>
#include<cuda_runtime.h>
#include "device_launch_parameters.h"
#include <helper_cuda.h>


//#define PINNED_MEM
#define DATA_SIZE  1024*1024*16 // 16*4MB int
#define NUM_BLOCK  32
#define NUM_THREAD 512


void DeviceQuery();
void GenerateData(int *panNumber, const int i_nSize);
void SumSqaurCPU(const int *id_panNum, int &nSum);

double EvalTimeFrom( double i_dStartTime);  


__global__ void SumSquar1(const int *id_panNum, int *od_panResult)
{

  const int nTid = threadIdx.x;
  const int nSize = DATA_SIZE/NUM_THREAD;

  int sum = 0;
  for(int i = nTid * nSize; i < (nTid + 1) *nSize; i++)
    sum += id_panNum[i] * id_panNum[i];

  od_panResult[nTid] = sum;
}

__global__ void SumSquar2(const int *id_panNum, int *od_panResult)
{

  const int nTid = threadIdx.x;
  int sum = 0;

  for(int i = nTid; i < DATA_SIZE; i += NUM_THREAD)
    sum += id_panNum[i] * id_panNum[i];

  od_panResult[nTid] = sum;
}

__global__ void SumSquar3(const int *id_panNum, int *od_panResult)
{

  const int nTid = threadIdx.x;
  const int nBid = blockIdx.x;

  int sum = 0;

  for(int i = nBid * NUM_THREAD + nTid; i < DATA_SIZE; i += NUM_BLOCK * NUM_THREAD)
    sum += id_panNum[i] * id_panNum[i];

  od_panResult[nBid * NUM_THREAD + nTid] = sum;
}

__global__ void SumSquar4(const int *id_panNum, int *od_panResult)
{
  const int nTid = threadIdx.x;
  const int nBid = blockIdx.x;
  int nSum = 0;

  extern __shared__ int s_panBuff[];

  for(int i = nBid * NUM_THREAD + nTid; i < DATA_SIZE; i += NUM_BLOCK * NUM_THREAD)
    nSum += id_panNum[i] * id_panNum[i];

  s_panBuff[nTid] = nSum;

  __syncthreads();

  int nS;

  
 /*for(nS = 1; nS < blockDim.x; nS *= 2)
  {
    if((nTid % (2*nS)) == 0)
      s_panBuff[nTid] += s_panBuff[nTid + nS];
    
    __syncthreads();
  }*

/*  for(nS = 1; nS < blockDim.x; nS *= 2)
  {
    int indx = 2*nS*nTid;

    if(indx < blockDim.x)
      s_panBuff[indx] += s_panBuff[indx + nS];
    
    __syncthreads();
  } */

 for(nS = blockDim.x/2; nS > 0; nS /= 2)
  {
    if(nTid < nS)
      s_panBuff[nTid] += s_panBuff[nTid + nS];
    
    __syncthreads();
  }

 
/*
if(nTid < 256) { s_panBuff[nTid] += s_panBuff[nTid + 256]; }
__syncthreads();
if(nTid < 128) { s_panBuff[nTid] += s_panBuff[nTid + 128]; }
__syncthreads();
if(nTid < 64) { s_panBuff[nTid] += s_panBuff[nTid + 64]; }
__syncthreads();
if(nTid < 32) { s_panBuff[nTid] += s_panBuff[nTid + 32]; }
__syncthreads();
if(nTid < 16) { s_panBuff[nTid] += s_panBuff[nTid + 16]; }
__syncthreads();
if(nTid < 8) { s_panBuff[nTid] += s_panBuff[nTid + 8]; }
__syncthreads();
if(nTid < 4) { s_panBuff[nTid] += s_panBuff[nTid + 4]; }
__syncthreads();
if(nTid < 2) { s_panBuff[nTid] += s_panBuff[nTid + 2]; }
__syncthreads();
if(nTid < 1) { s_panBuff[nTid] += s_panBuff[nTid + 1]; }
__syncthreads();
*/

  if(nTid == 0)
    od_panResult[nBid] = s_panBuff[0]; 
}


int main(int argc, char* argv[])
{

  // AllocSizes
//#define SumArrSize NUM_BLOCK * NUM_THREAD // for kernel 1,2,3
//#define SHARED_MEM_SIZE 0

#define SumArrSize NUM_BLOCK              // for kernel 4
#define SHARED_MEM_SIZE NUM_THREAD *sizeof(int)

  cudaError_t Status;
  int *panData, *panSum;

  // Creat host data.
#ifdef PINNED_MEM
  Status = cudaMallocHost((void**) &panData,sizeof(*panData) * DATA_SIZE);
  if(Status != cudaSuccess)
  {
    fprintf(stderr,"Cuda Malloc Failed!\n");
    goto ERROR;
  }

  Status = cudaMallocHost((void**) &panSum,sizeof(*panSum)* SumArrSize);
  if(Status != cudaSuccess)
  {
    fprintf(stderr,"Cuda Malloc Failed!\n");
    goto ERROR;
  }
 
#else
  panData = new int [DATA_SIZE];
  panSum = new int[SumArrSize];
#endif //PINNED_MEM

  // Initialize.
  GenerateData(panData, DATA_SIZE);

  // Run on CPU.
  int nCPUSum = 0;
  
  double dT0 = EvalTimeFrom(0.0);
  SumSqaurCPU(panData,nCPUSum);
  double dT1 = EvalTimeFrom(dT0);

  

  // Choose which GPU to run on, change this on a multi-GPU system.
  int * d_panData, *d_panResult;
  int nDevice = 0;
  DeviceQuery();
n1000:
  Status = cudaSetDevice(nDevice);
  if (Status != cudaSuccess) 
  {
    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
     goto ERROR;
  }
  // Allocate data on device.

  Status = cudaMalloc((void**) &d_panData,sizeof(*d_panData) * DATA_SIZE);
  if(Status != cudaSuccess)
  {
    fprintf(stderr,"Cuda Malloc Failed!\n");
    goto ERROR;
  }

  Status = cudaMalloc((void**) &d_panResult,sizeof(*d_panData) * SumArrSize);
  if(Status != cudaSuccess)
  {
    fprintf(stderr,"Cuda Malloc Failed!\n");
    goto ERROR;
  }


  // Transfer data to device.
  Status = cudaMemcpy(d_panData, panData, sizeof(*panData) * DATA_SIZE, cudaMemcpyHostToDevice);
  if(Status != cudaSuccess)
  {
    fprintf(stderr,"cudaMemcpy Failed!\n");
    goto ERROR;
  }

  // Run Kernel.
  
  printf("--- DEVICE %d :\n",nDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  
  // Call the kernel here
  SumSquar4<<<NUM_BLOCK,NUM_THREAD,SHARED_MEM_SIZE>>>(d_panData,d_panResult);
  
  cudaError_t err  = cudaGetLastError();
  if(err != cudaSuccess)
     printf("CUDA error: %s\n", cudaGetErrorString(err));

  cudaEventRecord(stop);

  // Get results from device.
  Status = cudaMemcpy(panSum, d_panResult, sizeof(*panSum) * SumArrSize, cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  
  // Print Results;
  int nSum = 0;
  for(int i = 0; i < SumArrSize; i++)
    nSum += panSum[i];

  int nDataRW = (DATA_SIZE + SumArrSize); 
  int nFOps = DATA_SIZE * 2; // 1 multiply 1 add for each datum.

  printf("CPUSum = %d\n",nCPUSum);
  printf("GPUSum = %d\n\n",nSum);

  printf("CPU time (ms) = %f\n", dT1 * 1000.0);
  printf("CPU GFLOPS    = %f\n\n", nFOps/(dT1 * 1.0e9));

  printf("GPU time (ms) = %f\n", milliseconds);
  printf("GPU GFLOPS    = %f\n", nFOps/(milliseconds*1.0e6));
  printf("GPU Bandwidth (GB/s) = %f\n\n", nDataRW*4.0/(milliseconds * 1.0e6));

  cudaFree(d_panData);
  cudaFree(d_panResult);

  nDevice++;
  if(nDevice < 2)
    goto n1000;
  // Free Mem.

ERROR:
#ifdef PINNED_MEM
  cudaFreeHost(panData);
  cudaFreeHost(panSum);
#else
  delete [] panData;
  delete [] panSum;
#endif //PINNED_MEM

  // Return.
  char ch;
  std::cin>>ch;
  //return EXIT_SUCCESS;
  cudaDeviceReset();
  return Status;
}

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