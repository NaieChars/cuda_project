#include <cstdio>
#include <cuda_runtime.h>

__global__ void writeGlobalIndex(int* out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = idx;
}

int main()
{
    const int N = 1000;

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        printf("No CUDA device found! Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    // 新增：直接把 Compute Capability 打出来，之后能一眼确认 CMake 里的架构设置对不对
    printf("GPU detected: %s (Compute Capability %d.%d)\n", prop.name, prop.major, prop.minor);

    int* d_out = nullptr;
    cudaMalloc(&d_out, N * sizeof(int));
    // 新增：显式把显存清零成 0xFF（而不是让它保持未定义），
    // 这样如果kernel没真正写入，我们会看到全是 -1，而不是随机数，方便判断
    cudaMemset(d_out, 0xFF, N * sizeof(int));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    printf("Launch config: gridSize=%d, blockSize=%d, total threads=%d (need=%d)\n",
           gridSize, blockSize, gridSize * blockSize, N);

    writeGlobalIndex<<<gridSize, blockSize>>>(d_out, N);

    // 新增：launch之后立刻检查——这一步能抓到"这块GPU压根没有匹配的机器码"这类配置错误
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(launchErr));
        return -1;
    }

    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(syncErr));
        return -1;
    }

    // 新增：初始化为一个明显的哨兵值，而不是让它保持未初始化的垃圾值
    int* h_out = new int[N];
    for (int i = 0; i < N; ++i) h_out[i] = -999;

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    bool allCorrect = true;
    int mismatchCount = 0;
    for (int i = 0; i < N; ++i) {
        if (h_out[i] != i) {
            // 新增：不在第一次失败就break，最多打印5条，方便看出错误的"模式"
            if (mismatchCount < 5) {
                printf("Mismatch at position %d: expected %d, actual %d\n", i, i, h_out[i]);
            }
            mismatchCount++;
            allCorrect = false;
        }
    }
    if (allCorrect) {
        printf("PASSED! CUDA pipeline works correctly.\n");
    } else {
        printf("FAILED. Total mismatches: %d / %d\n", mismatchCount, N);
    }

    delete[] h_out;
    cudaFree(d_out);
    return 0;
}