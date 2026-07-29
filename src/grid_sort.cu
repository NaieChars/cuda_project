#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>


#ifdef _WIN32
#include <windows.h>
#endif

#define EMPTY_CELL 0xFFFFFFFFu

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

struct Particle 
{
    float posX, posY;
};

// 每个线程负责一个粒子：算出它属于哪个cell，然后原子递增该cell的计数器
__global__ void assignCellIndex(const Particle* particles, int n, float cellSize, int gridWidth, int gridHeight, int* particleCellIndex)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int cellX = static_cast<int>(floorf(particles[i].posX / cellSize));
    int cellY = static_cast<int>(floorf(particles[i].posY / cellSize));

    // 边界钳制：防止粒子位置正好在边界外(比如posX=8.0时cellX算出来是4，但grid只有0~3)
    cellX = max(0, min(cellX, gridWidth - 1));
    cellY = max(0, min(cellY, gridHeight - 1));

    particleCellIndex[i] = cellY * gridWidth + cellX;
}

// 每个线程负责排序后数组里的一个位置k，检测自己是不是某个cell的起点/终点
__global__ void findCellStartEnd(const int* sortedKeys, int n,
                                  unsigned int* cellStart, unsigned int* cellEnd)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;

    int myKey = sortedKeys[k];

    // 判断是否是起点：自己是数组第0个，或者前一个位置的key和自己不同
    if (k == 0 || sortedKeys[k - 1] != myKey) {
        cellStart[myKey] = k;
    }

    // 判断是否是终点：自己是数组最后一个，或者后一个位置的key和自己不同
    if (k == n - 1 || sortedKeys[k + 1] != myKey) {
        cellEnd[myKey] = k + 1; // 用开区间[start,end)约定
    }
}

int main()
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    // ---------- 网格参数 ----------
    const float cellSize = 2.0f;
    const int gridWidth = 4;   // 覆盖 x: [0, 8)
    const int gridHeight = 4;  // 覆盖 y: [0, 8)
    const int numCells = gridWidth * gridHeight; // 16个cell

    // ---------- 手动摆放20个粒子，方便手算验证 ----------
    const int N = 20;
    Particle h_particles[N] = {
        {0.5f, 0.5f}, {0.5f, 0.5f}, {0.5f, 0.5f}, {0.5f, 0.5f}, {0.5f, 0.5f}, // 5个都在cell(0,0)
        {3.0f, 0.5f}, {3.9f, 0.9f},                                          // 2个在cell(1,0)
        {6.5f, 6.5f}, {6.5f, 6.5f}, {6.5f, 6.5f},                            // 3个在cell(3,3)
        {1.0f, 3.0f},                                                       // 1个在cell(0,1)
        {5.0f, 5.0f}, {5.5f, 5.5f}, {5.9f, 4.1f}, {4.1f, 5.9f},              // 4个在cell(2,2)
        {7.9f, 0.0f},                                                       // 1个在cell(3,0)
        {0.0f, 7.9f},                                                       // 1个在cell(0,3)
        {2.5f, 2.5f}, {2.9f, 2.1f},                                         // 2个在cell(1,1)
        {8.0f, 8.0f}                                                        // 1个，正好在边界外，测试钳制逻辑，应该落在cell(3,3)
    };

    // ---------- 显存分配+计算cellIndex ----------
    Particle* d_particles = nullptr;
    int* d_particleCellIndex = nullptr;
    CUDA_CHECK(cudaMalloc(&d_particles, N * sizeof(Particle)));
    CUDA_CHECK(cudaMalloc(&d_particleCellIndex, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_particles, h_particles, N * sizeof(Particle), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    assignCellIndex<<<gridSize, blockSize>>>(d_particles, N, cellSize, gridWidth, gridHeight, d_particleCellIndex);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------- 用thrust构造 (key=cellIndex, value=原始下标) 并排序 ----------
    // 根据粒子网格编号对粒子排序，同时记录排序后每个位置对应的原始粒子下标
    thrust::device_ptr<int> d_keys_ptr(d_particleCellIndex);
    thrust::device_vector<int> d_keys(d_keys_ptr, d_keys_ptr + N); // 排序会破坏原数据，所以拷贝一份专门用于排序

    thrust::device_vector<int> d_values(N);
    thrust::sequence(d_values.begin(), d_values.end()); // 填入0,1,2,...,N-1，即原始下标

    // sort_by_key: 按d_keys的大小重新排列，d_values跟着同步移动
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());

    // ---------- 构建cellStart/cellEnd索引表 ----------
    unsigned int* d_cellStart = nullptr;
    unsigned int* d_cellEnd = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cellStart, numCells * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_cellEnd, numCells * sizeof(unsigned int)));
    // 关键一步：先把整个数组填成EMPTY_CELL哨兵值，这样没有粒子的cell会保持这个标记
    CUDA_CHECK(cudaMemset(d_cellStart, 0xFF, numCells * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_cellEnd, 0xFF, numCells * sizeof(unsigned int)));

    int* d_sortedKeysRaw = thrust::raw_pointer_cast(d_keys.data());
    findCellStartEnd<<<gridSize, blockSize>>>(d_sortedKeysRaw, N, d_cellStart, d_cellEnd);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------- 拷回结果 ----------
    thrust::host_vector<int> h_sortedKeys = d_keys;
    thrust::host_vector<int> h_sortedValues = d_values;
    unsigned int h_cellStart[numCells];
    unsigned int h_cellEnd[numCells];
    CUDA_CHECK(cudaMemcpy(h_cellStart, d_cellStart, numCells * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cellEnd, d_cellEnd, numCells * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    printf("=== cellStart / cellEnd 索引表 ===\n");
    for (int c = 0; c < numCells; ++c) {
        int cx = c % gridWidth;
        int cy = c / gridWidth;
        if (h_cellStart[c] == EMPTY_CELL) {
            printf("cellIndex=%2d (x=%d,y=%d): 空cell\n", c, cx, cy);
        } else {
            int count = h_cellEnd[c] - h_cellStart[c];
            printf("cellIndex=%2d (x=%d,y=%d): start=%2u end=%2u count=%d\n",
                   c, cx, cy, h_cellStart[c], h_cellEnd[c], count);
        }
    }

    // ---------- 5. 用索引表实际"查询"一个cell，验证能不能正确取出该cell所有粒子 ----------
    printf("\n=== 用索引表查询 cellIndex=10 (预期4个粒子, 都在(4.1~5.9, 4.1~5.9)附近) ===\n");
    int queryCell = 10;
    if (h_cellStart[queryCell] != EMPTY_CELL) {
        for (unsigned int k = h_cellStart[queryCell]; k < h_cellEnd[queryCell]; ++k) {
            int originalIdx = h_sortedValues[k];
            printf("  找到粒子 原始下标=%d pos=(%.1f,%.1f)\n",
                   originalIdx, h_particles[originalIdx].posX, h_particles[originalIdx].posY);
        }
    }

    printf("\n=== 用索引表查询 cellIndex=2 (预期是空cell) ===\n");
    int emptyQueryCell = 2;
    if (h_cellStart[emptyQueryCell] == EMPTY_CELL) {
        printf("  正确识别为空cell，没有遍历任何粒子\n");
    } else {
        printf("  错误！这个cell不应该有粒子\n");
    }

    CUDA_CHECK(cudaFree(d_particles));
    CUDA_CHECK(cudaFree(d_particleCellIndex));
    CUDA_CHECK(cudaFree(d_cellStart));
    CUDA_CHECK(cudaFree(d_cellEnd));

    return 0;
}