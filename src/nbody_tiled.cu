#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <fstream>
#include <cuda_runtime.h>

#ifdef _WIN32
#include <windows.h>
#endif

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

struct Particle {
    float posX, posY;
    float velX, velY;
    float mass;
};

// 用于shared memory tile的精简数据：只需要位置+质量，不需要速度
// (速度积分不在这个kernel里做，和naive版本保持一致的两阶段结构)
struct TileData {
    float posX, posY, mass;
};

#define TILE_SIZE 256 // 必须和launch时的blockDim.x一致，下面会检查

__global__ void computeForcesTiled(const Particle* particles, float* accX, float* accY,
                                    int n, float G, float softening)
{
    // 静态声明的shared memory数组，大小=block里的线程数
    // 每个block独占一份，block之间互不可见
    __shared__ TileData tile[TILE_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 注意：这里不能提前return！即使i>=n(线程数不整除n时最后一个block会有多余线程)，
    // 这些"多余线程"仍然需要参与后面协作搬运数据到shared memory的工作，
    // 否则会导致该block内的__syncthreads()死锁(见下面解释)
    float myPosX = (i < n) ? particles[i].posX : 0.0f;
    float myPosY = (i < n) ? particles[i].posY : 0.0f;

    float aX = 0.0f;
    float aY = 0.0f;

    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int loadIdx = t * TILE_SIZE + threadIdx.x; // 这个线程负责搬运的粒子下标

        // 协作加载：每个线程只搬1个粒子进shared memory，分摊带宽
        if (loadIdx < n) {
            tile[threadIdx.x].posX = particles[loadIdx].posX;
            tile[threadIdx.x].posY = particles[loadIdx].posY;
            tile[threadIdx.x].mass = particles[loadIdx].mass;
        } else {
            // 越界的情况填0质量，保证下面计算时这个"虚拟粒子"不产生任何引力贡献
            tile[threadIdx.x].posX = 0.0f;
            tile[threadIdx.x].posY = 0.0f;
            tile[threadIdx.x].mass = 0.0f;
        }

        // 第1个同步点：确保整个block都把这一tile的数据搬完了，才能开始读
        __syncthreads();

        // 用shared memory里的数据计算这个tile对当前线程i的引力贡献
        for (int k = 0; k < TILE_SIZE; ++k) {
            int j = t * TILE_SIZE + k; // 换算回全局下标，用来判断是否是自己
            if (j == i || j >= n) continue;

            float dx = tile[k].posX - myPosX;
            float dy = tile[k].posY - myPosY;
            float distSqr = dx * dx + dy * dy + softening * softening;

            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            aX += G * tile[k].mass * dx * invDist3;
            aY += G * tile[k].mass * dy * invDist3;
        }

        // 第2个同步点：确保整个block都用完了这一tile的数据，才能进入下一轮覆盖写shared memory
        __syncthreads();
    }

    if (i < n) {
        accX[i] = aX;
        accY[i] = aY;
    }
}

__global__ void integrateEuler(Particle* particles, const float* accX, const float* accY,
                                int n, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Particle p = particles[i];
    p.velX += accX[i] * dt;
    p.velY += accY[i] * dt;
    p.posX += p.velX * dt;
    p.posY += p.velY * dt;
    particles[i] = p;
}

int main()
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    const int N = 200000;
    const float G = 1.0f;
    const float softening = 0.2f;
    const float dt = 0.01f;
    const int FRAME_COUNT = 20;
    const float CENTER_MASS = 5000.0f;
    const float ORBIT_MASS = 0.02f; // 和naive版本保持一致的修正值

    Particle* h_particles = new Particle[N];
    h_particles[0] = {0.0f, 0.0f, 0.0f, 0.0f, CENTER_MASS};

    std::mt19937 rng(42); // 和naive版本用相同种子，保证初始条件完全一致，方便后面逐帧对比正确性
    std::uniform_real_distribution<float> distR(3.0f, 10.0f);
    std::uniform_real_distribution<float> distTheta(0.0f, 6.2831853f);

    for (int i = 1; i < N; ++i) {
        float r = distR(rng);
        float theta = distTheta(rng);
        float px = r * cosf(theta);
        float py = r * sinf(theta);
        float v = sqrtf(G * CENTER_MASS / r);
        float vx = -v * sinf(theta);
        float vy = v * cosf(theta);
        h_particles[i] = {px, py, vx, vy, ORBIT_MASS};
    }

    printf("初始化完成：%d 个粒子，模拟 %d 帧（tiled版本，TILE_SIZE=%d）\n", N, FRAME_COUNT, TILE_SIZE);

    Particle* d_particles = nullptr;
    float* d_accX = nullptr;
    float* d_accY = nullptr;
    CUDA_CHECK(cudaMalloc(&d_particles, N * sizeof(Particle)));
    CUDA_CHECK(cudaMalloc(&d_accX, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_accY, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_particles, h_particles, N * sizeof(Particle), cudaMemcpyHostToDevice));

    int blockSize = TILE_SIZE; // 必须和TILE_SIZE一致，这是tiled kernel的硬性要求
    int gridSize = (N + blockSize - 1) / blockSize;

    std::ofstream outFile("day30_tiled_output.bin", std::ios::binary);
    if (!outFile.is_open()) {
        printf("无法创建输出文件\n");
        return -1;
    }
    outFile.write(reinterpret_cast<const char*>(&N), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(&FRAME_COUNT), sizeof(int));
    for (int i = 0; i < N; ++i) {
        outFile.write(reinterpret_cast<const char*>(&h_particles[i].mass), sizeof(float));
    }

    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    CUDA_CHECK(cudaEventRecord(startEvent));

    for (int frame = 0; frame < FRAME_COUNT; ++frame) {
        computeForcesTiled<<<gridSize, blockSize>>>(d_particles, d_accX, d_accY, N, G, softening);
        CUDA_CHECK(cudaGetLastError());

        integrateEuler<<<gridSize, blockSize>>>(d_particles, d_accX, d_accY, N, dt);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(h_particles, d_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost));

        for (int i = 0; i < N; ++i) {
            outFile.write(reinterpret_cast<const char*>(&h_particles[i].posX), sizeof(float));
            outFile.write(reinterpret_cast<const char*>(&h_particles[i].posY), sizeof(float));
        }

        if (frame % 100 == 0) {
            printf("已完成第 %d / %d 帧\n", frame, FRAME_COUNT);
        }
    }

    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    float totalMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalMs, startEvent, stopEvent));

    outFile.close();
    printf("模拟完成，结果已写入 day30_tiled_output.bin\n");
    printf("=== 性能数据(tiled版本) ===\n");
    printf("总耗时: %.3f ms, 平均每帧: %.4f ms\n", totalMs, totalMs / FRAME_COUNT);

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    delete[] h_particles;
    CUDA_CHECK(cudaFree(d_particles));
    CUDA_CHECK(cudaFree(d_accX));
    CUDA_CHECK(cudaFree(d_accY));

    return 0;
}