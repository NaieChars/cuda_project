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

// 新增mass字段
struct Particle {
    float posX, posY;
    float velX, velY;
    float mass;
};

__global__ void  computeForcesNaive(const Particle* particles, float* accX, float* accY, int n, float G, float softening)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float myPosX = particles[i].posX;
    float myPosY = particles[i].posY;

    float aX = 0.0f;
    float aY = 0.0f;

    for (int j = 0; j < n; ++j) {
        if (j == i) continue;

        float dx = particles[j].posX - myPosX;
        float dy = particles[j].posY - myPosY;
        float distSqr = dx * dx + dy * dy + softening * softening;

        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        aX += G * particles[j].mass * dx * invDist3;
        aY += G * particles[j].mass * dy * invDist3;
    }

    accX[i] = aX;
    accY[i] = aY;
}

// 积分kernel：用上一步算好的加速度，做半隐式Euler更新
__global__ void integrateEuler(Particle* particles, const float* accX, const float* accY, int n, float dt)
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
    const int FRAME_COUNT = 3;
    const float CENTER_MASS = 5000.0f;
    const float ORBIT_MASS = 0.02f;

    Particle* h_particles = new Particle[N];
    h_particles[0] = {0.0f, 0.0f, 0.0f, 0.0f, CENTER_MASS}; // 中心天体静止在原点

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distR(3.0f, 10.0f);      // 轨道半径范围
    std::uniform_real_distribution<float> distTheta(0.0f, 6.2831853f); // 0~2pi

    for (int i = 1; i < N; i++)
    {
        float r = distR(rng);
        float theta = distTheta(rng);

        float px = r * cosf(theta);
        float py = r * sinf(theta);

        // 圆周轨道速度大小: v = sqrt(G * M_center / r)
        float v = sqrtf(G * CENTER_MASS / r);
        // 切向方向：把位置向量(cos,sin)逆时针旋转90度，得到(-sin,cos)
        float vx = -v * sinf(theta);
        float vy = v * cosf(theta);

        h_particles[i] = {px, py, vx, vy, ORBIT_MASS};
    }

    // 显存分配
    Particle* d_particles = nullptr;
    float* d_accX = nullptr;
    float* d_accY = nullptr;
    CUDA_CHECK(cudaMalloc(&d_particles, N * sizeof(Particle)));
    CUDA_CHECK(cudaMalloc(&d_accX, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_accY, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_particles, h_particles, N * sizeof(Particle), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 输出文件
    std::ofstream outFile("day30_output.bin", std::ios::binary);
    if (!outFile.is_open()) {
        printf("无法创建输出文件\n");
        return -1;
    }
    outFile.write(reinterpret_cast<const char*>(&N), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(&FRAME_COUNT), sizeof(int));
    for (int i = 0; i < N; ++i) {
        outFile.write(reinterpret_cast<const char*>(&h_particles[i].mass), sizeof(float));
    }

    // ---------- 计时用的CUDA event ----------
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    CUDA_CHECK(cudaEventRecord(startEvent));

    for (int frame = 0; frame < FRAME_COUNT; frame++)
    {
        computeForcesNaive<<<gridSize, blockSize>>>(d_particles, d_accX, d_accY, N, G, softening);
        CUDA_CHECK(cudaGetLastError());

        integrateEuler<<<gridSize, blockSize>>>(d_particles, d_accX, d_accY, N, dt);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(h_particles, d_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost));

        for (int i = 0; i < N; i++)
        {
            outFile.write(reinterpret_cast<const char*>(&h_particles[i].posX), sizeof(float));
            outFile.write(reinterpret_cast<const char*>(&h_particles[i].posY), sizeof(float));
        }

        if (frame % 100 == 0) 
        {
            printf("已完成第 %d / %d 帧\n", frame, FRAME_COUNT);
        }
    }

    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    float totalMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalMs, startEvent, stopEvent));

    outFile.close();
    printf("模拟完成，结果已写入 day30_output.bin\n");
    printf("=== 性能基线(naive版本) ===\n");
    printf("总耗时: %.3f ms, 平均每帧: %.4f ms\n", totalMs, totalMs / FRAME_COUNT);

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    delete[] h_particles;
    CUDA_CHECK(cudaFree(d_particles));
    CUDA_CHECK(cudaFree(d_accX));
    CUDA_CHECK(cudaFree(d_accY));

    return 0;
}