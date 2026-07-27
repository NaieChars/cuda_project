#include <cstdio>
#include <cstdlib>
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

// 粒子数据结构：AoS，每个元素连续存放4个float
struct Particle
{
    float posX, posY;
    float velX, velY;
};

// 每个线程负责一个粒子，做一次新半隐式Euler积分
__global__ void updateParticles(Particle* particles, int n, float dt, float gravity, float boxMinX, float boxMaxX, float boxMinY, float boxMaxY, float restitution)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    Particle p = particles[idx];

    p.velY += gravity * dt;
    p.posX += p.velX * dt;
    p.posY += p.velY * dt;

    if (p.posX < boxMinX) 
    {
        p.posX = boxMinX;
        p.velX = -p.velX * restitution;
    } 
    else if (p.posX > boxMaxX) 
    {
        p.posX = boxMaxX;
        p.velX = -p.velX * restitution;
    }

    // 
    if (p.posY < boxMinY) 
    {
        p.posY = boxMinY;
        p.velY = -p.velY * restitution;
    } 
    else if (p.posY > boxMaxY) 
    {
        p.posY = boxMaxY;
        p.velY = -p.velY * restitution;
    }

    particles[idx] = p;
}


int main()
{
    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    #endif

    const int N = 2000;
    const float dt = 1.0f / 60.0f;
    const float gravity = -9.8f;
    const int FRAME_COUNT = 300;

    const float BOX_MIN_X = 0.0f, BOX_MAX_X = 20.0f;
    const float BOX_MIN_Y = 0.0f, BOX_MAX_Y = 20.0f;
    const float RESTITUTION = 0.8f; 

    // CPU 上初始化粒子
    Particle* h_particles = new Particle[N];
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distX(BOX_MIN_X + 1.0f, BOX_MAX_X - 1.0f);
    std::uniform_real_distribution<float> distY(10.0f, BOX_MAX_Y - 1.0f); 

    for (int i = 0; i < N; ++i) 
    {
        h_particles[i].posX = distX(rng);
        h_particles[i].posY = distY(rng);
        h_particles[i].velX = 0.0f;
        h_particles[i].velY = 0.0f;
    }

    printf("初始化完成：%d 个粒子，模拟 %d 帧\n", N, FRAME_COUNT);

    Particle* d_particles = nullptr;
    CUDA_CHECK(cudaMalloc(&d_particles, N * sizeof(Particle)));
    CUDA_CHECK(cudaMemcpy(d_particles, h_particles, N * sizeof(Particle), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 打开输出文件，写文件头
    std::ofstream outFile("particles_output.bin", std::ios::binary);
    if (!outFile.is_open())
    {
        printf("无法创建输出文件\n");
        return -1;
    }
    outFile.write(reinterpret_cast<const char*>(&N), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(&FRAME_COUNT), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(&BOX_MIN_X), sizeof(float));
    outFile.write(reinterpret_cast<const char*>(&BOX_MAX_X), sizeof(float));
    outFile.write(reinterpret_cast<const char*>(&BOX_MIN_Y), sizeof(float));
    outFile.write(reinterpret_cast<const char*>(&BOX_MAX_Y), sizeof(float));

    // 时间循环
    for (int frame = 0; frame < FRAME_COUNT; frame++)
    {
        updateParticles<<<gridSize, blockSize>>>(d_particles, N, dt, gravity, BOX_MIN_X, BOX_MAX_X, BOX_MIN_Y, BOX_MAX_Y, RESTITUTION);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_particles, d_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost));

        // 只写位置
        for (int i = 0; i < N; i++)
        {
            outFile.write(reinterpret_cast<const char*>(&h_particles[i].posX), sizeof(float));
            outFile.write(reinterpret_cast<const char*>(&h_particles[i].posY), sizeof(float));
        }
        if (frame % 50 == 0)
        {
            printf("已完成第 %d / %d 帧\n", frame, FRAME_COUNT);
        }
    }

    outFile.close();
    printf("模拟完成，结果已写入 particles_output.bin\n");

    delete[] h_particles;
    CUDA_CHECK(cudaFree(d_particles));
    return 0;
}