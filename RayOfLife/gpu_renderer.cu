#include "gpu_renderer.cuh"
#include "support.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>

#include "vector_math.cuh"
#include "amantides_woo.cuh"
#include "intersect.cuh"
#include "game.cuh"

namespace
{
  __device__ bool isAlive(bool* game, int x, int y, int z, int cellsPerDim)
  {
    return game[z * cellsPerDim * cellsPerDim + y * cellsPerDim + x];
  }

  __device__ void castRay(rol::RayIntersection& intersection, 
    rol::AmantidesWooState& awstate, 
    fptype3 rayOrigin, fptype3 rayDirection, 
    bool* game, int cellsPerDim, rol::SceneData* scene, fptype3 cameraOrigin)
  {
    while (true)
    {
      if (awstate.pos.x < 0 || awstate.pos.x >= cellsPerDim
        || awstate.pos.y < 0 || awstate.pos.y >= cellsPerDim
        || awstate.pos.z < 0 || awstate.pos.z >= cellsPerDim)
      {
        // We have fallen out of the cell grid
        intersection.hit = false;
        return;
      }

      if (isAlive(game, awstate.pos.x, awstate.pos.y, awstate.pos.z, cellsPerDim))
      {
        intersection = rol::traceRay(rayOrigin, rayDirection, awstate.pos, *scene, cameraOrigin);
        if (intersection.hit)
        {
          return;
        }
      }

      nextAwStep(awstate);
    }

    intersection.hit = false;
  }
  
  __device__ fptype3 subpixelColor(fptype x, fptype y, fptype3 cameraOrigin, 
    int cellsPerDim, int depth, bool* game, rol::SceneData* scene)
  {
    auto rayOrigin = cameraOrigin;
    auto cameraTarget = makeFp3(cameraOrigin.x + 1.f, x, y);

    auto rayDirection = normalize(cameraTarget - rayOrigin);

    auto color = makeFp3(0.f, 0.f, 0.f);
    fptype reflection = 1.f;

    rol::AmantidesWooState awstate;
    rol::initAmantidesWoo(awstate, rayOrigin, rayDirection, cellsPerDim);
    if (awstate.pos.x != 0)
    {
      // Ray from origin does not hit cell grid
      color = makeFp3(1.f, 1.f, 0.f);
      return color;
    }

    rol::RayIntersection intersection;
    while (depth--)
    {
      castRay(intersection, awstate, rayOrigin, rayDirection, game, cellsPerDim, scene, cameraOrigin);
      if (!intersection.hit)
      {
        break;
      }

      rayOrigin = intersection.point + intersection.normal * static_cast<fptype>(0.0001f);
      rayDirection = normalize(rayDirection - 2 * dot(rayDirection, intersection.normal) * intersection.normal);

      rol::initAmantidesWooInside(awstate, rayOrigin, rayDirection, cellsPerDim);
      rol::nextAwStep(awstate);

      color += reflection * intersection.color;
      reflection *= scene->sphereReflection;
    }

    return color;
  }

  __global__ void renderSubpixels(fptype3* image, int w, int h,
    fptype2 screenMin, fptype2 screenMax, int subpixels, int maxDepth,
    bool* game, int cellsPerDim,
    fptype3 cameraOrigin, rol::SceneData* scene)
  {
    auto ix = blockIdx.x * blockDim.x + threadIdx.x;
    auto iy = blockIdx.y * blockDim.y + threadIdx.y;
    
    auto x = screenMin.x + (screenMax.x - screenMin.x) * ix / fptype(w * subpixels);
    auto y = screenMin.y + (screenMax.y - screenMin.y) * iy / fptype(h * subpixels);
    
    int imx = ix / subpixels;
    int imy = iy / subpixels;

    auto imoffset = imy * w + imx;

    auto color = subpixelColor(x, y, cameraOrigin, cellsPerDim, maxDepth, game, scene);
    atomicAdd(&image[imoffset].x, color.x);
    atomicAdd(&image[imoffset].y, color.y);
    atomicAdd(&image[imoffset].z, color.z);
  }

  __global__ void normalizePixels(fptype3* image, int w, int subpixels)
  {
    auto ix = blockIdx.x * blockDim.x + threadIdx.x;
    auto iy = blockIdx.y * blockDim.y + threadIdx.y;

    auto offset = iy * w + ix;
    auto factor = fptype{ 1.f } / fptype{ subpixels };
    factor *= factor;

    image[offset].x *= factor;
    image[offset].y *= factor;
    image[offset].z *= factor;
  }

}

rol::GpuRenderer::GpuRenderer(size_t w, size_t h)
  : Renderer(w, h)
  , m_d_game(nullptr)
  , m_imageData(nullptr)
  , m_d_subpixelBuffer(nullptr)
{
  CHK_ERR(cudaMallocManaged(&m_imageData, sizeof(fptype3) * w * h))
  CHK_ERR(cudaMallocManaged(&m_scene, sizeof(rol::SceneData)))

  *m_scene = SceneData();
}

rol::GpuRenderer::~GpuRenderer()
{
  auto freeMem = [](void* ptr)
  {
    if (!ptr)
    {
      return;
    }
    auto err = cudaFree(ptr);
    if (err != cudaSuccess)
    {
      std::cerr << "Warning! Could not free memory while destroying GPU renderer: " << cudaGetErrorString(err) << '\n';
    }
  };
  
  freeMem(m_imageData);
  freeMem(m_d_game);
  freeMem(m_scene);
  freeMem(m_d_subpixelBuffer);
}

void rol::GpuRenderer::produceFrame(const Game& game, const Camera& camera,
  const fptype2& screenMin, const fptype2& screenMax)
{
  transferGameToGpu(game);

  for (int i = 0; i < width() * height(); ++i)
  {
    m_imageData[i] = makeFp3(0, 0, 0);
  }

  int blockDim = 16;
  if (width() % blockDim != 0 || height() % blockDim != 0
    || width() * subpixelCount() % blockDim != 0 || height() * subpixelCount() % blockDim != 0)
  {
    // Don't want to deal with block misalignment in the kernel if we don't absolutely have to.
    throw std::runtime_error("Screen dimensions and subpixel expansions must be multiples of " + std::to_string(blockDim));
  }

  auto subpixelBlocks = dim3(width() * subpixelCount() / blockDim, height() * subpixelCount() / blockDim);
  auto subpixelThreadsPerBlock = dim3(blockDim, blockDim);

  std::cout << "spb " << subpixelBlocks.x << " " << subpixelBlocks.y << std::endl;

  renderSubpixels<<<subpixelBlocks, subpixelThreadsPerBlock >>>(
    m_imageData, width(), height(),
    screenMin, screenMax, subpixelCount(), maxDepth(),
    m_d_game, game.cellsPerDim(),
    camera.origin, m_scene);

  auto err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    throw CudaError(err, __FILE__, __LINE__);
  }

  auto pixelBlocks = dim3(width() / blockDim, height() / blockDim);
  auto pixelThreadsPerBlock = dim3(blockDim, blockDim);

  normalizePixels<<<pixelBlocks, pixelThreadsPerBlock>>>(m_imageData, width(), subpixelCount());
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    throw CudaError(err, __FILE__, __LINE__);
  }

  CHK_ERR(cudaDeviceSynchronize())

}

void
rol::GpuRenderer::transferGameToGpu(const Game& game)
{
  if (!m_d_game)
  {
    CHK_ERR(cudaMalloc(&m_d_game, sizeof(bool) * game.cellsPerDim() * game.cellsPerDim() * game.cellsPerDim()));
  }

  auto nCellsPerDim = game.cellsPerDim();

  if (!m_h_game)
  {
    m_h_game = std::unique_ptr<bool[]>(new bool[nCellsPerDim * nCellsPerDim * nCellsPerDim]);
  }
  for (auto z = 0; z < nCellsPerDim; ++z)
  {
    for (auto y = 0; y < nCellsPerDim; ++y)
    {
      for (auto x = 0; x < nCellsPerDim; ++x)
      {
        m_h_game[z * nCellsPerDim * nCellsPerDim + y * nCellsPerDim + x] = game.isAlive(x, y, z);
      }
    }
  }

  CHK_ERR(cudaMemcpy(m_d_game, m_h_game.get(), sizeof(bool) * nCellsPerDim * nCellsPerDim * nCellsPerDim, cudaMemcpyHostToDevice))
}

const fptype3* rol::GpuRenderer::imageData() const
{
  return m_imageData;
}
