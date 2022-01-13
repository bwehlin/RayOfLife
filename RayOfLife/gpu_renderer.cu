#include "gpu_renderer.cuh"
#include "support.cuh"
#include <iostream>

#include "vector_math.cuh"
#include "amantides_woo.cuh"
#include "intersect.cuh"
#include "game.cuh"

rol::GpuRenderer::GpuRenderer(size_t w, size_t h)
  : Renderer(w, h)
  , m_d_game(nullptr)
  , m_imageData(nullptr)
{
  CHK_ERR(cudaMallocManaged(&m_imageData, sizeof(fptype3) * w * h))
}

rol::GpuRenderer::~GpuRenderer()
{
  auto err = cudaFree(m_imageData);
  if (err != cudaSuccess)
  {
    std::cerr << "Warning! Could not free memory while destroying GPU renderer: " << cudaGetErrorString(err) << '\n';
  }
}

void rol::GpuRenderer::produceFrame(const Game& game, const Camera& camera,
  const std::vector<fptype>& xspace, const std::vector<fptype>& yspace)
{

}

const fptype3* rol::GpuRenderer::imageData() const
{
  return m_imageData;
}
