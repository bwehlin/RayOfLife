
#include "cellular_scene.cuh"
#include <stdexcept>
#include <memory>

__host__ __device__
size_t getCellIndex(size_t i, size_t j, size_t k, size_t strideI, size_t strideJ)
{
  return i * strideI + j * strideJ + k / 8;
}

rol::CellularScene::CellularScene(size_t i, size_t j, size_t k)
  : m_strideI(i), m_strideJ(j)
{
  if (k % 8 != 0)
  {
    throw std::runtime_error("'k' must be a multiple of 8");
  }
  
  m_h_cellBlocks.resize(i * j * (k / 8));

  //cudaMallocHost(&m_h_cellBlocks, (i / 8) * j * k * sizeof(char));
}

rol::CellularScene::~CellularScene()
{
#if 0
  if (m_h_cellBlocks)
  {
    cudaFreeHost(m_h_cellBlocks);
  }
#endif
}

void
rol::CellularScene::evolveCpu()
{
  auto isAlive = [](size_t i, size_t j, size_t k)
  {

  };

  auto newCells = m_h_cellBlocks;

}
