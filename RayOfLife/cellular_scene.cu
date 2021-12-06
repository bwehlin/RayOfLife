
#include "cellular_scene.cuh"
#include <stdexcept>

rol::CellularScene::CellularScene(size_t i, size_t j, size_t k)
{
  if (i % 8 != 0)
  {
    throw std::runtime_error("'i' must be a multiple of 8");
  }
  
  cudaMallocHost(&m_h_cellBlocks, (i / 8) * j * k * sizeof(char));
}

rol::CellularScene::~CellularScene()
{
  if (m_h_cellBlocks)
  {
    cudaFreeHost(m_h_cellBlocks);
  }
}
