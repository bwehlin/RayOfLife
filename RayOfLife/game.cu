
#include "game.cuh"
#include <stdexcept>
#include <memory>

__host__ __device__
size_t getCellIndex(size_t i, size_t j, size_t k, size_t strideI, size_t strideJ)
{
  return i * strideI + j * strideJ + k / 8;
}

rol::Game::Game(TransitionRule rule, size_t nCellsPerDimension, size_t blockSz = 16u)
{
  if (nCellsPerDimension % 2 != 0)
  {
    throw std::runtime_error("'nCellsPerDimension' must be a multiple of 2");
  }

  if (nCellsPerDimension % blockSz != 0)
  {
    throw std::runtime_error("'nCellsPerDimension' must be a multiple of blockSz");
  }
}

rol::Game::~Game()
{
#if 0
  if (m_h_cellBlocks)
  {
    cudaFreeHost(m_h_cellBlocks);
  }
#endif
}
