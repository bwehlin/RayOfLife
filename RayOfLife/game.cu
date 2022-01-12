
#include "game.cuh"
#include <stdexcept>
#include <memory>

__host__ __device__
size_t getCellIndex(size_t i, size_t j, size_t k, size_t strideI, size_t strideJ)
{
  return i * strideI + j * strideJ + k / 8;
}

rol::Game::Game(TransitionRule rule, size_t nCellsPerDimension, size_t blockSz)
  : m_cellsPerDim(nCellsPerDimension)
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

