
#include "game.cuh"
#include <stdexcept>
#include <memory>

__host__ __device__
size_t getCellIndex(size_t i, size_t j, size_t k, size_t strideI, size_t strideJ)
{
  return i * strideI + j * strideJ + k / 8;
}

rol::Game::Game(
  uint16_t blockSzX, uint16_t blockSzY, uint16_t blockSzZ,
  size_t i, size_t j, size_t k, 
  rol::TransitionRule rule)
{
  if (i % 2 != 0 || j % 2 != 0 || k % 2 != 0)
  {
    throw std::runtime_error("'i', 'j' and 'k' must be multiples of 2");
  }

  if (i % blockSzX != 0 || j % blockSzY != 0 || k % blockSzZ != 0)
  {
    throw std::runtime_error("'i', 'j' and 'k' must be multiples of their corresponding blockSz");
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
