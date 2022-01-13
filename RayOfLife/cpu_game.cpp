#include "cpu_game.h"

#include <random>

rol::CpuGame::CpuGame(TransitionRule rule, size_t nCellsPerDimension, size_t blockSz)
  : Game(rule, nCellsPerDimension, blockSz)
  , m_grid0()
  , m_grid1()
  , m_nCellsPerDimension(nCellsPerDimension)
  , m_blockSz(blockSz)
{
  initGrid(m_grid0);
  initGrid(m_grid1);
}

rol::CpuGame::~CpuGame()
{
  auto nBlocksPerDim = m_nCellsPerDimension / m_blockSz / 2;
  auto nBlocksTotal = nBlocksPerDim * nBlocksPerDim * nBlocksPerDim;
  for (size_t i = 0ul; i < nBlocksTotal; ++i)
  {
    delete[] m_grid0.blocks[i].octets;
    delete[] m_grid1.blocks[i].octets;
  }
  delete[] m_grid0.blocks;
  delete[] m_grid1.blocks;
}

void
rol::CpuGame::initGrid(CellGrid3d& grid)
{
  auto nBlocksPerDim = (m_nCellsPerDimension / m_blockSz + 1) / 2;

  grid.blockDim = m_blockSz;
  grid.blockCount = nBlocksPerDim;

  auto nOctetsPerDim = m_blockSz / 2;

  grid.blocks = new CellBlock[nBlocksPerDim * nBlocksPerDim * nBlocksPerDim];
  auto nBlocksTotal = nBlocksPerDim * nBlocksPerDim * nBlocksPerDim;
  for (size_t i = 0ul; i < nBlocksTotal; ++i)
  {
    grid.blocks[i].octets = new uint8_t[nOctetsPerDim * nOctetsPerDim * nOctetsPerDim];
  }
}

void
rol::CpuGame::initRandomPrimordialSoup(int seed)
{
  auto nBlocksPerDim = (m_nCellsPerDimension / m_blockSz + 1) / 2;
  auto nBlocksTotal = nBlocksPerDim * nBlocksPerDim * nBlocksPerDim;
  auto nOctetsPerDim = m_blockSz / 2;
  auto nOctetsPerBlock = nOctetsPerDim * nOctetsPerDim * nOctetsPerDim;

  std::random_device rd;
  std::mt19937 gen(seed);
  std::uniform_int_distribution<short> dist(0, 255);
  for (size_t i = 0ul; i < nBlocksTotal; ++i)
  {
    for (size_t j = 0ul; j < nOctetsPerBlock; ++j)
    {
      m_grid0.blocks[i].octets[j] = static_cast<std::uint8_t>(dist(gen));
    }
  }

  m_isEvenFrame = true;
}

bool rol::CpuGame::isAlive(int x, int y, int z) const
{
  return rol::isAlive(m_isEvenFrame ? m_grid0 : m_grid1 , x, y, z);
}

void rol::CpuGame::evolve()
{
  auto& oldGrid = m_isEvenFrame ? m_grid0 : m_grid1;
  auto& newGrid = m_isEvenFrame ? m_grid1 : m_grid0;

  m_isEvenFrame = !m_isEvenFrame;
}
