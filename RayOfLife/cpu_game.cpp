#include "cpu_game.h"

#include <random>

rol::CpuGame::CpuGame(TransitionRule rule, size_t nCellsPerDimension, size_t blockSz)
  : Game(rule, nCellsPerDimension, blockSz)
  , m_grid()
  , m_nCellsPerDimension(nCellsPerDimension)
  , m_blockSz(blockSz)
{
  initGrid();
}

rol::CpuGame::~CpuGame()
{
  auto nBlocksPerDim = m_nCellsPerDimension / m_blockSz / 2;
  auto nBlocksTotal = nBlocksPerDim * nBlocksPerDim * nBlocksPerDim;
  for (size_t i = 0ul; i < nBlocksTotal; ++i)
  {
    delete[] m_grid.blocks[i].octets;
  }
  delete[] m_grid.blocks;
}

void
rol::CpuGame::initGrid()
{
  auto nBlocksPerDim = (m_nCellsPerDimension / m_blockSz + 1) / 2;

  m_grid.blockDim = m_blockSz;
  m_grid.blockCount = nBlocksPerDim;

  auto nOctetsPerDim = m_blockSz / 2;

  m_grid.blocks = new CellBlock[nBlocksPerDim * nBlocksPerDim * nBlocksPerDim];
  auto nBlocksTotal = nBlocksPerDim * nBlocksPerDim * nBlocksPerDim;
  for (size_t i = 0ul; i < nBlocksTotal; ++i)
  {
    m_grid.blocks[i].octets = new uint8_t[nOctetsPerDim * nOctetsPerDim * nOctetsPerDim];
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
      m_grid.blocks[i].octets[j] = static_cast<std::uint8_t>(dist(gen));
    }
  }
}

bool rol::CpuGame::isAlive(int x, int y, int z) const
{
  return rol::isAlive(m_grid, x, y, z);
}
