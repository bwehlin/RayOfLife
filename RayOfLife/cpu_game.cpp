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
    for (auto j = 0ul; j < nOctetsPerDim * nOctetsPerDim * nOctetsPerDim; ++j)
    {
      grid.blocks[i].octets[j] = 0;
    }
  }

  m_isEvenFrame = true;
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

void rol::CpuGame::setAlive(int x, int y, int z, bool alive)
{
  rol::setAlive(m_isEvenFrame ? m_grid0 : m_grid1, x, y, z, alive);
}

void rol::CpuGame::evolve()
{
  auto& oldGrid = m_isEvenFrame ? m_grid0 : m_grid1;
  auto& newGrid = m_isEvenFrame ? m_grid1 : m_grid0;

  auto const& rule = transitionRule();

  for (auto z = 0; z < cellsPerDim(); ++z)
  {
    for (auto y = 0; y < cellsPerDim(); ++y)
    {
      for (auto x = 0; x < cellsPerDim(); ++x)
      {
        if (x == 4 && y == 8 && z == 8)
        {
          int asdf = 0;

        }

        auto neighbors = livingNeighbors(x, y, z);
        bool shouldLive;

        

        if (isAlive(x, y, z))
        {
          shouldLive = neighbors >= rule.el && neighbors <= rule.eu;
        }
        else
        {
          shouldLive = neighbors >= rule.fl && neighbors <= rule.fu;
        }

        rol::setAlive(newGrid, x, y, z, shouldLive);
      }
    }
  }

  m_isEvenFrame = !m_isEvenFrame;
}

int
rol::CpuGame::livingNeighbors(int x, int y, int z) const
{
  auto nCellsPerDim = cellsPerDim();

  auto minX = x == 0 ? 0 : x - 1;
  auto maxX = x == nCellsPerDim - 1 ? nCellsPerDim - 1 : x + 1;
  
  auto minY = y == 0 ? 0 : y - 1;
  auto maxY = y == nCellsPerDim - 1 ? nCellsPerDim - 1 : y + 1;

  auto minZ = z == 0 ? 0 : z - 1;
  auto maxZ = z == nCellsPerDim - 1 ? nCellsPerDim - 1 : z + 1;

  int neighbors = 0;
  for (auto zn = minZ; zn <= maxZ; ++zn)
  {
    for (auto yn = minY; yn <= maxY; ++yn)
    {
      for (auto xn = minX; xn <= maxX; ++xn)
      {
        if (!(xn == x && yn == y && zn == z) && isAlive(xn, yn, zn))
        {
          ++neighbors;
        }
      }
    }
  }

  return neighbors;
}
