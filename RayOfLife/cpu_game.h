#pragma once

#include "game.cuh"

namespace rol
{
  class CpuGame : public Game
  {
  public:
    CpuGame(TransitionRule rule, size_t nCellsPerDimension, size_t blockSz = 16u);
    virtual ~CpuGame();

    CpuGame(const CpuGame&) = delete;
    CpuGame& operator=(const CpuGame&) = delete;

  private:
    void initGrid();
    void initRandomPrimordialSoup(int seed = 2360) override;

    CellGrid3d m_grid;

    size_t m_nCellsPerDimension;
    size_t m_blockSz;
  };
}
