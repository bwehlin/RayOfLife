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

    void initRandomPrimordialSoup(int seed = 2360) override;
    void evolve() override;

    bool isAlive(int x, int y, int z) const override;

  private:
    void initGrid(CellGrid3d& grid);
    
    CellGrid3d m_grid0, m_grid1;

    size_t m_nCellsPerDimension;
    size_t m_blockSz;
    bool m_isEvenFrame = false;
  };
}
