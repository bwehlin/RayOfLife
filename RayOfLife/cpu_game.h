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

    bool isAlive(itype x, itype y, itype z) const override;
    void setAlive(itype x, itype y, itype z, bool alive = true) override;

  private:
    void initGrid(CellGrid3d& grid);
    itype livingNeighbors(itype x, itype y, itype z) const;
    
    CellGrid3d m_grid0, m_grid1;

    size_t m_nCellsPerDimension;
    size_t m_blockSz;
    bool m_isEvenFrame = false;
  };
}
