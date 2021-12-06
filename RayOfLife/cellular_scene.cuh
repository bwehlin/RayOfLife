#pragma once

#include "scene_object.h"

#include <cstdlib>

namespace rol
{
  // A cellular scene consists of an (i,j,k) grid of positions that
  // can is either occupied by a sphere or is empty. In other words,
  // a 'CellularScene' is a 3D bitmap.
  class CellularScene
  {
  public:
    CellularScene(size_t i, size_t j, size_t k);
    ~CellularScene();

    CellularScene(const CellularScene&) = delete;
    CellularScene& operator=(const CellularScene&) = delete;

    //void transferToDevice();
    //void transferToHost();

    // Evolve the scene according to the Game of Life rules.
    //void evolve();

  private:

    // Each cell block (sorry for the rather ominous terminology!)
    // consists of 8 slots along the x axis. As such, we will
    // allocate (i/8 * j * k) blocks.

    char* m_h_cellBlocks = nullptr;
    char* m_d_cellBlocks = nullptr;
  };

  //__host__ __device__ 
}
