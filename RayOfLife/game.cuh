﻿#pragma once

#include "scene_object.h"

#include <cstdlib>
#include <memory>

namespace rol
{
  // An Octet is a 2x2x2 grid of cells that are either 'on' or 'off'.
  //
  // We can store an Octet as a single 1-byte value (e.g., uint8_t) and
  // address into it using bit shifts.
  //
  // The memory layout of an Octet is
  //
  // 0: [ z0 y0 x0 ]
  // 1: [ z0 y0 x1 ]
  // 2: [ z0 y1 x0 ]
  // 3: [ z0 y1 x1 ]
  // 4: [ z1 y0 x0 ]
  // 5: [ z1 y0 x1 ]
  // 6: [ z1 y1 x0 ]
  // 7: [ z1 y1 x1 ]
  //
  // To avoid having to convert between booleans and integers, we use
  // uint8_t for x, y, z as well. It is undefined behavior to use anything
  // other than 0 or 1.
  
  // To address [ zk yj xi ], we use a bit mask
  __host__ __device__ inline
  uint8_t getOctetMask(uint8_t x, uint8_t y, uint8_t z)
  {
    return ((1 << (z * 4)) << (y * 2)) << x;
  }
  
  __host__ __device__ inline
  void setOctetValue(uint8_t& octet, uint8_t x, uint8_t y, uint8_t z, bool value)
  {
    auto mask = getOctetMask(x, y, z);
    if (value)
    {
      octet |= mask;
    }
    else
    {
      octet &= ~mask;
    }
  }

  __host__ __device__ inline
  bool getOctetValue(uint8_t octet, uint8_t x, uint8_t y, uint8_t z)
  {
    // Optimization note: explicitly not using a reference here since the reference itself
    // is a 64-bit value that is used to hold the address to an 8-bit value. Better to just
    // copy the 8 bits.

    return octet & getOctetMask(x, y, z);
  }

  struct CellBlock
  {
    uint8_t* octets;
  };

  struct CellGrid3d
  {
    size_t blockDim;   // Side length of each CellBlock
    size_t blockCount; // Number of CellBlocks along each axis

    CellBlock* blocks;
  };

  struct TransitionRule
  {
    // Bayes (1987) defines a transition rule R by a 4-tuple (El, Eu, Fl, Fu)
    // where a cell survives an evolution step if it has El <= E <= Eu living
    // neighbors, and a non-living cell becomes living if it has Fl <= F <= Fu
    // living neighbors.

    uint8_t el, eu;
    uint8_t fl, fu;
  };

  inline TransitionRule makeTransitionRule(uint8_t el, uint8_t eu, uint8_t fl, uint8_t fu)
  {
    TransitionRule rule;
    rule.el = el;
    rule.eu = eu;
    rule.fl = fl;
    rule.fu = fu;
    return rule;
  }

  // A Game consists of an (i,j,k) grid of positions that can be either
  // occupied by a sphere or empty. That is, a 3D bitmap.
  class Game
  {
  public:
    Game(TransitionRule rule, size_t nCellsPerDimension, size_t blockSz = 16u);
    virtual ~Game() = default;

    Game(const Game&) = delete;
    Game& operator=(const Game&) = delete;

  private:
    virtual void initRandomPrimordialSoup(int seed = 2360) = 0;
  };

  

  //__host__ __device__ 
}