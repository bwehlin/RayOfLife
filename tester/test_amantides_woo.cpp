#include "pch.h"

#include "../RayOfLife/amantides_woo.cuh"

// Here we are testing that our implementation gives the same results as an existing implementation.
// 
// As a reference, we used Jesus P. Mena-Chalco's MATLAB implementation that is available
// at https://mathworks.com/matlabcentral/fileexchange/26852-a-fast-voxel-traversal-algorithm-for-ray-tracing

TEST(AmantidesWoo, StartAtOrigin) 
{
  auto state = rol::initAmantidesWoo(makeFp3(0.f, 0.f, 0.f), makeFp3(0.3f, 0.5f, 0.7f), 16);
  auto lastState = state;
  while (state.pos.x < 16 && state.pos.y < 16 && state.pos.z < 16)
  {
    lastState = state;
    rol::nextAwStep(state);
  }

  EXPECT_EQ(lastState.pos.x, 6);
  EXPECT_EQ(lastState.pos.y, 11);
  EXPECT_EQ(lastState.pos.z, 15);
}

TEST(AmantidesWoo, StartAtOriginWithOffset)
{
  auto state = rol::initAmantidesWoo(makeFp3(0.5f, 0.3f, 0.2f), makeFp3(0.3f, 0.5f, 0.7f), 16);
  auto lastState = state;
  while (state.pos.x < 16 && state.pos.y < 16 && state.pos.z < 16)
  {
    lastState = state;
    rol::nextAwStep(state);
  }

  EXPECT_EQ(lastState.pos.x, 7);
  EXPECT_EQ(lastState.pos.y, 11);
  EXPECT_EQ(lastState.pos.z, 15);
}

TEST(AmantidesWoo, StartInsideCube)
{
  auto state = rol::initAmantidesWoo(makeFp3(4.6f, 1.3f, 3.2f), makeFp3(0.3f, 0.5f, 0.7f), 16);

  EXPECT_EQ(state.pos.x, 4);
  EXPECT_EQ(state.pos.y, 1);
  EXPECT_EQ(state.pos.z, 3);

  auto lastState = state;
  while (state.pos.x < 16 && state.pos.y < 16 && state.pos.z < 16)
  {
    lastState = state;
    rol::nextAwStep(state);
  }

  EXPECT_EQ(lastState.pos.x, 10);
  EXPECT_EQ(lastState.pos.y, 10);
  EXPECT_EQ(lastState.pos.z, 15);
}

TEST(AmantidesWoo, StartOutsideCube)
{
  auto state = rol::initAmantidesWoo(makeFp3(-4.6f, -1.3f, -3.2f), makeFp3(0.3f, 0.5f, 0.7f), 16);

  EXPECT_EQ(state.pos.x, 0);
  EXPECT_EQ(state.pos.y, 6);
  EXPECT_EQ(state.pos.z, 7);

  auto lastState = state;
  while (state.pos.x < 16 && state.pos.y < 16 && state.pos.z < 16)
  {
    lastState = state;
    rol::nextAwStep(state);
  }

  EXPECT_EQ(lastState.pos.x, 3);
  EXPECT_EQ(lastState.pos.y, 12);
  EXPECT_EQ(lastState.pos.z, 15);
}

TEST(AmantidesWoo, StartOutsideWithOffsetNegDir)
{
  auto state = rol::initAmantidesWoo(makeFp3(-10.f, 8.f, 8.f), makeFp3(1.f, 0.5f, -0.5f), 16);
  auto lastState = state;
  while (state.pos.x >= 0 && state.pos.x < 16
    && state.pos.y >= 0 && state.pos.y < 16
    && state.pos.z >= 0 && state.pos.z < 16)
  {
    lastState = state;
    rol::nextAwStep(state);
  }

  EXPECT_EQ(lastState.pos.x, 5);
  EXPECT_EQ(lastState.pos.y, 15);
  EXPECT_EQ(lastState.pos.z, 0);
}
