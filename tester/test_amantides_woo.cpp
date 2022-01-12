#include "pch.h"

#include "../RayOfLife/amantides_woo.cuh"

TEST(AmantidesWoo, StartAtOrigin) 
{
  auto state = rol::initAmantidesWoo(make_float3(0.f, 0.f, 0.f), make_float3(0.3f, 0.5f, 0.7f));
  auto lastState = state;
  do
  {
    lastState = state;
    rol::nextAwStep(state);
  } while (state.pos.x < 16 && state.pos.y < 16 && state.pos.z < 16);

  EXPECT_EQ(lastState.pos.x, 6);
  EXPECT_EQ(lastState.pos.y, 11);
  EXPECT_EQ(lastState.pos.z, 15);
}

TEST(AmantidesWoo, StartAtOriginWithOffset)
{
  auto state = rol::initAmantidesWoo(make_float3(0.5f, 0.3f, 0.2f), make_float3(0.3f, 0.5f, 0.7f));
  auto lastState = state;
  do
  {
    lastState = state;
    rol::nextAwStep(state);
  } while (state.pos.x < 16 && state.pos.y < 16 && state.pos.z < 16);

  EXPECT_EQ(lastState.pos.x, 7);
  EXPECT_EQ(lastState.pos.y, 11);
  EXPECT_EQ(lastState.pos.z, 15);
}
