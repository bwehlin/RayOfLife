#pragma once

#include <memory>
#include <vector>

#include "support.cuh"
#include "amantides_woo.cuh"
#include "scene_data.cuh"
#include "intersect.cuh"

#include "renderer.h"

namespace rol
{
  class CpuRenderer final : public Renderer
  {
  public:
    CpuRenderer(size_t w, size_t h);

  private:
    void produceFrame(const Game& game, const Camera& camera) override;
    const fptype3* imageData() const override;

    void renderPixel(int ix, int iy, fptype x, fptype y, const Game& game, const Camera& camera);
    RayIntersection castRay(AmantidesWooState& awstate, fptype3 rayOrigin, fptype3 rayDirection, const Game& game, const Camera& camera);

    std::unique_ptr<fptype3[]> m_imageData;
    SceneData m_scene;
  };
}