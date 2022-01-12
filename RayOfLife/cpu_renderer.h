#pragma once

#include <memory>
#include <vector>

#include "support.cuh"

#include "renderer.h"

namespace rol
{
  class CpuRenderer final : public Renderer
  {
  public:
    CpuRenderer(size_t w, size_t h);

    void render(const Game& game, const Camera& camera) override;

  private:
    const fptype3* imageData() const override;

    void renderPixel(int ix, int iy, const std::vector<fptype>& xspace, const std::vector<fptype>& yspace, const Camera& camera);

    std::unique_ptr<fptype3[]> m_imageData;
  };
}