#pragma once

#include <memory>
#include <vector>

#include "renderer.h"

namespace rol
{
  class CpuRenderer final : public Renderer
  {
  public:
    CpuRenderer(size_t w, size_t h);

    void render(const Game& game, const Camera& camera) override;

  private:
    const float3* imageData() const override;

    void renderPixel(int ix, int iy, const std::vector<float>& xspace, const std::vector<float>& yspace, const Camera& camera);

    std::unique_ptr<float3[]> m_imageData;
  };
}