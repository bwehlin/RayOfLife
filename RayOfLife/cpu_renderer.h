#pragma once

#include <memory>

#include "renderer.h"

namespace rol
{
  class CpuRenderer final : public Renderer
  {
  public:
    CpuRenderer(size_t w, size_t h);

  private:
    const float3* imageData() const override;

    std::unique_ptr<float3[]> m_imageData;
  };
}