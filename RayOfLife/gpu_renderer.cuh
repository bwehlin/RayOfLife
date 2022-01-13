#pragma once

#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "support.cuh"
#include "amantides_woo.cuh"
#include "scene_data.cuh"
#include "intersect.cuh"

#include "renderer.h"

namespace rol
{
  class GpuRenderer final : public Renderer
  {
  public:
    GpuRenderer(size_t w, size_t h);
    ~GpuRenderer();

  private:
    void produceFrame(const Game& game, const Camera& camera,
      const std::vector<fptype>& xspace, const std::vector<fptype>& yspace) override;
    const fptype3* imageData() const override;

    fptype3* m_imageData;
    bool* m_d_game;

    SceneData m_scene;
  };
}
