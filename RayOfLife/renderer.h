#pragma once

#include <cuda_runtime.h>
#include <vector>

#include "support.cuh"
#include "camera.cuh"

namespace rol
{
  class Game;

  class Renderer
  {
  public:
    Renderer(size_t w, size_t h);

    virtual ~Renderer() = default;
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    [[nodiscard]] size_t width()  const { return m_width;  }
    [[nodiscard]] size_t height() const { return m_height; }

    // Returns width / height.
    [[nodiscard]] fptype aspectRatio() const { return static_cast<fptype>(width()) / static_cast<fptype>(height()); }

    void saveFrameBmp(const char* filename);

    void render(const Game& game, const Camera& camera);

    void setMaxDepth(int depth) noexcept { m_maxDepth = depth; }
    [[nodiscard]] int maxDepth() const noexcept { return m_maxDepth; }

    void setSubpixelCount(int subpixels) noexcept { m_subpixelCount = subpixels; }
    [[nodiscard]] int subpixelCount() const noexcept { return m_subpixelCount; }


    [[nodiscard]] fptype3 sphereColor() const noexcept { return m_sphereColor; }
    [[nodiscard]] fptype sphereDiffuseC() const noexcept { return m_sphereDiffuseC; }
    [[nodiscard]] fptype sphereSpecularC() const noexcept { return m_sphereSpecularC; }
    [[nodiscard]] fptype sphereReflection() const noexcept { return m_sphereReflection; }
    
  private:
    virtual void produceFrame(const Game& game, const Camera& camera,
      const fptype2& screenMin, const fptype2& screenMax) = 0;

    // Returns a w*h array of fptype (0-1). The array should be strided on w (row major).
    // The x component represents red, y green and z blue. In other words, the array can
    // be thought of as RGB scanlines stacked one after another.
    [[nodiscard]] virtual const fptype3* imageData() const = 0;

    int m_maxDepth = 5;
    int m_subpixelCount = 1;

    fptype3 m_sphereColor = { 1.f, .572f, .184f };
    fptype m_sphereDiffuseC = 1.f;
    fptype m_sphereSpecularC = 1.f;
    fptype m_sphereReflection = 0.5f;

    size_t m_width;
    size_t m_height;
    
  };
}