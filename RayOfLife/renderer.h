#pragma once

#include <cuda_runtime.h>
#include "support.cuh"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4996)
#endif

#include <boost/gil/image.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/gil/extension/io/bmp.hpp>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

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

    [[nodiscard]] size_t width()  const { return m_image.width();  }
    [[nodiscard]] size_t height() const { return m_image.height(); }

    // Returns width / height.
    [[nodiscard]] fptype aspectRatio() const { return static_cast<fptype>(width()) / static_cast<fptype>(height()); }

    void saveFrameBmp(const char* filename);

    void render(const Game& game, const Camera& camera);

    void setMaxDepth(int depth) { m_maxDepth = depth; }
    [[nodiscard]] int maxDepth() const { return m_maxDepth; }

    [[nodiscard]] fptype3 sphereColor() const noexcept { return m_sphereColor; }
    [[nodiscard]] fptype sphereDiffuseC() const noexcept { return m_sphereDiffuseC; }
    [[nodiscard]] fptype sphereSpecularC() const noexcept { return m_sphereSpecularC; }
    [[nodiscard]] fptype sphereReflection() const noexcept { return m_sphereReflection; }
    
  private:
    virtual void produceFrame(const Game& game, const Camera& camera) = 0;

    // Returns a w*h array of fptype (0-1). The array should be strided on w (row major).
    // The x component represents red, y green and z blue. In other words, the array can
    // be thought of as RGB scanlines stacked one after another.
    [[nodiscard]] virtual const fptype3* imageData() const = 0;

    // Writes channel data from the derived Renderer into m_image (for subsequent export).
    void writeChannelsToImage();

    boost::gil::rgb8_image_t m_image;
    bool m_imageIsCurrent = false;

    int m_maxDepth = 5;

    fptype3 m_sphereColor = { 1.f, .572f, .184f };
    fptype m_sphereDiffuseC = 1.f;
    fptype m_sphereSpecularC = 1.f;
    fptype m_sphereReflection = 0.5f;
    
  };
}