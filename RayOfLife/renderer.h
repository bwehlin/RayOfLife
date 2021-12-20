#pragma once

#include <cuda_runtime.h>

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

namespace rol
{
  class Renderer
  {
  public:
    Renderer(size_t w, size_t h);

    virtual ~Renderer() = default;
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    size_t width()  const { return m_image.width();  }
    size_t height() const { return m_image.height(); }

    // Returns width / height.
    float aspectRatio() const { return width() / height(); }

    void saveFrameBmp(const char* filename);

  private:
    // Returns a w*h array of float (0-1). The array should be strided on w (row major).
    // The x component represents red, y green and z blue. In other words, the array can
    // be thought of as RGB scanlines stacked one after another.
    virtual const float3* imageData() const = 0;

    // Writes channel data from the derived Renderer into m_image (for subsequent export).
    void writeChannelsToImage();

    boost::gil::rgb8_image_t m_image;
    bool m_imageIsCurrent = false;
  };
}