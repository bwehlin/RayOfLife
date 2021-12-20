#include "renderer.h"

rol::Renderer::Renderer(size_t w, size_t h)
  : m_image(w, h)
{

}

void
rol::Renderer::saveFrameBmp(const char* filename)
{
  writeChannelsToImage();
  auto const& view = boost::gil::view(m_image);
  boost::gil::write_view(filename, view, boost::gil::bmp_tag());
}

void
rol::Renderer::writeChannelsToImage()
{
  if (m_imageIsCurrent)
  {
    return;
  }

  auto const& dstImageView = boost::gil::view(m_image);

  const float3* data = imageData();

  auto w = width();
  auto h = height();

  // Sure, we could parallelize this, but in the grand scheme of things, how much
  // time will an in-memory copy+scaling of w*h pixels (parallelizable) take
  // compared to a memory->disk transfer (not parallelizable)?
  for (size_t y = 0ul; y < h; ++y)
  {
    auto baseIdx = y * w;
    auto yinv = h - 1 - y; // GIL images are 'upside down' compared to our view.
    for (size_t x = 0ul; x < w; ++x)
    {
      auto idx = baseIdx + x;
      
      auto fr = data[idx].x * 255.;
      auto fg = data[idx].y * 255.;
      auto fb = data[idx].z * 255.;

      auto r = std::clamp(static_cast<int>(fr), 0, 255);
      auto g = std::clamp(static_cast<int>(fg), 0, 255);
      auto b = std::clamp(static_cast<int>(fb), 0, 255);

      dstImageView(x, yinv) = boost::gil::rgb8_pixel_t(
        static_cast<unsigned char>(r),
        static_cast<unsigned char>(g),
        static_cast<unsigned char>(b));
    }
  }

  m_imageIsCurrent = true;
}

