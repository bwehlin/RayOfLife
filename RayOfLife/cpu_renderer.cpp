#include "cpu_renderer.h"
#include <stdexcept>

rol::CpuRenderer::CpuRenderer(size_t w, size_t h)
  : Renderer(w, h)
{
  m_imageData = std::make_unique<float3[]>(w * h);
}

const float3*
rol::CpuRenderer::imageData() const
{
  return m_imageData.get();
}
