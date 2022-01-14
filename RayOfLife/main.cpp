/*
This file contains a C++ port of the 'very simple ray tracing engine' at https://gist.github.com/rossant/6046463, which is licensed as follows:

MIT License
Copyright (c) 2017 Cyrille Rossant
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/


#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <numeric>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

#include <boost/math/constants/constants.hpp>

#include "scene_object.h"
#include "intersect.cuh"
#include "vector_math.cuh"

#include "cpu_game.h"
#include "cpu_renderer.h"
#include "gpu_renderer.cuh"
#include "amantides_woo.cuh"

int main(int, char**)
{
  try
  {
    rol::GpuGame game(rol::makeTransitionRule(4,7,5,7), 16);

    auto camera = rol::makeCamera(makeFp3(-15.f, 8.f, 8.f), makeFp3(0.3f, 0.5f, 0.7f));

    rol::CpuRenderer renderer(320, 240);
    game.initRandomPrimordialSoup(236011);
    renderer.setMaxDepth(10000000);
    renderer.render(game, camera);
    renderer.saveFrameBmp("frame0.bmp");

    for (int i = 0; i < 1; ++i)
    {
      std::cout << "Working on game " << i + 1 << std::endl;

      game.evolve();
      renderer.render(game, camera);

      auto frameTitle = "frame" + std::to_string(i + 1) + ".bmp";
      renderer.saveFrameBmp(frameTitle.c_str());
    }

    return EXIT_SUCCESS;
  }
  catch (const std::exception& ex)
  {
    std::cerr << ex.what() << '\n';
  }
  catch (...)
  {
    std::cerr << "Unknown exception." << '\n';
  }

  return EXIT_FAILURE;
}
