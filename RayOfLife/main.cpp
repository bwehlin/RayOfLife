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
#include <iostream>
#include <boost/program_options.hpp>

#include "cpu_game.h"
#include "cpu_renderer.h"
#include "gpu_renderer.cuh"

namespace
{
  void printDensity(const rol::Game& game)
  {
    long livingCells = 0;
    for (auto z = 0; z < game.cellsPerDim(); ++z)
    {
      for (auto y = 0; y < game.cellsPerDim(); ++y)
      {
        for (auto x = 0; x < game.cellsPerDim(); ++x)
        {
          if (game.isAlive(x, y, z))
          {
            ++livingCells;
          }
        }
      }
    }

    auto totalCells = game.cellsPerDim() * game.cellsPerDim() * game.cellsPerDim();
    std::cout << "Game density: " << livingCells << "/" << totalCells << " = " << static_cast<double>(livingCells) / static_cast<double>(totalCells) << '\n';
  }

  struct CmdOpts
  {
    enum class RendererType { CPU, GPU } renderer = RendererType::GPU;
    int w = 1024;
    int h = 768;
    int maxdepth = 50;
    int subpixels = 4;
  };

  std::pair<CmdOpts, bool> parseArgs(int argc, char** argv)
  {
    // Code based on https://www.boost.org/doc/libs/1_78_0/doc/html/program_options/tutorial.html
    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
      ("help", "display this message")
      ("renderer", po::value<std::string>(), "gpu or cpu (default is gpu)")
      ("w", po::value<int>(), "width (default 1024)")
      ("h", po::value<int>(), "height (default 768)")
      ("maxdepth", po::value<int>(), "maximum reflection depth (default: 50)")
      ("subpixels", po::value<int>(), "subpixel grid (default: 4)")
      ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    CmdOpts opts;

    if (vm.count("help"))
    {
      std::cout << desc << '\n';
      return std::make_pair(opts, false);
    }

    if (vm.count("renderer"))
    {
      auto rendererVal = vm["renderer"].as<std::string>();
      if (rendererVal == "gpu")
      {
        opts.renderer = CmdOpts::RendererType::GPU;
      }
      else if (rendererVal == "cpu")
      {
        opts.renderer = CmdOpts::RendererType::CPU;
      }
      else
      {
        throw std::runtime_error("Renderer must be either 'cpu' or 'gpu'.");
      }
    }

    if (vm.count("w"))
    {
      auto val = vm["w"].as<int>();
      if (val <= 0)
      {
        throw std::runtime_error("Must have positive width.");
      }
      opts.w = val;
    }

    if (vm.count("h"))
    {
      auto val = vm["h"].as<int>();
      if (val <= 0)
      {
        throw std::runtime_error("Must have positive height.");
      }
      opts.h = val;
    }

    if (vm.count("w") && vm.count("w") != vm.count("h"))
    {
      throw std::runtime_error("Please specify both w and h.");
    }

    if (vm.count("maxdepth"))
    {
      auto val = vm["maxdepth"].as<int>();
      if (val <= 0)
      {
        throw std::runtime_error("Must have positive maxdepth.");
      }
      opts.maxdepth = val;
    }

    if (vm.count("subpixels"))
    {
      auto val = vm["subpixels"].as<int>();
      if (val <= 0)
      {
        throw std::runtime_error("Must have positive subpixels.");
      }
      opts.subpixels = val;
    }

    return std::make_pair(opts, true);
  }
  
}

int main(int argc, char** argv)
{
  try
  {
    auto const [opts, cont] = parseArgs(argc, argv);
    if (!cont)
    {
      return EXIT_SUCCESS;
    }

    std::unique_ptr<rol::Renderer> renderer = [&opts]() -> std::unique_ptr<rol::Renderer> {
      switch (opts.renderer)
      {
      case CmdOpts::RendererType::CPU: return std::make_unique<rol::CpuRenderer>(opts.w, opts.h);
      case CmdOpts::RendererType::GPU:
      default: return std::make_unique<rol::GpuRenderer>(opts.w, opts.h);
      }
    }();

    rol::CpuGame game(rol::makeTransitionRule(4,7,5,7), 16, 16);

    auto camera = rol::makeCamera(makeFp3(-0.2f, 10.f, 10.f), makeFp3(0.3f, 0.5f, 0.7f));

    game.initRandomPrimordialSoup(2360);

    renderer->setMaxDepth(opts.maxdepth);
    renderer->setSubpixelCount(opts.subpixels);

    //renderer.render(game, camera);
    //printDensity(game);
    //renderer.saveFrameBmp("frame0.bmp");

    for (itype i = 0; i < 3; ++i)
    {
      game.evolve();
      printDensity(game);

      renderer->render(game, camera);

      auto frameTitle = "frame" + std::to_string(i + 1) + ".bmp";
      renderer->saveFrameBmp(frameTitle.c_str());
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
