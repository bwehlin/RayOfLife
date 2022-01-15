#include <cstdlib>
#include <iostream>
#include <numeric>
#include <boost/program_options.hpp>

#include "cpu_game.h"
#include "cpu_renderer.h"
#include "gpu_renderer.cuh"

#include <omp.h>

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
    int ompthreads = -1;
    int measureframe = -1;
    int blockdim = 16;
    int frames = 5;
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
      ("blockdim", po::value<int>(), "GPU block dimension (default: 16)")
      ("ompthreads", po::value<int>(), "set OpenMP threads for CPU implementation (default: -1, max threads)")
      ("measureframe", po::value<int>(), "measure time for a specific frame number (default: -1, off)")
      ("frames", po::value<int>(), "number of frames to render (default: 5)")
      ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    omp_set_num_threads(1);
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

    if (vm.count("w") && !vm.count("h"))
    {
      opts.h = opts.w;
    }

    if (vm.count("h") && !vm.count("w"))
    {
      opts.w = opts.h;
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

    if (vm.count("blockdim"))
    {
      auto val = vm["blockdim"].as<int>();
      if (val <= 0)
      {
        throw std::runtime_error("Must have positive blockdim.");
      }
      opts.blockdim = val;
    }

    if (vm.count("ompthreads"))
    {
      auto val = vm["ompthreads"].as<int>();
      opts.ompthreads = val;
    }

    if (vm.count("measureframe"))
    {
      auto val = vm["measureframe"].as<int>();
      opts.measureframe = val;
    }

    if (vm.count("frames"))
    {
      auto val = vm["frames"].as<int>();
      if (val <= 0)
      {
        throw std::runtime_error("Must have positive frames.");
      }
      opts.frames = val;
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
      default: return std::make_unique<rol::GpuRenderer>(opts.w, opts.h, opts.blockdim);
      }
    }();

    if (opts.ompthreads > -1)
    {
      omp_set_num_threads(opts.ompthreads);
    }

    rol::CpuGame game(rol::makeTransitionRule(4,7,5,7), 16, 16);

    auto camera = rol::makeCamera(makeFp3(-0.2f, 10.f, 10.f), makeFp3(0.3f, 0.5f, 0.7f));

    game.initRandomPrimordialSoup(2360);

    renderer->setMaxDepth(opts.maxdepth);
    renderer->setSubpixelCount(opts.subpixels);

    if (opts.measureframe == -1)
    {
      printDensity(game);
      renderer->render(game, camera);
      renderer->saveFrameBmp("frame0.bmp");
    }

    for (itype i = 0; i < opts.frames - 1; ++i)
    {
      game.evolve();
      printDensity(game);

      if (opts.measureframe == -1)
      {
        renderer->render(game, camera);
      }
      else
      {
        if (opts.measureframe == i)
        {
          renderer->render(game, camera);
        }
      }


      if (opts.measureframe == i)
      {
        int nMeasurements = 10;
        std::vector<double> measurements(nMeasurements);
        for (int j = 0; j < nMeasurements; ++j)
        {
          renderer->render(game, camera);
          measurements[j] = renderer->lastFrameTimeSeconds();
        }

        auto total = std::accumulate(measurements.begin(), measurements.end(), 0.);
        auto mean = total / static_cast<double>(nMeasurements);
        
        auto stdev = 0.;
        for (auto j = 0; j < nMeasurements; ++j)
        {
          stdev += std::pow(measurements[j] - mean, 2.);
        }
        stdev /= (nMeasurements - 1);
        stdev = std::sqrt(stdev);

        std::cout << "Over " << nMeasurements << " runs:\n"
          << "  mean: " << std::scientific << mean << "s\n"
          << "  std:  " << std::scientific << stdev << "s\n";
      }
      else
      {
        renderer->render(game, camera);
      }

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
