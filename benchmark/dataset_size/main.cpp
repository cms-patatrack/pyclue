
#include <alpaka/alpaka.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>

#include "CLUEstering/CLUEstering.hpp"
#include "CLUEstering/DataFormats/Points.hpp"
#include "CLUEstering/DataFormats/alpaka/PointsAlpaka.hpp"

#include "CLUEstering/utility/read_csv.hpp"

#include <pybind11/embed.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MAKE_OPAQUE(std::vector<int>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);

struct TimeMeasures {
  std::vector<int> sizes;
  std::vector<float> time_averages;
  std::vector<float> time_stddevs;

  TimeMeasures(size_t size) : sizes(size), time_averages(size), time_stddevs(size) {}
};

float mean(const std::vector<float>& values) {
  return std::accumulate(values.begin(), values.end(), 0.f) / values.size();
}

float stddev(const std::vector<float>& values) {
  auto mean_ = mean(values);
  auto view{values | std::views::transform(
                         [mean_](auto x) -> float { return (x - mean_) * (x - mean_); })};
  auto sqSize = values.size() * (values.size() - 1);
  return std::sqrt(std::accumulate(view.begin(), view.end(), 0.f) / sqSize);
}

std::vector<std::string> GetFiles(int min, int max) {
  std::vector<std::string> files(max - min + 1);
  std::generate(files.begin(), files.end(), [n = min]() mutable -> std::string {
    auto filename =
        "../../data/data_" + std::to_string(static_cast<int>(std::pow(2, n))) + ".csv";
    ++n;
    return filename;
  });
  return files;
}

void plot(const TimeMeasures& measures, const std::string& filename) {
  py::scoped_interpreter guard{};
  py::module plt = py::module::import("matplotlib.pyplot");
  py::bind_vector<std::vector<int>>(plt, "VectorInt");
  py::bind_vector<std::vector<float>>(plt, "VectorFloat");
  plt.attr("errorbar")(
      measures.sizes, measures.time_averages, measures.time_stddevs, "fmt"_a = "r--^");
  plt.attr("xlabel")("Number of points");
  plt.attr("ylabel")("Execution time (ms)");
  plt.attr("grid")("ls"_a = "--", "lw"_a = .5);
  plt.attr("savefig")(filename);
}

void to_csv(const TimeMeasures& measures, const std::string& filename) {
  std::ofstream file{filename};
  if (!file.is_open()) {
    std::cerr << "Error opening file " << filename << std::endl;
    return;
  }

  file << "size,avg,std\n";
  for (auto i = 0ul; i < measures.sizes.size(); ++i) {
    file << measures.sizes[i] << "," << measures.time_averages[i] << ","
         << measures.time_stddevs[i] << "\n";
  }
  file.close();
}

namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE {
  void run(const std::string& input_file) {
    auto coords = read_csv<float, 2>(input_file);
    const auto n_points = coords.size() / 3;
    std::vector<int> results(2 * n_points);

    const auto dev_acc = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
    Queue queue_(dev_acc);

    PointsSoA<2> h_points(coords.data(), results.data(), PointInfo<2>{(uint32_t)n_points});
    PointsAlpaka<2> d_points(queue_, n_points);

    const float dc{1.5f}, rhoc{10.f}, outlier{1.5f};
    const int pPBin{128};
    CLUEAlgoAlpaka<2> algo(dc, rhoc, outlier, pPBin, queue_);

    const std::size_t block_size{256};
    algo.make_clusters(h_points, d_points, FlatKernel{.5f}, queue_, block_size);
  }

};  // namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE

namespace cluetest = ALPAKA_ACCELERATOR_NAMESPACE_CLUE;

int main(int argc, char* argv[]) {
  auto min = std::stoi(argv[1]);
  auto max = std::stoi(argv[2]);
  auto range = max - min + 1;
  auto files = GetFiles(min, max);

  std::string oFilename{"measures.csv"};
  if (argc == 4) {
    oFilename = argv[3];
  }

  TimeMeasures measures(range);
  auto& sizes = measures.sizes;
  auto& time_averages = measures.time_averages;
  auto& time_stddevs = measures.time_stddevs;
  auto nruns{10};
  std::transform(files.begin(), files.end(), sizes.begin(), [](auto file) {
    auto first_it = std::find(file.begin(), file.end(), '_') + 1;
    auto first = std::distance(file.begin(), first_it);
    auto len = std::distance(first_it, std::find(file.begin(), file.end(), '.') - 1);
    return std::stoi(file.substr(first, len));
  });

  auto avgIt = time_averages.begin();
  auto stdIt = time_stddevs.begin();
  std::for_each(
      files.begin(), files.end(), [nruns, &avgIt, &stdIt](const auto& file) -> void {
        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();
        std::vector<float> times(nruns);
        for (auto i = 0; i < nruns; ++i) {
          start = std::chrono::high_resolution_clock::now();
          cluetest::run(file);
          end = std::chrono::high_resolution_clock::now();
          auto duration =
              std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
          times[i] = duration;
        }
        *avgIt++ = mean(times);
        *stdIt++ = stddev(times);
      });

  auto figname = oFilename.substr(0, oFilename.find_last_of('.')) + ".pdf";
  plot(measures, figname);
  to_csv(measures, oFilename);
}
