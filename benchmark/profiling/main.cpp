
#include <alpaka/alpaka.hpp>
#include <algorithm>
#include <chrono>
#include <vector>

#include "CLUEstering/CLUEstering.hpp"
#include "CLUEstering/DataFormats/Points.hpp"
#include "CLUEstering/DataFormats/alpaka/PointsAlpaka.hpp"

#include "CLUEstering/utility/read_csv.hpp"

namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE {
  void run(const std::string& input_file) {
    auto coords = read_csv<float, 2>(input_file);
    const auto n_points = coords.size() / 3;
    std::vector<int> results(2 * n_points);

    const auto dev_acc = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, 0u);
    Queue queue_(dev_acc);

    PointsSoA<2> h_points(coords.data(), results.data(), PointInfo<2>{static_cast<uint32_t>(n_points)});
    PointsAlpaka<2> d_points(queue_, n_points);

    const float dc{1.5f}, rhoc{10.f}, outlier{1.5f};
    const int pPBin{128};
    CLUEAlgoAlpaka<2> algo(dc, rhoc, outlier, pPBin, queue_);

    const std::size_t block_size{256};
    algo.make_clusters(h_points, d_points, FlatKernel{.5f}, queue_, block_size);
    auto clusters = algo.getClusters(h_points);
  }
};  // namespace ALPAKA_ACCELERATOR_NAMESPACE_CLUE

int main(int, char* argv[]) {
  auto input_file{std::string(argv[1])};
  ALPAKA_ACCELERATOR_NAMESPACE_CLUE::run(input_file);
}
