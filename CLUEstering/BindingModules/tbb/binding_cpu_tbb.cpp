
#include <alpaka/alpaka.hpp>
#include <vector>

#include "Run.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace alpaka_tbb_async {

  void listDevices(const std::string& backend) {
    const char tab = '\t';
    const std::vector<Device> devices = alpaka::getDevs(alpaka::Platform<Acc1D>());
    if (devices.empty()) {
      std::cout << "No devices found for the " << backend << " backend." << std::endl;
      return;
    } else {
      std::cout << backend << " devices found: \n";
      for (size_t i{}; i < devices.size(); ++i) {
        std::cout << tab << "Device " << i << ": " << alpaka::getName(devices[i]) << '\n';
      }
    }
  }

  template <typename Kernel>
  void mainRun(float dc,
               float rhoc,
               float dm,
               int pPBin,
               py::array_t<float> data,
               py::array_t<int> results,
               const Kernel& kernel,
               int Ndim,
               uint32_t n_points,
               size_t block_size,
               size_t device_id) {
    auto rData = data.request();
    float* pData = static_cast<float*>(rData.ptr);
    auto rResults = results.request();
    int* pResults = static_cast<int*>(rResults.ptr);

    const auto dev_acc = alpaka::getDevByIdx(alpaka::Platform<Acc1D>{}, device_id);

    // Create the queue
    Queue queue_(dev_acc);

    // Running the clustering algorithm //
    switch (Ndim) {
      [[unlikely]] case (1):
        run<1, Kernel>(dc,
                       rhoc,
                       dm,
                       pPBin,
                       std::make_tuple(pData, pResults),
                       PointInfo<1>{n_points},
                       kernel,
                       queue_,
                       block_size);
        return;
      [[likely]] case (2):
        run<2, Kernel>(dc,
                       rhoc,
                       dm,
                       pPBin,
                       std::make_tuple(pData, pResults),
                       PointInfo<2>{n_points},
                       kernel,
                       queue_,
                       block_size);
        return;
      [[likely]] case (3):
        run<3, Kernel>(dc,
                       rhoc,
                       dm,
                       pPBin,
                       std::make_tuple(pData, pResults),
                       PointInfo<3>{n_points},
                       kernel,
                       queue_,
                       block_size);
        return;
      [[unlikely]] case (4):
        run<4, Kernel>(dc,
                       rhoc,
                       dm,
                       pPBin,
                       std::make_tuple(pData, pResults),
                       PointInfo<4>{n_points},
                       kernel,
                       queue_,
                       block_size);
        return;
      [[unlikely]] case (5):
        run<5, Kernel>(dc,
                       rhoc,
                       dm,
                       pPBin,
                       std::make_tuple(pData, pResults),
                       PointInfo<5>{n_points},
                       kernel,
                       queue_,
                       block_size);
        return;
      [[unlikely]] case (6):
        run<6, Kernel>(dc,
                       rhoc,
                       dm,
                       pPBin,
                       std::make_tuple(pData, pResults),
                       PointInfo<6>{n_points},
                       kernel,
                       queue_,
                       block_size);
        return;
      [[unlikely]] case (7):
        run<7, Kernel>(dc,
                       rhoc,
                       dm,
                       pPBin,
                       std::make_tuple(pData, pResults),
                       PointInfo<7>{n_points},
                       kernel,
                       queue_,
                       block_size);
        return;
      [[unlikely]] case (8):
        run<8, Kernel>(dc,
                       rhoc,
                       dm,
                       pPBin,
                       std::make_tuple(pData, pResults),
                       PointInfo<8>{n_points},
                       kernel,
                       queue_,
                       block_size);
        return;
      [[unlikely]] case (9):
        run<9, Kernel>(dc,
                       rhoc,
                       dm,
                       pPBin,
                       std::make_tuple(pData, pResults),
                       PointInfo<9>{n_points},
                       kernel,
                       queue_,
                       block_size);
        return;
      [[unlikely]] case (10):
        run<10, Kernel>(dc,
                        rhoc,
                        dm,
                        pPBin,
                        std::make_tuple(pData, pResults),
                        PointInfo<10>{n_points},
                        kernel,
                        queue_,
                        block_size);
        return;
      [[unlikely]] default:
        std::cout << "This library only works up to 10 dimensions\n";
    }
  }

  PYBIND11_MODULE(CLUE_CPU_TBB, m) {
    m.doc() = "Binding of the CLUE algorithm running on CPU with TBB";

    m.def("listDevices", &listDevices, "List the available devices for the TBB backend");
    m.def("mainRun",
          pybind11::overload_cast<float,
                                  float,
                                  float,
                                  int,
                                  py::array_t<float>,
                                  py::array_t<int>,
                                  const FlatKernel&,
                                  int,
                                  uint32_t,
                                  size_t,
                                  size_t>(&mainRun<FlatKernel>),
          "mainRun");
    m.def("mainRun",
          pybind11::overload_cast<float,
                                  float,
                                  float,
                                  int,
                                  py::array_t<float>,
                                  py::array_t<int>,
                                  const ExponentialKernel&,
                                  int,
                                  uint32_t,
                                  size_t,
                                  size_t>(&mainRun<ExponentialKernel>),
          "mainRun");
    m.def("mainRun",
          pybind11::overload_cast<float,
                                  float,
                                  float,
                                  int,
                                  py::array_t<float>,
                                  py::array_t<int>,
                                  const GaussianKernel&,
                                  int,
                                  uint32_t,
                                  size_t,
                                  size_t>(&mainRun<GaussianKernel>),
          "mainRun");
  }
};  // namespace alpaka_tbb_async
