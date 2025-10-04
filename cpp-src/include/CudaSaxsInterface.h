#ifndef CUDASAXS_INTERFACE_H
#define CUDASAXS_INTERFACE_H

#include <string>
#include <vector>
#include <pybind11/pybind11.h>

namespace py = pybind11;

struct CudaSaxsConfig
{
    std::string topology_path;
    std::string trajectory_path;
    std::vector<int> grid_shape;
    std::vector<int> scaled_grid_shape;
    int begin_frame{0};
    int end_frame{0};
    int frame_stride{1};
    std::string output_path;
    int spline_order{4};
    float scale_factor{1.0f};
    float bin_size{0.0f};
    float qcut{0.0f};
    std::string water_model;
    int sodium_atoms{0};
    int chlorine_atoms{0};
    std::string simulation_ensemble;
};

struct CudaSaxsResult
{
    int nx{0};
    int ny{0};
    int nz{0};
    int nnx{0};
    int nny{0};
    int nnz{0};
    std::string summary;
};

CudaSaxsResult run_cuda_saxs(py::object Topol, const CudaSaxsConfig &config);

#endif // CUDASAXS_INTERFACE_H
