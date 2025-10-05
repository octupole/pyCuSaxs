#include "CudaSaxsInterface.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace
{

    py::dict run_with_kwargs(
        py::object obj_topology,
        const std::string &topology,
        const std::string &trajectory,
        const std::vector<int> &grid,
        const std::vector<int> &scaled_grid,
        int begin,
        int end,
        int stride,
        const std::string &output,
        int order,
        double scale_factor,
        double bin_size,
        double qcut,
        const std::string &water_model,
        int sodium,
        int chlorine,
        const std::string &simulation)
    {
        try
        {
            // Input validation
            if (grid.size() != 3)
            {
                throw std::invalid_argument("Grid must have exactly 3 dimensions (nx, ny, nz)");
            }
            if (grid[0] <= 0 || grid[1] <= 0 || grid[2] <= 0)
            {
                throw std::invalid_argument("Grid dimensions must be positive");
            }
            if (!scaled_grid.empty() && scaled_grid.size() != 3)
            {
                throw std::invalid_argument("Scaled grid must have exactly 3 dimensions or be empty");
            }
            if (!scaled_grid.empty() && (scaled_grid[0] <= 0 || scaled_grid[1] <= 0 || scaled_grid[2] <= 0))
            {
                throw std::invalid_argument("Scaled grid dimensions must be positive");
            }
            if (stride <= 0)
            {
                throw std::invalid_argument("Stride must be positive");
            }
            if (order < 1 || order > 8)
            {
                throw std::invalid_argument("Spline order must be between 1 and 8");
            }
            if (bin_size < 0.0)
            {
                throw std::invalid_argument("Bin size must be non-negative");
            }
            if (qcut < 0.0)
            {
                throw std::invalid_argument("Q cutoff must be non-negative");
            }
            if (scale_factor <= 0.0)
            {
                throw std::invalid_argument("Scale factor must be positive");
            }
            if (begin < 0 || end < 0)
            {
                throw std::invalid_argument("Frame indices must be non-negative");
            }
            if (end > 0 && begin >= end)
            {
                throw std::invalid_argument("Begin frame must be less than end frame");
            }

            CudaSaxsConfig config;

            config.topology_path = topology;
            config.trajectory_path = trajectory;
            config.grid_shape = grid;
            config.scaled_grid_shape = scaled_grid;
            config.begin_frame = begin;
            config.end_frame = end;
            config.frame_stride = stride;
            config.output_path = output;
            config.spline_order = order;
            config.scale_factor = static_cast<float>(scale_factor);
            config.bin_size = static_cast<float>(bin_size);
            config.qcut = static_cast<float>(qcut);
            config.water_model = water_model;
            config.sodium_atoms = sodium;
            config.chlorine_atoms = chlorine;
            config.simulation_ensemble = simulation;

            const auto result = run_cuda_saxs(obj_topology, config);

            py::dict response;
            response["summary"] = result.summary;
            response["grid"] = std::vector<int>{result.nx, result.ny, result.nz};
            response["scaled_grid"] = std::vector<int>{result.nnx, result.nny, result.nnz};
            return response;
        }
        catch (const std::invalid_argument &e)
        {
            throw py::value_error(e.what());
        }
        catch (const std::runtime_error &e)
        {
            // Translate C++ runtime_error to Python RuntimeError
            PyErr_SetString(PyExc_RuntimeError, e.what());
            throw py::error_already_set();
        }
        catch (const std::exception &e)
        {
            // Translate generic C++ exceptions to Python RuntimeError
            std::string error_msg = std::string("CuSAXS error: ") + e.what();
            PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
            throw py::error_already_set();
        }
    }

} // namespace

PYBIND11_MODULE(pycusaxs_cuda, m)
{
    m.doc() = "Pybind11 interface for configuring CuSAXS";

    m.def(
        "run",
        &run_with_kwargs,
        py::arg("obj_topology"),
        py::arg("topology"),
        py::arg("trajectory"),
        py::arg("grid"),
        py::arg("scaled_grid") = std::vector<int>{},
        py::arg("begin") = 0,
        py::arg("end") = 0,
        py::arg("stride") = 1,
        py::arg("output") = std::string{},
        py::arg("order") = 4,
        py::arg("scale_factor") = 1.0,
        py::arg("bin_size") = 0.0,
        py::arg("qcut") = 0.0,
        py::arg("water_model") = std::string{},
        py::arg("sodium") = 0,
        py::arg("chlorine") = 0,
        py::arg("simulation") = std::string{});
}
