#include "CudaSaxsInterface.h"
#include "Options.h"
#include "Ftypedefs.h"
#include "RunSaxs.h"
#include <pybind11/pybind11.h>
#include <array>
#include <sstream>
#include <stdexcept>

namespace
{

    std::array<int, 3> expand_grid(const std::vector<int> &values, const char *label)
    {
        if (values.empty())
        {
            throw std::invalid_argument(std::string(label) + " requires 1 or 3 integers.");
        }

        if (values.size() == 1)
        {
            return {values[0], values[0], values[0]};
        }

        if (values.size() == 3)
        {
            return {values[0], values[1], values[2]};
        }

        throw std::invalid_argument(std::string(label) + " must contain either 1 or 3 integers.");
    }

    std::array<int, 3> expand_optional_grid(const std::vector<int> &values, const char *label)
    {
        if (values.empty())
        {
            return {0, 0, 0};
        }

        return expand_grid(values, label);
    }

} // namespace

CudaSaxsResult run_cuda_saxs(py::object Topol, const CudaSaxsConfig &config)
{
    if (config.topology_path.empty())
    {
        throw std::invalid_argument("Topology path must not be empty.");
    }
    if (config.trajectory_path.empty())
    {
        throw std::invalid_argument("Trajectory path must not be empty.");
    }
    if (config.end_frame < config.begin_frame)
    {
        throw std::invalid_argument("Last frame must be greater than or equal to initial frame.");
    }
    if (config.frame_stride <= 0)
    {
        throw std::invalid_argument("Frame stride must be a positive integer.");
    }

    auto primary_grid = expand_grid(config.grid_shape, "Grid size");
    auto scaled_grid = expand_optional_grid(config.scaled_grid_shape, "Scaled grid size");

    Options::tpr_file = config.topology_path;
    Options::xtc_file = config.trajectory_path;
    Options::order = config.spline_order;
    Options::beginFrame = config.begin_frame;
    Options::endFrame = config.end_frame;
    Options::frameStride = config.frame_stride;
    Options::sigma = config.scale_factor;
    Options::Dq = config.bin_size;
    Options::Qcut = config.qcut;
    Options::Wmodel = config.water_model;
    Options::Sodium = config.sodium_atoms;
    Options::Chlorine = config.chlorine_atoms;
    Options::outFile = config.output_path.empty() ? Options::outFile : config.output_path;
    if (!config.simulation_ensemble.empty())
    {
        Options::Simulation = config.simulation_ensemble;
    }

    Options::nx = primary_grid[XX];
    Options::ny = primary_grid[YY];
    Options::nz = primary_grid[ZZ];

    Options::nnx = scaled_grid[XX];
    Options::nny = scaled_grid[YY];
    Options::nnz = scaled_grid[ZZ];

    Options::myPadding = Options::Wmodel.empty() ? padding::avg : padding::given;

    std::ostringstream summary;
    summary << "cudaSAXS configured\n"
            << "  Topology: " << Options::tpr_file << '\n'
            << "  Trajectory: " << Options::xtc_file << '\n'
            << "  Frames: " << config.begin_frame << " -> " << config.end_frame
            << " (stride " << config.frame_stride << ")\n"
            << "  Grid: (" << Options::nx << ", " << Options::ny << ", " << Options::nz << ")\n";

    if (!config.scaled_grid_shape.empty())
    {
        summary << "  Scaled grid: (" << Options::nnx << ", " << Options::nny << ", " << Options::nnz << ")\n";
    }

    summary << "  Order: " << Options::order << '\n'
            << "  Scale factor: " << Options::sigma << '\n'
            << "  Bin size: " << Options::Dq << '\n'
            << "  Q-cut: " << Options::Qcut << '\n';

    if (!Options::Wmodel.empty())
    {
        summary << "  Water model: " << Options::Wmodel << '\n';
    }
    summary << "  Na atoms: " << Options::Sodium << '\n'
            << "  Cl atoms: " << Options::Chlorine;

    CudaSaxsResult result;
    result.nx = Options::nx;
    result.ny = Options::ny;
    result.nz = Options::nz;
    result.nnx = Options::nnx;
    result.nny = Options::nny;
    result.nnz = Options::nnz;
    result.summary = summary.str();

    RunSaxs saxs(Options::tpr_file, Options::xtc_file);
    saxs.Run(Topol, Options::beginFrame, Options::endFrame, Options::frameStride);

    return result;
}
