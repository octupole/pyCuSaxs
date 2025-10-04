#include "RunSaxs.h"
#include <fstream>
#include "fmt/core.h"
#include "fmt/format.h"
#include "Options.h"
#include "saxsKernel.h"
#include "Cell.h"

/// Creates a vector of integers with a specified start, end, and step.
///
/// This function calculates the size of the vector based on the given start, end, and step values.
/// It then creates a vector and fills it with sequential integers starting from 0, and transforms
/// the values to match the desired sequence.
///
/// @param start The starting value for the sequence.
/// @param end The ending value for the sequence.
/// @param step The step size between values.
/// @return A vector of integers representing the desired sequence.
std::vector<int> RunSaxs::createVector(int start, int end, int step)
{
    // Calculate the size of the vector
    int size = (end - start) / step + 1;

    // Create a vector to hold the values
    std::vector<int> result(size);

    // Fill the vector with sequential integers starting from 0
    std::iota(result.begin(), result.end(), 0);

    // Transform the values to match the desired sequence
    std::transform(result.begin(), result.end(), result.begin(),
                   [start, step](int x)
                   { return start + x * step; });

    return result;
}
/// Runs the SAXS (Small-Angle X-ray Scattering) analysis on a range of frames.
///
/// This function creates a vector of frame indices to process, and then iterates over each frame.
/// For each frame, it retrieves the centered coordinates and box dimensions, calculates the
/// transformation matrices, and runs the SAXS kernel on the coordinates. The elapsed time for
/// the entire process is measured and printed.
///
/// @param beg The starting frame index.
/// @param end The ending frame index.
/// @param dt The step size between frames.
void RunSaxs::Run(py::object Topol, int beg, int end, int dt)
{
    auto NToA = [](std::vector<std::vector<float>> &vec)
    {
        for (auto &row : vec)
        {
            for (auto &col : row)
                col = col * 10.0f;
        };
    };
    std::cout << beg << " " << " dt" << std::endl;
    auto args = createVector(beg, end, dt);
    py::gil_scoped_acquire gil;
    std::string result;
    std::map<std::string, std::vector<std::vector<float>>> coord_map;
    std::map<std::string, std::vector<int>> index_map;
    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("append")(PY_SOURCE_DIR);

    // Set up the arguments to pass to the Python script
    // Load the Python script
    py::object analyzer = Topol;
    py::dict gather_dict = analyzer.attr("get_atom_index")();
    for (auto item : gather_dict)
    {
        std::string key = py::str(item.first);

        std::vector<int> value = item.second.cast<std::vector<int>>();
        index_map[key] = value;
    }
    auto start = std::chrono::high_resolution_clock::now();
    saxsKernel myKernel(Options::nx, Options::ny, Options::nz, Options::order);
    myKernel.setnpx(8);
    myKernel.scaledCell();
    analyzer.attr("read_frame")(0);
    auto box_dimensions = analyzer.attr("get_box")().cast<std::vector<std::vector<float>>>();
    Cell::calculateMatrices(box_dimensions);
    auto oc = Cell::getOC();
    if (Options::myPadding == padding::given)
    {
        if (index_map.find("Na") != index_map.end() && Options::Sodium == 0)
            Options::Sodium = index_map["Na"].size();
        if (index_map.find("Cl") != index_map.end() && Options::Chlorine == 0)
            Options::Chlorine = index_map["Cl"].size();

        AtomCounter Density(box_dimensions[XX][XX], box_dimensions[YY][YY],
                            box_dimensions[ZZ][ZZ], Options::Sodium, Options::Chlorine,
                            Options::Wmodel, Options::nx, Options::ny, Options::nz);
        Options::myWmodel = Density.calculateAtomCounts();
        for (auto &pair : Options::myWmodel)
        {
            auto type = pair.first;
            if (index_map.find(type) == index_map.end())
                pair.second = 0.0f;
        }
    }
    myKernel.resetHistogramParameters(oc);
    myKernel.createMemory();
    myKernel.writeBanner();

    myKernel.setcufftPlan(Options::nnx, Options::nny, Options::nnz);
    for (auto frame : args)
    {

        try
        {
            analyzer.attr("read_frame")(frame);

            float simTime = analyzer.attr("get_time")().cast<py::float_>();
            py::object coords_obj = analyzer.attr("get_coordinates")();
            auto coords = analyzer.attr("get_coordinates")().cast<std::vector<std::vector<float>>>();
            NToA(coords);
            auto box_dimensions = analyzer.attr("get_box")().cast<std::vector<std::vector<float>>>();

            Cell::calculateMatrices(box_dimensions);
            auto co = Cell::getCO();
            auto oc = Cell::getOC();
            myKernel.runPKernel(frame, simTime, coords, index_map, oc);
            std::cout << " here " << std::endl;
        }

        catch (const py::error_already_set &e)
        {
            std::cerr << "Python error: " << e.what() << std::endl;
        }
    }
    std::vector<std::vector<double>> myhisto;

    if (Options::Simulation == "nvt")
    {
        myKernel.getHistogram(oc);
    }
    myhisto = myKernel.getSaxs();
    std::ofstream myfile;
    myfile.open(Options::outFile);
    for (auto data : myhisto)
    {
        myfile << std::fixed << std::setw(10) << std::setprecision(5) << data[0];
        myfile << std::scientific << std::setprecision(5) << std::setw(12) << data[1] << std::endl;
    }
    std::cout << "Done " << args.size() << " Steps " << std::endl;
    std::cout << "Results written to  " << Options::outFile << std::endl;
    myfile.close();
    auto end0 = std::chrono::high_resolution_clock::now();

    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end0 - start);
    auto cudaTime = myKernel.getCudaTime();
    auto totalTime = duration_ms.count() / (float)args.size();
    auto readTime = totalTime - cudaTime;

    std::string banner = fmt::format(
        "\n=========================================================\n"
        "=                                                       =\n"
        "=                   cudaSAXS Timing                     =\n"
        "=                                                       =\n"
        "=           CUDA Time:     {:<10.2f} ms/per step       =\n"
        "=           Read Time:     {:<10.2f} ms/per step       =\n"
        "=           Total Time:    {:<10.2f} ms/per step       =\n"
        "=                                                       =\n"
        "=========================================================\n\n",
        cudaTime, readTime, totalTime);

    fmt::print("{}", banner);
};
RunSaxs::~RunSaxs() {};