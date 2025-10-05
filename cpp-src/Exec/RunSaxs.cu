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

bool RunSaxs::loadFrameData(py::handle frame_handle, FrameData &data)
{
    try
    {
        py::dict frame_data = py::cast<py::dict>(frame_handle);

        data.frame_num = frame_data["frame"].cast<int>();
        data.time = frame_data["time"].cast<float>();

        // Extract positions array
        py::array_t<float> positions = frame_data["positions"].cast<py::array_t<float>>();
        auto pos = positions.unchecked<2>();

        const size_t n_atoms = pos.shape(0);

        // Resize and fill coords in std::vector<std::vector<float>> format
        data.coords.resize(n_atoms, std::vector<float>(3));
        for (size_t i = 0; i < n_atoms; ++i)
        {
            data.coords[i][0] = pos(i, 0);
            data.coords[i][1] = pos(i, 1);
            data.coords[i][2] = pos(i, 2);
        }

        // Extract box dimensions
        py::array_t<float> box_array = frame_data["box"].cast<py::array_t<float>>();
        auto box_data = box_array.unchecked<2>();

        data.box.resize(3, std::vector<float>(3));
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                data.box[i][j] = box_data(i, j);
            }
        }

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error loading frame data: " << e.what() << std::endl;
        return false;
    }
}

void RunSaxs::Run(py::object Topol, int beg, int end, int dt)
{
    const int start_frame = beg;
    const int stop_frame = end;
    const int stride = std::max(dt, 1);

    py::object analyzer = std::move(Topol);

    try
    {
        // ===== Setup phase (with GIL) =====
        py::gil_scoped_acquire gil;

        // Get atom index map from Python
        std::map<std::string, std::vector<int>> index_map;
        py::dict gather_dict = analyzer.attr("get_atom_index")();
        for (auto item : gather_dict)
        {
            std::string key = py::str(item.first);
            std::vector<int> value = item.second.cast<std::vector<int>>();
            index_map[key] = value;
        }

        // Initialize SAXS kernel
        auto start = std::chrono::high_resolution_clock::now();
        saxsKernel myKernel(Options::nx, Options::ny, Options::nz, Options::order);
        myKernel.setnpx(8);
        myKernel.scaledCell();

        // Read first frame to get box dimensions for initialization
        analyzer.attr("read_frame")(0);
        auto box_dimensions = analyzer.attr("get_box")().cast<std::vector<std::vector<float>>>();
        Cell::calculateMatrices(box_dimensions);
        auto oc = Cell::getOC();

        // Setup padding if needed
        if (Options::myPadding == padding::given)
        {
            if (index_map.find("Na") != index_map.end() && Options::Sodium == 0)
                Options::Sodium = index_map["Na"].size();
            if (index_map.find("Cl") != index_map.end() && Options::Chlorine == 0)
                Options::Chlorine = index_map["Cl"].size();

            AtomCounter Density(box_dimensions[0][0], box_dimensions[1][1],
                                box_dimensions[2][2], Options::Sodium, Options::Chlorine,
                                Options::Wmodel, Options::nx, Options::ny, Options::nz);
            Options::myWmodel = Density.calculateAtomCounts();
            for (auto &pair : Options::myWmodel)
            {
                auto type = pair.first;
                if (index_map.find(type) == index_map.end())
                    pair.second = 0.0f;
            }
        }

        // Finalize kernel setup
        myKernel.resetHistogramParameters(oc);
        myKernel.createMemory();
        myKernel.writeBanner();
        myKernel.setcufftPlan(Options::nnx, Options::nny, Options::nnz);

        // ===== Streaming phase =====
        // Create iterator with GIL held
        auto frames_iter = analyzer.attr("iter_frames_stream")(
            start_frame, stop_frame + 1, stride);

        // Double buffering for pipeline optimization
        FrameData current_frame;
        FrameData next_frame;
        bool has_next = false;

        // Prime the pipeline - load first frame
        auto iter = frames_iter.begin();
        if (iter != frames_iter.end())
        {
            has_next = loadFrameData(*iter, next_frame);
            ++iter;
        }

        while (has_next)
        {
            // Swap buffers (cheap pointer swap)
            std::swap(current_frame, next_frame);

            // Start loading next frame while processing current
            bool has_more = (iter != frames_iter.end());
            if (has_more)
            {
                has_next = loadFrameData(*iter, next_frame);
                ++iter;
            }
            else
            {
                has_next = false;
            }

            // Release GIL for GPU processing
            {
                py::gil_scoped_release release;

                // ===== Process current frame (GIL released) =====
                try
                {
                    // Calculate transformation matrices from box dimensions
                    Cell::calculateMatrices(current_frame.box);
                    auto co = Cell::getCO();
                    auto oc = Cell::getOC();

                    // Run SAXS kernel computation
                    myKernel.runPKernel(current_frame.frame_num, current_frame.time,
                                        current_frame.coords, index_map, oc);
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Error processing frame " << current_frame.frame_num
                              << ": " << e.what() << std::endl;
                }
                // ================================================
            }
            // GIL automatically re-acquired here when 'release' goes out of scope
        }

        // ===== Finalization phase (with GIL) =====
        std::vector<std::vector<double>> myhisto;

        if (Options::Simulation == "nvt")
        {
            myKernel.getHistogram(oc);
        }
        myhisto = myKernel.getSaxs();

        // Write results
        std::ofstream myfile;
        myfile.open(Options::outFile);
        for (auto data : myhisto)
        {
            myfile << std::fixed << std::setw(10) << std::setprecision(5) << data[0];
            myfile << std::scientific << std::setprecision(5) << std::setw(12) << data[1] << std::endl;
        }
        myfile.close();

        // Print timing information
        auto frames_to_process = createVector(start_frame, stop_frame, stride);
        std::cout << "Done " << frames_to_process.size() << " Steps" << std::endl;
        std::cout << "Results written to " << Options::outFile << std::endl;

        auto end0 = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end0 - start);
        auto cudaTime = myKernel.getCudaTime();
        auto totalTime = duration_ms.count() / (float)frames_to_process.size();
        auto readTime = totalTime - cudaTime;

        std::string banner = fmt::format(
            "\n=========================================================\n"
            "=                                                       =\n"
            "=                    CuSAXS Timing                     =\n"
            "=                                                       =\n"
            "=           CUDA Time:     {:<10.2f} ms/per step       =\n"
            "=           Read Time:     {:<10.2f} ms/per step       =\n"
            "=           Total Time:    {:<10.2f} ms/per step       =\n"
            "=                                                       =\n"
            "=========================================================\n\n",
            cudaTime, readTime, totalTime);

        fmt::print("{}", banner);
    }
    catch (const py::error_already_set &err)
    {
        std::cerr << "Python error while iterating frames: " << err.what() << std::endl;
    }
};

RunSaxs::~RunSaxs() {};
