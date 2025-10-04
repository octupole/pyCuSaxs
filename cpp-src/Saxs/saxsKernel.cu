#include "saxsKernel.h"
#include "BSpmod.h"
#include "Scattering.h"
#include "opsfact.h"
#include <cuda_runtime.h> // Include CUDA runtime header
#include <cuComplex.h>

#include <cufft.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include "Splines.h"
#include "Ftypedefs.h"
#include "opsfact.h"
#include "saxsDeviceKernels.cuh"
int saxsKernel::frame_count = 0;

void saxsKernel::getHistogram(std::vector<std::vector<float>> &oc)
{
    auto nnpz = nnz / 2 + 1;
    dim3 blockDim(npx, npy, npz);
    dim3 gridDim((nnx + blockDim.x - 1) / blockDim.x,
                 (nny + blockDim.y - 1) / blockDim.y,
                 (nnpz + blockDim.z - 1) / blockDim.z);

    float mySigma = (float)Options::nx / (float)Options::nnx;

    thrust::host_vector<float> h_oc(DIM * DIM);
    for (int i = 0; i < DIM; ++i)
        for (int j = 0; j < DIM; ++j)
        {
            h_oc[i * DIM + j] = mySigma * oc[i][j];
        }
    thrust::device_vector<float> d_oc = h_oc;
    float *d_oc_ptr = thrust::raw_pointer_cast(d_oc.data());
    float frames_fact = 1.0 / (float)frame_count;
    std::cout << "frames_fact: " << bin_size << " " << kcut << " " << num_bins << std::endl;
    calculate_histogram<<<gridDim, blockDim>>>(d_Iq_ptr, d_histogram_ptr, d_nhist_ptr, d_oc_ptr, nnx, nny, nnz,
                                               bin_size, kcut, num_bins, frames_fact);
}
void saxsKernel::zeroIq()
{
    const int THREADS_PER_BLOCK = 256;
    int numBlocksGrid = (d_Iq.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    zeroDensityKernel<<<numBlocksGrid, THREADS_PER_BLOCK>>>(d_Iq_ptr, d_Iq.size());
}
// Kernel to calculate |K| values and populate the histogram

/**
 * Processes a set of particles and computes their contribution to the SAXS intensity.
 *
 * This function iterates over a set of particles, transforms their coordinates based on the orientation matrix,
 * and computes their contribution to the SAXS intensity. It then performs padding, supersampling, and Fourier
 * transform operations on the density grid to compute the final SAXS intensity.
 *
 * @param coords A vector of particle coordinates.
 * @param index_map A map of particle indices, where the keys are particle types and the values are vectors of indices.
 * @param oc The orientation matrix.
 */
void saxsKernel::runPKernel(int frame, float Time, std::vector<std::vector<float>> &coords, std::map<std::string, std::vector<int>> &index_map, std::vector<std::vector<float>> &oc)
{

    // Cudaevents

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int mx = borderBins(nx, SHELL);
    int my = borderBins(ny, SHELL);
    int mz = borderBins(nz, SHELL);
    float mySigma = (float)Options::nx / (float)Options::nnx;

    thrust::host_vector<float> h_oc(DIM * DIM);
    thrust::host_vector<float> h_oc_or(DIM * DIM);
    for (int i = 0; i < DIM; ++i)
        for (int j = 0; j < DIM; ++j)
        {
            h_oc[i * DIM + j] = mySigma * oc[i][j];
            h_oc_or[i * DIM + j] = oc[i][j];
        }
    thrust::device_vector<float> d_oc = h_oc;
    thrust::device_vector<float> d_oc_or = h_oc_or;
    float *d_oc_ptr = thrust::raw_pointer_cast(d_oc.data());
    float *d_oc_or_ptr = thrust::raw_pointer_cast(d_oc_or.data());
    auto nnpz = nnz / 2 + 1;

    dim3 blockDim(npx, npy, npz);

    dim3 gridDim((nnx + blockDim.x - 1) / blockDim.x,
                 (nny + blockDim.y - 1) / blockDim.y,
                 (nnpz + blockDim.z - 1) / blockDim.z);
    dim3 gridDimR((nnx + blockDim.x - 1) / blockDim.x,
                  (nny + blockDim.y - 1) / blockDim.y,
                  (nnz + blockDim.z - 1) / blockDim.z);
    dim3 gridDim0((nx + blockDim.x - 1) / blockDim.x,
                  (ny + blockDim.y - 1) / blockDim.y,
                  (nz + blockDim.z - 1) / blockDim.z);
    const int THREADS_PER_BLOCK = 256;
    int numBlocksGrid = (d_grid.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int numBlocksGridSuperC = (d_gridSupC.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int numBlocksGridSuperAcc = (d_gridSupAcc.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int numBlocksGridSuper = (d_gridSup.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int numBlocksGridIq = (d_Iq.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // zeroes the Sup density grid
    zeroDensityKernel<<<numBlocksGridSuperAcc, THREADS_PER_BLOCK>>>(d_gridSupAcc_ptr, d_gridSupAcc.size());

    float totParticles{0};
    std::string formatted_string = fmt::format("--> Frame: {:<7}  Time Step: {:.2f} fs", frame, Time);

    // Print the formatted string
    std::cout << formatted_string << std::endl;
    float N1 = (float)nnx / (float)nx;
    float N2 = (float)nny / (float)ny;
    float N3 = (float)nnz / (float)nz;
    auto plan = this->getPlan();
    for (const auto &pair : index_map)
    {

        std::string type = pair.first;
        std::vector<int> value = pair.second;
        // Create a host vector to hold the particles
        thrust::host_vector<float> h_particles;
        h_particles.reserve(value.size() * 3);
        // Fill the host vector with the particles according to the indices
        std::for_each(value.begin(), value.end(), [&h_particles, &coords](int i)
                      { h_particles.insert(h_particles.end(), coords[i].begin(), coords[i].end()); });
        // define the number of particles
        this->numParticles = value.size();

        // Allocate and copy particles to the device
        thrust::device_vector<float> d_particles = h_particles;
        // Copy the host vector to the device
        thrust::host_vector<float> h_scatter = Scattering::getScattering(type);
        thrust::device_vector<float> d_scatter = h_scatter;

        float *d_particles_ptr = thrust::raw_pointer_cast(d_particles.data());
        float *d_scatter_ptr = thrust::raw_pointer_cast(d_scatter.data());

        int numBlocks = (numParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        //    Kernels launch for the rhoKernel
        zeroDensityKernel<<<numBlocksGrid, THREADS_PER_BLOCK>>>(d_grid_ptr, d_grid.size());
        cudaDeviceSynchronize();
        // Check for errors
        rhoCartKernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_particles_ptr, d_oc_or_ptr, d_grid_ptr, order,
                                                        numParticles, nx, ny, nz);

        // Synchronize the device
        cudaDeviceSynchronize();
        // picking the padding
        float myDens = 0.0f;
        if (Options::myPadding == padding::avg)
        {
            thrust::host_vector<float> h_Dens = {0.0f};
            thrust::host_vector<int> h_count = {0};
            thrust::device_vector<float> d_Dens = h_Dens;
            thrust::device_vector<int> d_count = h_count;
            paddingKernel<<<gridDim0, blockDim>>>(d_grid_ptr, nx, ny, nz, mx, my, mz,
                                                  thrust::raw_pointer_cast(d_Dens.data()),
                                                  thrust::raw_pointer_cast(d_count.data()));
            // Synchronize the device
            cudaDeviceSynchronize();
            h_Dens = d_Dens;
            h_count = d_count;
            myDens = h_Dens[0] / (float)h_count[0];
        }
        else
        {
            if (Options::myWmodel.find(type) != Options::myWmodel.end())
            {
                myDens = Options::myWmodel[type];
            }
        }

        // zeroes the Sup density grid
        zeroDensityKernel<<<numBlocksGridSuperC, THREADS_PER_BLOCK>>>(d_gridSupC_ptr, d_gridSupC.size());
        cudaDeviceSynchronize();

        superDensityKernel<<<gridDimR, blockDim>>>(d_grid_ptr, d_gridSup_ptr, myDens, nx, ny, nz, nnx, nny, nnz);
        // Synchronize the device
        cudaDeviceSynchronize();

        cufftExecR2C(plan, d_gridSup_ptr, d_gridSupC_ptr);
        // Synchronize the device
        cudaDeviceSynchronize();

        thrust::host_vector<float> h_nato = {0.0f};
        thrust::device_vector<float> d_nato = h_nato;
        scatterKernel<<<gridDim, blockDim>>>(d_gridSupC_ptr, d_gridSupAcc_ptr, d_oc_ptr, d_scatter_ptr, nnx, nny, nnz, kcut, thrust::raw_pointer_cast(d_nato.data()));
        cudaDeviceSynchronize();
        h_nato = d_nato;
        totParticles += h_nato[0];
    }
    modulusKernel<<<gridDim, blockDim>>>(d_gridSupAcc_ptr, d_moduleX_ptr, d_moduleY_ptr, d_moduleZ_ptr, nnx, nny, nnz);
    // // Synchronize the device
    cudaDeviceSynchronize();
    if (Options::Simulation == "nvt")
    {
        gridAddKernel<<<numBlocksGridIq, THREADS_PER_BLOCK>>>(d_gridSupAcc_ptr, d_Iq_ptr, d_Iq.size());
        cudaDeviceSynchronize();
        frame_count++;
    }
    else if (Options::Simulation == "npt")
    {

        calculate_histogram<<<gridDim, blockDim>>>(d_gridSupAcc_ptr, d_histogram_ptr, d_nhist_ptr, d_oc_ptr, nnx, nny, nnz,
                                                   bin_size, kcut, num_bins);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time in milliseconds
    float gpuElapsedTime;
    cudaEventElapsedTime(&gpuElapsedTime, start, stop);
    cudaTime += gpuElapsedTime;
    cudaCalls += 1.0;

    // Destroy the events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
std::vector<std::vector<double>> saxsKernel::getSaxs()
{
    std::vector<std::vector<double>> saxs;
    h_histogram = d_histogram;
    h_nhist = d_nhist;

    for (auto o{1}; o < h_histogram.size(); o++)
    {
        if (h_nhist[o] != 0.0f)
        {
            vector<double> val = {o * this->bin_size, h_histogram[o] / h_nhist[o]};
            saxs.push_back(val);
        }
    }
    return saxs;
}

/**
 * @brief Creates the necessary memory for the SAXS computation.
 *
 * This function sets up the memory buffers and allocates memory for the SAXS computation.
 * It calculates the optimal grid sizes (nnx, nny, nnz) based on the original grid sizes (nx, ny, nz)
 * and the given sigma value. It then creates the necessary host and device memory buffers for the
 * grid, super-grid, and module data.
 *
 * @param[in,out] nnx The optimal x-dimension of the super-grid.
 * @param[in,out] nny The optimal y-dimension of the super-grid.
 * @param[in,out] nnz The optimal z-dimension of the super-grid.
 * @param[in] sigma The sigma value used to calculate the optimal grid sizes.
 */
void saxsKernel::createMemory()
{
    size_t nnpz = nnz / 2 + 1;

    this->bin_size = Options::Dq;
    this->kcut = Options::Qcut;

    this->num_bins = static_cast<int>(kcut / bin_size) + 1;
    h_histogram = thrust::host_vector<float>(num_bins, 0.0f);
    h_nhist = thrust::host_vector<long int>(num_bins, 0);

    d_histogram = h_histogram;
    d_nhist = h_nhist;
    d_histogram_ptr = thrust::raw_pointer_cast(d_histogram.data());
    d_nhist_ptr = thrust::raw_pointer_cast(d_nhist.data());
    BSpline::BSpmod *bsp_modx = new BSpline::BSpmod(nnx, nny, nnz);

    thrust::host_vector<float> h_moduleX = bsp_modx->ModX();
    thrust::host_vector<float> h_moduleY = bsp_modx->ModY();
    thrust::host_vector<float> h_moduleZ = bsp_modx->ModZ();

    d_moduleX = h_moduleX;
    d_moduleY = h_moduleY;
    d_moduleZ = h_moduleZ;
    d_moduleX_ptr = thrust::raw_pointer_cast(d_moduleX.data());
    d_moduleY_ptr = thrust::raw_pointer_cast(d_moduleY.data());
    d_moduleZ_ptr = thrust::raw_pointer_cast(d_moduleZ.data());

    d_grid.resize(nx * ny * nz);
    d_gridSup.resize(nnx * nny * nnz);
    d_gridSupC.resize(nnx * nny * nnpz);
    d_gridSupAcc.resize(nnx * nny * nnpz);
    d_Iq.resize(nnx * nny * nnpz);

    d_grid_ptr = thrust::raw_pointer_cast(d_grid.data());
    d_gridSup_ptr = thrust::raw_pointer_cast(d_gridSup.data());
    d_gridSupC_ptr = thrust::raw_pointer_cast(d_gridSupC.data());
    d_gridSupAcc_ptr = thrust::raw_pointer_cast(d_gridSupAcc.data());
    d_Iq_ptr = thrust::raw_pointer_cast(d_Iq.data());
    // Do bspmod
}
/**
 * Generates a vector of multiples of 2, 3, 5, and 7 up to a given limit.
 *
 * This function generates all possible multiples of 2, 3, 5, and 7 up to the
 * specified limit, and returns them as a sorted, unique vector.
 *
 * @param limit The maximum value to generate multiples up to.
 * @return A vector of all multiples of 2, 3, 5, and 7 up to the given limit.
 */
// Function to generate multiples of 2, 3, 5, and 7 up to a given limit
std::vector<long long> saxsKernel::generateMultiples(long long limit)
{
    std::vector<long long> multiples;
    for (int a = 0; std::pow(2, a) <= limit; ++a)
    {
        for (int b = 0; std::pow(2, a) * std::pow(3, b) <= limit; ++b)
        {
            for (int c = 0; std::pow(2, a) * std::pow(3, b) * std::pow(5, c) <= limit; ++c)
            {
                for (int d = 0; std::pow(2, a) * std::pow(3, b) * std::pow(5, c) * std::pow(7, d) <= limit; ++d)
                {
                    long long multiple = std::pow(2, a) * std::pow(3, b) * std::pow(5, c) * std::pow(7, d);
                    if (multiple <= limit)
                    {
                        multiples.push_back(multiple);
                    }
                }
            }
        }
    }
    std::sort(multiples.begin(), multiples.end());
    multiples.erase(std::unique(multiples.begin(), multiples.end()), multiples.end());
    return multiples;
}

/**
 * Finds the closest integer to N * sigma that is obtainable by multiplying only 2, 3, 5, and 7.
 *
 * This function takes a target value N and a standard deviation sigma, and finds the closest integer
 * to N * sigma that can be expressed as a product of only the prime factors 2, 3, 5, and 7.
 *
 * @param n The target value N.
 * @param sigma The standard deviation.
 * @return The closest integer to N * sigma that is obtainable by multiplying only 2, 3, 5, and 7.
 */
// Function to find the closest integer to N * sigma that is obtainable by multiplying only 2, 3, 5, and 7
long long saxsKernel::findClosestProduct(int n, float sigma)
{
    long long target = std::round(n * sigma);
    long long limit = target * 2; // A generous limit for generating multiples
    std::vector<long long> multiples = generateMultiples(limit);

    long long closest = target;
    long long minDifference = std::numeric_limits<long long>::max();

    for (long long multiple : multiples)
    {
        long long difference = std::abs(multiple - target);
        if (difference < minDifference)
        {
            minDifference = difference;
            closest = multiple;
        }
    }

    return closest;
}
void saxsKernel::scaledCell()
{
    sigma = Options::sigma;
    if (Options::nnx == 0)
    {
        nnx = this->nnx = static_cast<int>(findClosestProduct(nx, sigma));
        nny = this->nny = static_cast<int>(findClosestProduct(ny, sigma));
        nnz = this->nnz = static_cast<int>(findClosestProduct(nz, sigma));
        Options::nnx = nnx;
        Options::nny = nny;
        Options::nnz = nnz;
    }
    else
    {
        this->nnx = Options::nnx;
        this->nny = Options::nny;
        this->nnz = Options::nnz;
    }
}
void saxsKernel::resetHistogramParameters(std::vector<std::vector<float>> &oc)
{
    using namespace std;
    auto qcut = Options::Qcut;
    auto dq = Options::Dq;
    int nfx{(nnx % 2 == 0) ? nnx / 2 : nnx / 2 + 1};
    int nfy{(nny % 2 == 0) ? nny / 2 : nny / 2 + 1};
    int nfz{(nnz % 2 == 0) ? nnz / 2 : nnz / 2 + 1};
    float argx{2.0f * (float)M_PI * oc[XX][XX] / sigma};
    float argy{2.0f * (float)M_PI * oc[YY][YY] / sigma};
    float argz{2.0f * (float)M_PI * oc[ZZ][ZZ] / sigma};

    std::vector<float> fx{(float)nfx - 1, (float)nfy - 1, (float)nfz - 1};

    vector<float> mydq0 = {argx, argy, argz, dq};
    vector<float> mycut0 = {argx * fx[XX], argy * fx[YY], argz * fx[ZZ], qcut};

    dq = (*std::max_element(mydq0.begin(), mydq0.end()));
    qcut = *std::min_element(mycut0.begin(), mycut0.end());
    if (qcut != Options::Qcut)
    {
        std::string formatted_string = fmt::format("----- Qcut had to be reset to {:.2f} from  {:.2f} ----", qcut, Options::Qcut);
        std::cout << "\n--------------------------------------------------\n";
        std::cout << formatted_string << "\n";
        std::cout << "--------------------------------------------------\n\n";

        Options::Qcut = qcut;
    }
    if (dq != Options::Dq)
    {
        std::string formatted_string = fmt::format("----- Dq had to be reset to {:.3f} from  {:.3f} ----", dq, Options::Dq);
        std::cout << "\n--------------------------------------------------\n";
        std::cout << formatted_string << "\n";
        std::cout << "--------------------------------------------------\n\n";

        Options::Dq = dq;
    }
}
void saxsKernel::writeBanner()
{
    std::string padding = Options::myPadding == padding::given ? Options::Wmodel : "avg Border";
    std::string banner{""};
    if (Options::myPadding == padding::avg)
    {
        banner = fmt::format(
            "*************************************************\n"
            "* {:^40}      *\n"
            "* {:<19} {:>4} * {:>4} * {:>4}        *\n"
            "* {:<19} {:>4} * {:>4} * {:>4}        *\n"
            "* {:<10} {:>4}      {:<10} {:>4}          *\n"
            "* {:<10} {:>4.3f}     {:<10}  {:>3.1f}          *\n"
            "* {:<10}           {:<14}           *\n"
            "*************************************************\n\n",
            "Running cudaSAXS", "Cell Grid", Options::nx, Options::ny, Options::nz,
            "Supercell Grid", Options::nnx, Options::nny, Options::nnz, "Order",
            Options::order, "Sigma", Options::sigma, "Bin Size", Options::Dq, "Q Cutoff ", Options::Qcut, "Padding ", padding);
    }
    else
    {
        banner = fmt::format(
            "*************************************************\n"
            "* {:^40}      *\n"
            "* {:<19} {:>4} * {:>4} * {:>4}        *\n"
            "* {:<19} {:>4} * {:>4} * {:>4}        *\n"
            "* {:<10} {:>4}      {:<10} {:>4}          *\n"
            "* {:<10} {:>4.3f}     {:<10}  {:>3.1f}          *\n"
            "* {:<10}           {:<14}           *\n"
            "* {:<10} {:>4d}      {:<10} {:>4d}          *\n"
            "*************************************************\n\n",
            "Running cudaSAXS", "Cell Grid", Options::nx, Options::ny, Options::nz,
            "Supercell Grid", Options::nnx, Options::nny, Options::nnz, "Order",
            Options::order, "Sigma", Options::sigma, "Bin Size", Options::Dq, "Q Cutoff ",
            Options::Qcut, "Padding ", padding,
            "Na ions", Options::Sodium, "Cl Ions", Options::Chlorine);
    }
    std::cout << banner;
}

saxsKernel::~saxsKernel()
{
}
