#ifndef CELLTRANSFORMATION_H
#define CELLTRANSFORMATION_H

#include <vector>
#include <utility> // for std::pair
#include <stdexcept>
#include <iostream>
class Cell
{
public:
    /**
     * Calculates the transformation matrices (co and oc) from cell parameters.
     *
     * @param a Length of the a axis in Angstroms.
     * @param b Length of the b axis in Angstroms.
     * @param c Length of the c axis in Angstroms.
     * @param alpha Angle alpha in degrees.
     * @param beta Angle beta in degrees.
     * @param gamma Angle gamma in degrees.
     * @return A pair of matrices: {co, oc}
     * @throws std::invalid_argument If cell parameters are invalid.
     */
    static void calculateMatrices(
        float a, float b, float c, float alpha, float beta, float gamma);
    static void calculateMatrices(
        float a, float b, float c, float alpha, float beta, float gamma, float sigma);
    static void calculateMatrices(std::vector<float> &cell_parameters)
    {
        if (cell_parameters.size() != 6)
        {
            throw std::invalid_argument("Cell parameters must be a vector of length 6.");
        }
        else
        {
            calculateMatrices(cell_parameters[0], cell_parameters[1], cell_parameters[2],
                              cell_parameters[3], cell_parameters[4], cell_parameters[5]);
            return;
        }
    };
    static void calculateMatrices(std::vector<float> &cell_parameters, float sigma)
    {
        if (cell_parameters.size() != 6)
        {
            throw std::invalid_argument("Cell parameters must be a vector of length 6.");
        }
        else
        {
            calculateMatrices(cell_parameters[0], cell_parameters[1], cell_parameters[2],
                              cell_parameters[3], cell_parameters[4], cell_parameters[5], sigma);
            return;
        }
    };
    static void calculateMatrices(std::vector<std::vector<float>> &cell_parameters);

    // Optional static accessors for pre-calculated matrices
    static const std::vector<std::vector<float>> &getCO() { return co; };
    static const std::vector<std::vector<float>> &getOC() { return oc; };

private:
    static std::vector<std::vector<float>> co;
    static std::vector<std::vector<float>> oc;
    Cell() {} // Private constructor to prevent instantiation
};

#endif // CELLTRANSFORMATION_H
