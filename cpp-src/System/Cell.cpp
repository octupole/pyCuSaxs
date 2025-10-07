#include "Cell.h"
#include <cmath>
#include <stdexcept>
#include "Ftypedefs.h"

std::vector<std::vector<float>> Cell::co = std::vector<std::vector<float>>(DIM, std::vector<float>(DIM, 0.0f));
std::vector<std::vector<float>> Cell::oc = std::vector<std::vector<float>>(DIM, std::vector<float>(DIM, 0.0f));
/**
 * Calculates the transformation matrix (co) and its inverse (oc) based on the given cell parameters.
 *
 * @param a The length of the a-axis of the unit cell.
 * @param b The length of the b-axis of the unit cell.
 * @param c The length of the c-axis of the unit cell.
 * @param alpha The angle between the b and c axes in degrees.
 * @param beta The angle between the a and c axes in degrees.
 * @param gamma The angle between the a and b axes in degrees.
 *
 * @throws std::invalid_argument If any of the cell parameters are invalid (e.g. negative lengths or angles outside the range of 0-180 degrees).
 */
void Cell::calculateMatrices(
    float a, float b, float c, float alpha, float beta, float gamma)
{
    // Error handling for invalid cell parameters
    if (a <= 0 || b <= 0 || c <= 0)
    {
        throw std::invalid_argument("Cell parameters must be positive.");
    }
    if (alpha <= 0 || alpha >= 180 || beta <= 0 || beta >= 180 || gamma <= 0 || gamma >= 180)
    {
        throw std::invalid_argument("Angles must be between 0 and 180 degrees.");
    }

    // Convert angles to radians
    float alphaRad = alpha * M_PI / 180.0;
    float betaRad = beta * M_PI / 180.0;
    float gammaRad = gamma * M_PI / 180.0;

    // Calculate cosines and sines
    float cosAlpha = std::cos(alphaRad);
    float cosBeta = std::cos(betaRad);
    float cosGamma = std::cos(gammaRad);
    float sinGamma = std::sin(gammaRad);

    // Transformation matrix (co)
    co = {
        {a, 0.0, 0.0},
        {b * cosGamma, b * sinGamma, 0.0},
        {c * cosBeta, c * (cosAlpha - cosBeta * cosGamma) / sinGamma, c * std::sqrt(1 - cosAlpha * cosAlpha - cosBeta * cosBeta - cosGamma * cosGamma + 2 * cosAlpha * cosBeta * cosGamma) / sinGamma}};

    // Calculate inverse matrix elements (oc)
    float V = co[0][0] * co[1][1] * co[2][2]; // Volume of the unit cell
    oc = {
        {1.0f / a, 0.0f, 0.0f},
        {(-cosGamma) / (a * sinGamma), 1.0f / (b * sinGamma), 0.0f},
        {b * c * (cosAlpha * cosGamma - cosBeta) / V, a * c * (cosBeta * cosGamma - cosAlpha) / V, a * b * sinGamma / V}};
}

/**
 * Calculates the transformation matrix (co) and its inverse (oc) based on the given cell parameters, including a scaling factor (sigma).
 *
 * @param a The length of the a-axis of the unit cell, scaled by sigma.
 * @param b The length of the b-axis of the unit cell, scaled by sigma.
 * @param c The length of the c-axis of the unit cell, scaled by sigma.
 * @param alpha The angle between the b and c axes in degrees.
 * @param beta The angle between the a and c axes in degrees.
 * @param gamma The angle between the a and b axes in degrees.
 * @param sigma The scaling factor to apply to the cell dimensions.
 *
 * @throws std::invalid_argument If any of the cell parameters are invalid (e.g. negative lengths or angles outside the range of 0-180 degrees).
 */
void Cell::calculateMatrices(
    float a, float b, float c, float alpha, float beta, float gamma, float sigma)
{
    // Error handling for invalid cell parameters

    if (a <= 0 || b <= 0 || c <= 0)
    {
        throw std::invalid_argument("Cell parameters must be positive.");
    }
    if (alpha <= 0 || alpha >= 180 || beta <= 0 || beta >= 180 || gamma <= 0 || gamma >= 180)
    {
        throw std::invalid_argument("Angles must be between 0 and 180 degrees.");
    }
    a *= sigma;
    b *= sigma;
    c *= sigma;
    // Convert angles to radians
    float alphaRad = alpha * M_PI / 180.0;
    float betaRad = beta * M_PI / 180.0;
    float gammaRad = gamma * M_PI / 180.0;

    // Calculate cosines and sines
    float cosAlpha = std::cos(alphaRad);
    float cosBeta = std::cos(betaRad);
    float cosGamma = std::cos(gammaRad);
    float sinGamma = std::sin(gammaRad);

    // Transformation matrix (co)
    co = {
        {a, 0.0, 0.0},
        {b * cosGamma, b * sinGamma, 0.0},
        {c * cosBeta, c * (cosAlpha - cosBeta * cosGamma) / sinGamma, c * std::sqrt(1 - cosAlpha * cosAlpha - cosBeta * cosBeta - cosGamma * cosGamma + 2 * cosAlpha * cosBeta * cosGamma) / sinGamma}};

    // Calculate inverse matrix elements (oc)
    float V = co[0][0] * co[1][1] * co[2][2]; // Volume of the unit cell
    oc = {
        {1.0f / a, 0.0f, 0.0f},
        {(-cosGamma) / (a * sinGamma), 1.0f / (b * sinGamma), 0.0f},
        {b * c * (cosAlpha * cosGamma - cosBeta) / V, a * c * (cosBeta * cosGamma - cosAlpha) / V, a * b * sinGamma / V}};
}
void Cell::calculateMatrices(std::vector<std::vector<float>> &cell_parameters)
{
    for (auto o{0}; o < DIM; o++)
        for (auto p{0}; p < DIM; p++)
            co[o][p] = 10.0f * cell_parameters[o][p];

    // Calculate the inverse elements
    oc[0][0] = 1.0 / co[0][0];
    oc[0][1] = -co[0][1] / (co[0][0] * co[1][1]);
    oc[0][2] = (co[0][1] * co[1][2] - co[0][2] * co[1][1]) / (co[0][0] * co[1][1] * co[2][2]);

    oc[1][1] = 1.0 / co[1][1];
    oc[1][2] = -co[1][2] / (co[1][1] * co[2][2]);

    oc[2][2] = 1.0 / co[2][2];
}

/**
 * Computes the volume from the co matrix using the determinant.
 * Volume = det(co) = co[0][0] * (co[1][1] * co[2][2] - co[1][2] * co[2][1])
 *                  - co[0][1] * (co[1][0] * co[2][2] - co[1][2] * co[2][0])
 *                  + co[0][2] * (co[1][0] * co[2][1] - co[1][1] * co[2][0])
 *
 * @return Volume in Angstrom^3
 */
float Cell::getVolume()
{
    return co[0][0] * (co[1][1] * co[2][2] - co[1][2] * co[2][1])
         - co[0][1] * (co[1][0] * co[2][2] - co[1][2] * co[2][0])
         + co[0][2] * (co[1][0] * co[2][1] - co[1][1] * co[2][0]);
}
