/**
 * Calculates the number of water molecules and atom counts in a simulation cell.
 *
 * The AtomCounter class provides methods to calculate the number of water molecules
 * and the counts of different atom types (O, H, Na, Cl) in a simulation cell based
 * on the cell volume, the number of added sodium and chlorine atoms, and the water
 * model used.
 *
 * The calculateWaterMolecules() method calculates the number of water molecules
 * in the simulation cell based on the cell volume and the density of the water
 * model. The calculateAtomCounts() method calculates the counts of different atom
 * types in the simulation cell based on the number of water molecules and the
 * added sodium and chlorine atoms.
 */
// AtomCounter.cpp
#include "AtomCounter.h"
#include <iostream>
#include <cmath>

const float AtomCounter::AVOGADRO = 6.022e23f;
const float AtomCounter::WATER_MOLAR_MASS = 18.015f; // g/mol

const std::map<std::string, float> AtomCounter::water_models = {
    {"SPCE", 0.998f}, // g/cm³
    {"TIP3P", 0.982f} // g/cm³
};

/**
 * Constructs an AtomCounter object with the given simulation cell dimensions,
 * number of added sodium and chlorine atoms, water model, and grid dimensions.
 *
 * @param lx The length of the simulation cell in the x-direction.
 * @param ly The length of the simulation cell in the y-direction.
 * @param lz The length of the simulation cell in the z-direction.
 * @param sodium The number of added sodium atoms.
 * @param chlorine The number of added chlorine atoms.
 * @param model The water model to use (e.g. "SPCE", "TIP3P").
 * @param gx The number of grid points in the x-direction.
 * @param gy The number of grid points in the y-direction.
 * @param gz The number of grid points in the z-direction.
 */
AtomCounter::AtomCounter(float lx, float ly, float lz,
                         int sodium, int chlorine, const std::string &model,
                         int gx, int gy, int gz)
    : cell_volume(lx * ly * lz),
      added_sodium(sodium), added_chlorine(chlorine), water_model(model),
      grid_x(gx), grid_y(gy), grid_z(gz) {}

/**
 * Calculates the number of water molecules in the simulation cell based on the cell volume and the density of the specified water model.
 *
 * This method first converts the cell volume from Angstrom^3 to cm^3, then calculates the mass of water in the cell using the density of the specified water model. Finally, it computes the number of water molecules by dividing the water mass by the molar mass of water and multiplying by Avogadro's number.
 *
 * @return The number of water molecules in the simulation cell.
 */
float AtomCounter::calculateWaterMolecules() const
{
    float volume_cm3 = 1000.0f * cell_volume * 1e-24f;            // convert Å³ to cm³
    float water_mass = water_models.at(water_model) * volume_cm3; // mass of water in g
    return (water_mass / WATER_MOLAR_MASS) * AVOGADRO;
}

/**
 * Calculates the counts of different atom types (O, H, Na, Cl) in the simulation cell.
 *
 * This method first checks if the specified water model is valid, and if not, uses the
 * SPC/E model as the default. It then calculates the number of water molecules in the
 * simulation cell using the `calculateWaterMolecules()` method, and computes the counts
 * of oxygen, hydrogen, sodium, and chlorine atoms based on the number of water molecules
 * and the added sodium and chlorine atoms. The atom counts are returned as a map, where
 * the keys are the atom types and the values are the counts.
 *
 * @return A map containing the counts of different atom types in the simulation cell.
 */
std::map<std::string, float> AtomCounter::calculateAtomCounts() const
{
    std::string used_model = water_model;
    if (water_models.find(water_model) == water_models.end())
    {
        std::cout << "Invalid water model. Using SPC/E as default." << std::endl;
        used_model = "SPCE";
    }

    float water_molecules = calculateWaterMolecules();
    int total_grid_points = grid_x * grid_y * grid_z;

    std::map<std::string, float> atom_counts;
    atom_counts["O"] = water_molecules / total_grid_points;
    atom_counts["H"] = 2.0f * water_molecules / total_grid_points;
    atom_counts["Na"] = static_cast<float>(added_sodium) / total_grid_points;
    atom_counts["Cl"] = static_cast<float>(added_chlorine) / total_grid_points;

    return atom_counts;
}