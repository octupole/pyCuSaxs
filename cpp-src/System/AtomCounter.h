// AtomCounter.h
#ifndef ATOM_COUNTER_H
#define ATOM_COUNTER_H

#include <string>
#include <map>

class AtomCounter
{
private:
    static const float AVOGADRO;
    static const float WATER_MOLAR_MASS;

    static const std::map<std::string, float> water_models;

    float cell_volume; // in Å³
    std::map<std::string, int> added_ions; // Map of ion type to count
    std::string water_model;
    int grid_x, grid_y, grid_z;

    float calculateWaterMolecules() const;

public:
    // Constructor with ion map supporting all ion types (Na, Cl, K, Ca, Mg, etc.)
    AtomCounter(float lx, float ly, float lz,
                const std::map<std::string, int> &ions, const std::string &model,
                int gx, int gy, int gz);

    std::map<std::string, float> calculateAtomCounts() const;
};

#endif // ATOM_COUNTER_H