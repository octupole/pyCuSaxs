#ifndef OPTIONS_H
#define OPTIONS_H
#include <string>
#include <vector>
#include <map>
#pragma once
enum class padding
{
    avg,
    given
};

class Options
{
public:
    static std::string tpr_file;
    static std::string xtc_file;
    static float sigma;
    static float Dq;
    static float Qcut;
    static int order;
    static int beginFrame;
    static int endFrame;
    static int frameStride;
    static int nx, ny, nz;
    static int nnx, nny, nnz;
    static std::string Wmodel;
    static std::map<std::string, int> IonCounts;  // Map of ion type to count (e.g., {"Na": 150, "Cl": 150, "K": 10})
    static padding myPadding;
    static std::map<std::string, float> myWmodel;
    static std::string outFile;
    static std::string Simulation;

private:
    Options() {};
    ~Options() {};
};

#endif
