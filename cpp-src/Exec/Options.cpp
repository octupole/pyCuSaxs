#include "Options.h"

std::string Options::tpr_file = "";
std::string Options::xtc_file = "";
float Options::sigma = 2.5f;
float Options::Dq = 0.05f;
float Options::Qcut = 4.0f;
int Options::order = 4;
int Options::nx = 0, Options::ny = 0, Options::nz = 0;
int Options::nnx = 0, Options::nny = 0, Options::nnz = 0;
std::string Options::Wmodel = "";
int Options::Sodium = 0, Options::Chlorine = 0;
padding Options::myPadding = padding::avg;
std::map<std::string, float> Options::myWmodel = {{"O", 0.0f}, {"H", 0.0f}, {"Na", 0.0f}, {"Cl", 0.0f}};
std::string Options::outFile = "saxs.dat";
std::string Options::Simulation = "npt";
int Options::beginFrame = 0;
int Options::endFrame = 0;
int Options::frameStride = 1;
