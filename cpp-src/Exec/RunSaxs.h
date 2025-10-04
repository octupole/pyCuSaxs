#ifndef RUNSAXS_H
#define RUNSAXS_H
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <numeric>
#include <chrono>
#include "AtomCounter.h"
#include <iomanip>
#pragma once
namespace py = pybind11;

class RunSaxs
{
public:
    RunSaxs(std::string tpr, std::string xtd) : tpr_file(tpr), xtc_file(xtd) {};

    void Run(py::object, int, int, int);
    ~RunSaxs();

private:
    std::string tpr_file, xtc_file;
    std::vector<int> createVector(int, int, int);
};

#endif