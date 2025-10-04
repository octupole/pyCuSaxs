#ifndef SCATTERING_H
#define SCATTERING_H
#include <string>
#include <vector>
#include <cmath>
#include <map>

#pragma once

class Scattering
{
public:
    static std::vector<float> getScattering(std::string str)
    {
        auto myvec = it1992[str];
        std::vector<float> vec;
        for (auto v0 : myvec)
        {
            for (auto v1 : v0)
            {
                vec.push_back(v1);
            }
        }
        return vec;
    }

private:
    Scattering() {};
    ~Scattering() {};
    static std::map<std::string, std::vector<std::vector<float>>> it1992;
};

#endif