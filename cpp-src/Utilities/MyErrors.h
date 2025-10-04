#ifndef MYERRORS_H
#define MYERRORS_H
#include <iostream>
#include <stdexcept>
#pragma once

class MyErrors : public std::runtime_error
{
public:
    explicit MyErrors(const std::string &message) : std::runtime_error(message) {}
};

#endif