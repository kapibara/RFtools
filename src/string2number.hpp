#ifndef STRING2NUMBER_H
#define STRING2NUMBER_H

#include <string>

template<typename T>
T str2num(const std::string &s){
    std::stringstream stream(s);
    T result;

    stream >> result;

    return result;
}

#endif
