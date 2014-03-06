#ifndef STRING2NUMBER_H
#define STRING2NUMBER_H

#include <string>
#include <sstream>

template<typename T>
T str2num(const std::string &s){
    std::stringstream stream(s);
    T result;

    stream >> result;

    return result;
}

template<typename T>
std::string num2str(T number, int fieldwidth = 0){
    std::stringstream stream;

    if(fieldwidth > 0){
        stream.width(fieldwidth);
        stream.fill('0');
    }
    stream << number;

    return stream.str();
}

#endif
