#ifndef PARAMETER_H
#define PARAMETER_H

#include <string>
#include <ostream>

template<typename T>
class Parameter
{

    friend std::ostream & operator<<(std::ostream &os, const Parameter<T>& p){
        os << p.desc_ << ":" << p.value_ << std::endl;
        return os;
    }

public:
    Parameter(T value, const std::string &description, T maxvalue = 0, T minvalue = 0){
        value_ = value; min_ = minvalue; max_ = maxvalue;
        desc_ = description;
    }

    T value(){
        return value_;
    }

private:
    T value_,min_,max_;
    std::string desc_;
};

#endif // PARAMETER_H
