#ifndef SPLIT_H

#define SPLIT_H

#include <string>
#include <vector>
#include <iostream>

std::vector<std::string> split(const std::string &str, const std::string &symbols)
{
    std::vector<std::string> result;
    size_t old=0,newpos;
    int stop = 0;

    while(((newpos=str.find_first_of(symbols,old))!=std::string::npos) & (stop < 5)){
        if (newpos-old>0){
            result.push_back(str.substr(old,newpos-old));
        }

        old = newpos+1;
        stop++;
    }

    if (old<str.length()){
        result.push_back(str.substr(old));
    }

    return result;
}

#endif
