#include "split.h"

void split(const std::string &str, const std::string &symbols, std::vector<std::string> &result)
{
    size_t old=0,newpos;
    int stop = 0;

    while(((newpos=str.find_first_of(symbols,old))!=std::string::npos)){
        if (newpos-old>0 | old==0){
            result.push_back(str.substr(old,newpos-old));
        }

        old = newpos+1;
        stop++;
    }

    if (old<str.length()){
        result.push_back(str.substr(old));
    }
}

std::string replace_substr(const std::string &input, const std::string &toreplace, const std::string &replacewith)
{
    std::string tmp = input;
    size_t pos = 0;
    while((pos = tmp.find(toreplace,pos))!=std::string::npos){
        tmp.replace(pos,toreplace.size(),replacewith);
        pos+=replacewith.size();
    }

    return tmp;
}
