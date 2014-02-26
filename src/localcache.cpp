#include "localcache.h"

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

#include <time.h>
#include <iostream>

#include "string2number.hpp"
#include "split.h"

namespace bfs = boost::filesystem;

LocalCache::LocalCache(int argc, char **argv, const std::string &localtmpdir)
{
    localtmpdir_ = replace_substr(localtmpdir,"\\" , "/");

    std::vector<std::string> buffer;
    split(std::string(argv[0]),"\\/",buffer);

    if(buffer.size()>0){
        progName_ = buffer[buffer.size()-1];
    }else{
        throw std::invalid_argument("could not extract program name from argument");
    }

    if(localtmpdir_.empty()){

        localtmpdir_ = std::string(getenv("HOME")); //can I get wrong '/' here?
        localtmpdir_ = replace_substr(localtmpdir_,"\\" , "/");

        bfs::path p(localtmpdir_);
        if (!bfs::is_directory(p)){
            throw std::invalid_argument("set temporary cache directory; HOME variable is not set or invalid");
        }
    }
}

bool LocalCache::init(){
     time_t now;
     struct tm * timeinfo;

     time(&now);
     timeinfo = localtime (&now);

     base_ = localtmpdir_ + "/" + progName_ + "/" +
             num2str<int>(timeinfo->tm_mday,2)+"_"+num2str<int>(timeinfo->tm_hour,2)+"_"+num2str<int>(timeinfo->tm_min,2)+"_"+num2str<int>(timeinfo->tm_sec,2);

     std::cerr << "base: " << base_ << std::endl;

     bool result = bfs::create_directories(bfs::path(base_ ));

     base_ = base_+"/"; //boost bag :) -> false returning value in presence of "/" in the end of the string

     return result;
}

bool LocalCache::createDir(const std::string &name){
    std::string newdir = base_+ name;

    return bfs::create_directory(bfs::path(newdir));
}
