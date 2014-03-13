#include "localcache.h"

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

#include <time.h>
#include <iostream>

#include "string2number.hpp"
#include "split.h"

namespace bfs = boost::filesystem;


LocalCache::LocalCache(const std::string &name, const std::string &localtmpdir)
{

    localtmpdir_ = replace_substr(localtmpdir,"\\" , "/");

    std::vector<std::string> buffer;
    split(std::string(name),"\\/",buffer);

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

LocalCache::~LocalCache()
{
    log_.close();
    for(int i=0; i<openStreams_.size(); i++){
        if((openStreams_[i])->is_open()){
            (openStreams_[i])->close();
        }
        delete openStreams_[i];
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

     log_.open((base_+"log.txt").c_str());

     return result;
}

std::ostream &LocalCache::log()
{
    return log_;
}

bool LocalCache::createDir(const std::string &name){
    std::string newdir = base_+ name;

    return bfs::create_directory(bfs::path(newdir));
}
