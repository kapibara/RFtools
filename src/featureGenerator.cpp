
#include <iostream>

#include "depthfeature.h"
#include "localcache.h"

int main(int argc, char** argv)
{
    LocalCache cache("DepthHOUGH","/home/kuznetso/tmp");

    std::cout << "arg1: " << argv[1] << std::endl;

    if(!cache.init()){
        std::cerr << "failed to initialize temporary directory" << std::endl;
        exit(-1);
    }

    std::ostream &log = cache.log();

    try{

    }catch(std::exception e){
        std::cerr << "exception caught: " << e.what() << std::endl;
        std::cerr.flush();
    }
}
