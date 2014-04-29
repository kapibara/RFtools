#include <fstream>

#include "configuration.h"
#include "copyfile.h"
#include "localcache.h"

#include "regression/votesstatst.h"
#include "regression/depthdbreg.h"
#include "regression/regtrainingcontext.h"
#include "regression/votesaggregator.h"

int main(int argc, char **argv)
{
    std::cout << "config: " << argv[1] << std::endl;
    std::cout << "reading config data" << std::endl;

    std::ifstream in(argv[1]);
    Configuration config(in);
    in.close();

    std::cout << "configuration loaded" << std::endl;

    LocalCache cache(config.cacheFolderName(),"/home/kuznetso/tmp");

    if(!cache.init()){
        std::cerr << "failed to initialize temporary directory" << std::endl;
        exit(-1);
    }

    std::ostream &log = cache.log();

    log << "copying config file" << std::endl;
    copyfile(argv[1],cache.base() + "config.xml");

    log << "load a forest from: " << config.forestFile().c_str() << std::endl;

    std::ifstream in(config.forestFile().c_str(),std::ios_base::binary);
    forest = Forest<DepthFeature, Stats>::Deserialize(in);

    log << "forest deserialized" << std::endl;
}
