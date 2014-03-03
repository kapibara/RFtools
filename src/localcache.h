#ifndef LOCALCACHE_H
#define LOCALCACHE_H

#include <string>
#include <fstream>

class LocalCache
{
public:
    LocalCache(int argc, char **argv,const std::string &localtmpdir="");
    ~LocalCache();

    bool init();
    std::ostream &log();
    bool createDir(const std::string &name);
    std::string base() const{
        return base_;
    }

private:

    std::string localtmpdir_;
    std::string base_;
    std::string progName_;
    std::ofstream log_;


};

#endif // LOCALCACHE_H
