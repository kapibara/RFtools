#ifndef LOCALCACHE_H
#define LOCALCACHE_H

#include <string>

class LocalCache
{
public:
    LocalCache(int argc, char **argv,const std::string &localtmpdir="");

    bool init();
    bool createDir(const std::string &name);
    std::string base() const{
        return base_;
    }

private:

    std::string localtmpdir_;
    std::string base_;
    std::string progName_;


};

#endif // LOCALCACHE_H
