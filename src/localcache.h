#ifndef LOCALCACHE_H
#define LOCALCACHE_H

#include <string>
#include <fstream>

#include <vector>

class LocalCache
{

public:
    LocalCache(const std::string &name,const std::string &localtmpdir="");
    ~LocalCache();

    bool init();
    std::ostream &log();
    bool createDir(const std::string &name);

    std::string base() const{
        return base_;
    }

    std::ostream &openBinStream(const std::string &name)
    {
        std::ofstream *out = new std::ofstream((base_+name).c_str(),std::ios_base::binary);
        openStreams_.push_back(out);

        return *out;
    }

    std::ostream &openTextStream(const std::string &name)
    {
        std::ofstream *out = new std::ofstream((base_+name).c_str());
        openStreams_.push_back(out);

        return *out;
    }

    void closeAllStreams()
    {
        for(int i=0; i<openStreams_.size(); i++){
            openStreams_[i]->close();
            delete openStreams_[i];
        }

        openStreams_.clear();
    }

private:


    std::string localtmpdir_;
    std::string base_;
    std::string progName_;
    std::ofstream log_;

    std::vector<std::ofstream *> openStreams_;
};


#endif // LOCALCACHE_H
