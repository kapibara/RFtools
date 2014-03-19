#ifndef INMEMORYCACHE_H
#define INMEMORYCACHE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class ICache
{
public:
    typedef unsigned short fileindex_type;

    virtual void addToCache(const std::string &file) = 0;
    virtual bool getImage(fileindex_type imgindex, cv::Mat &image) = 0;
    virtual fileindex_type imageCount() const = 0;
    virtual std::string imageIdx2Filename(fileindex_type i) const = 0;
    virtual unsigned int clearCacheCallCount() = 0;
};

class InMemoryCache: public ICache
{
public:

    void addToCache(const std::string &file);
    bool getImage(fileindex_type imgindex, cv::Mat &image){
        image = mat_[imgindex];
        return true;
    }

    fileindex_type imageCount() const{
        return mat_.size();
    }
    std::string imageIdx2Filename(fileindex_type i) const{
        return filenames_[i];
    }
    unsigned int clearCacheCallCount(){
        unsigned int tmp = cachCallCount_;
        cachCallCount_ = 0;
        return tmp;
    }

private:
    unsigned int cachCallCount_;
    std::vector<cv::Mat> mat_;
    std::vector<std::string> filenames_;
};

class SimpleCache: public ICache
{
public:

    void addToCache(const std::string &file);

    bool getImage(fileindex_type imgindex, cv::Mat &image);

    fileindex_type imageCount() const
    {
        return files_.size();
    }

    std::string imageIdx2Filename(fileindex_type i) const
    {
        return files_[i];
    }

    unsigned int clearCacheCallCount(){
        unsigned int tmp = cachCallCount_;
        cachCallCount_ = 0;
        return tmp;
    }

private:
    std::vector<std::string> files_;
    unsigned int cachCallCount_;
    cv::Mat cached_;
    fileindex_type previous_;
};

#endif // INMEMORYCACHE_H
