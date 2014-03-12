#include "imagecache.h"

#include <opencv2/opencv.hpp>


void InMemoryCache::addToCache(const std::string &file)
{
    cv::Mat image = cv::imread(file,-1);

    mat_.push_back(image);

    filenames_.push_back(file);
}

void SimpleCache::addToCache(const std::string &file)
{
    files_.push_back(file);
}

bool SimpleCache::getImage(fileindex_type imageindex, cv::Mat &image)
{
    if (imageindex != previous_){
        cachCallCount_++;
        cached_ = cv::imread(files_[imageindex],-1);
        previous_ = imageindex;
    }

    image = cached_;

    return true;
}
