#ifndef IMAGEPIXELSTATS_H
#define IMAGEPIXELSTATS_H

#include <opencv2/opencv.hpp>
#include "classstats.h"

#include <map>


class cvPointicompare {
    public:
        bool operator()(const cv::Point2i &x,const cv::Point2i &y) const { return (x.x > y.x) | ((x.x == y.x) & (x.y > y.y)); }
};

class ImagePixelStats
{
    typedef std::pair<cv::Point2i,ClassStats> pixel_val;

public:
    ImagePixelStats(){
        clCount_=0;
    }

    void Clear(){
        pixels_.clear();
        palette_.clear();
    }

    void Aggregate(cv::Point2i p, ClassStats stats){
        /*std::cerr << "ImagePixelStats::Aggregate()" << std::endl;
        std::cerr << p.x << ";" << p.y <<":" << (int)stats.ClassCount() << std::endl;
        std::cerr.flush();*/

        if(palette_.empty()){
            clCount_ = stats.ClassCount();
            for (int i=0; i< clCount_; i++){
                palette_.push_back(cv::Vec3b((i+1)*256/(clCount_+1),(i+1)*256/(clCount_+1),(i+1)*256/(clCount_+1)));
            }
        }

        pixels_[p].Aggregate(stats);

        /*std::cerr << "ImagePixelStats::Aggregate(): done" << std::endl;
        std::cerr.flush();*/
    }

    bool Serialize(const std::string  &filename) const; //save image
     bool Serialize(std::ostream  &stream) const; //save matrix

private:

    cv::Vec3b toColor(unsigned short stats) const;

    std::vector<cv::Vec3b> palette_;
    std::map<cv::Point2i,ClassStats,cvPointicompare> pixels_;\
    unsigned char clCount_;
};

#endif // IMAGEPIXELSTATS_H
