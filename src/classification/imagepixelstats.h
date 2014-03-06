#ifndef IMAGEPIXELSTATS_H
#define IMAGEPIXELSTATS_H

#include <opencv2/opencv.hpp>
#include "classstats.h"

#include <map>
#include <cmath>

class ImagePixelStats
{
    typedef std::pair<cv::Point2i,ClassStats> pixel_val;

public:
    ImagePixelStats(unsigned char clCount = 0){
        clCount_=clCount;
    }

    void Clear(){
        pixels_.clear();
        palette_.clear();
    }

    void Aggregate(cv::Point2i p, ClassStats stats){

        if(palette_.empty()){
            clCount_ = stats.ClassCount();
            int color_divs = floor(pow((double)clCount_,1/3.0))+1;
            int r,g,b;
            for (int i=0; i< clCount_; i++){
                r = ((i)%(color_divs*color_divs))%color_divs+1;
                g = ((i)%(color_divs*color_divs))/color_divs+1;
                b = (i)/(color_divs*color_divs)+1;
                palette_.push_back(cv::Vec3b(r*255/(color_divs),g*255/(color_divs),b*255/(color_divs)));
            }
        }

        /*this is too much pixels; leads to the overflow
        pixels_[p].Aggregate(stats);*/

        std::pair<pToStatsMap::iterator,bool> result = pixels_.insert(std::make_pair(p,ClassStats(clCount_)));
        ((result.first)->second).Aggregate(stats);

        if (!result.second){
            std::cerr << "duplicated pixel!" << std::endl;
        }

    }

    bool Serialize(const std::string  &filename) const; //save image
    bool Serialize(std::ostream  &stream) const; //save matrix

private:

    class cvPointicompare {
        public:
            bool operator()(const cv::Point2i &x,const cv::Point2i &y) const { return (x.x > y.x) | ((x.x == y.x) & (x.y > y.y)); }
    };

    cv::Vec3b toColor(unsigned short stats) const;

    typedef std::map<cv::Point2i,ClassStats,cvPointicompare> pToStatsMap;

    std::vector<cv::Vec3b> palette_;
    pToStatsMap pixels_;\
    unsigned char clCount_;
};

#endif // IMAGEPIXELSTATS_H
