#include "subsampler.h"

bool RandomSubsampler::add(const cv::Point2i &p, value_type val)
{
    if(random_.NextDouble()<acceptRate_){
        return true;
    }else{
        return false;
    }
}

bool FixedPointNumberSubsampler::add(const cv::Point2i &p, value_type val)
{
    /*some pixel tranversal order is assumed; not good*/
    if (p == points_[currentImgIdx_][pc_]){
        pc_++;
        return true;
    }else{
        return false;
    }
}
