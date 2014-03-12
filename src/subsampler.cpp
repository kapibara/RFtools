#include "subsampler.h"

bool RandomSubsampler::add(const cv::Point2i &p, value_type val)
{
    if(random_.NextDouble()<acceptRate_){
        return true;
    }else{
        return false;
    }
}
