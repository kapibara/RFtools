#ifndef SUBSAMPLER_H
#define SUBSAMPLER_H

#include "Random.h"
#include <opencv2/opencv.hpp>

namespace MRCS = MicrosoftResearch::Cambridge::Sherwood;

class Subsampler{
public:
    typedef unsigned short value_type;
    virtual bool add(const cv::Point2i &p,value_type val) = 0;
};

class StubSubsampler: public Subsampler
{
public:
    StubSubsampler()
    {

    }
    bool add(const cv::Point2i &p,value_type val){
        return true;
    }
};

class RandomSubsampler: public Subsampler
{
public:
    RandomSubsampler(double acceptRate, MRCS::Random &random):random_(random)
    {
        acceptRate_ = acceptRate;
    }

    bool add(const cv::Point2i &p,value_type val);

private:
    MRCS::Random &random_;
    double acceptRate_;
};

#endif // SUBSAMPLER_H
