#ifndef SUBSAMPLER_H
#define SUBSAMPLER_H

#include "Random.h"
#include <opencv2/opencv.hpp>

namespace MRCS = MicrosoftResearch::Cambridge::Sherwood;

class Subsampler{
public:
    typedef unsigned short value_type;
    virtual bool add(const cv::Point2i &p,value_type val) = 0;
    virtual void newImage(unsigned int imgIdx) = 0;
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

    void newImage(unsigned int imgIdx){

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

    void newImage(unsigned int imgIdx){

    }

private:
    MRCS::Random &random_;
    double acceptRate_;
};

class FixedPointNumberSubsampler: public Subsampler
{
public:
    FixedPointNumberSubsampler(std::istream &points, MRCS::Random &random):random_(random)
    {
        pc_ = 0;
        currentImgIdx_ = -1;
        readPoints(points);
    }

    bool add(const cv::Point2i &p,value_type val);

    void newImage(unsigned int imgIdx){
        if(imgIdx != currentImgIdx_){
            pc_ = 0;
            currentImgIdx_ = imgIdx;
        }
    }

    void clear(){
        points_.clear();
    }

private:

    void readPoints(std::istream &points){
        unsigned int ppi; //points per image
        points.read((char *)&ppi, sizeof(ppi));
        unsigned int in; //image count
        points.read((char *)&in, sizeof(in));
        cv::Point2i tmp;

        for(int i=0; i< in; i++){
            points_.push_back(std::vector<cv::Point2i>(ppi));
            for(int j=0; j<ppi; j++){
                points.read((char *)(&tmp.x), sizeof(tmp.x));
                points.read((char *)(&tmp.y), sizeof(tmp.y));
                points_.back()[j] = tmp;
            }
        }
    }


    MRCS::Random &random_;
    unsigned int currentImgIdx_;
    unsigned int pc_;
    std::vector<std::vector<cv::Point2i> > points_;
};

#endif // SUBSAMPLER_H
