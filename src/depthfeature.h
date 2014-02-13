#ifndef DEPTHFEATURE_H
#define DEPTHFEATURE_H

#include "Interfaces.h"
#include "Random.h"

#include <iostream>
#include <opencv2/opencv.hpp>

class DepthFeature;

class DepthFeatureFactory{
public:
    DepthFeatureFactory(MicrosoftResearch::Cambridge::Sherwood::Random &random);

    DepthFeature getDepthFeature();

    void setUVLimit(int UVLimit) {uvlimit_ = UVLimit;}
    void setZeroPlane(unsigned short zeroPlane) {zeroplane_ = zeroPlane;}

private:
    int uvlimit_;
    unsigned short zeroplane_;
    MicrosoftResearch::Cambridge::Sherwood::Random &random_;
};

class DepthFeature // : public MicrosoftResearch::Cambridge::Sherwood::IFeatureResponse
{
   friend class DepthFeatureFactory;

protected:
    DepthFeature(cv::Point2i u, cv::Point2i v, unsigned short zeroplane):u_(u),v_(v),zeroplane_(zeroplane)
    {
    }

public:

    DepthFeature(){
        u_ = cv::Point2i(0,0);
        v_ = cv::Point2i(0,0);
        zeroplane_ = 1;
    }

    bool isDepthValid(unsigned short val) const {
        return val>0;
    }

    static bool isValid(float featureValue){
        /*this is ok, since featureValue supposed to be exactly the same*/
        return (featureValue != NaN());
    }

    float GetResponse(const MicrosoftResearch::Cambridge::Sherwood::IDataPointCollection& data, unsigned int dataIndex);

    bool Serialize(std::ostream &stream) const;

private:

    static float NaN(){
        return 1e10;
    }

    cv::Point2i u_,v_;
    unsigned short zeroplane_;
};

#endif // DEPTHFEATURE_H
