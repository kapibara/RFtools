#ifndef DEPTHFEATURE_H
#define DEPTHFEATURE_H

#include "Interfaces.h"
#include "Random.h"

#include <iostream>
#include <opencv2/opencv.hpp>

class DepthFeature;


struct DepthFeatureParameters{

    friend std::ostream & operator<<(std::ostream &os, const DepthFeatureParameters& p){
        os << "UV limit: " << p.uvlimit_ << std::endl
           << "Zero plane: " << p.zeroplane_ << std::endl;
    }

    DepthFeatureParameters(){
        uvlimit_ = 45;
        zeroplane_ = 300;
    }

    unsigned int uvlimit_;
    unsigned short zeroplane_;
};

class DepthFeatureFactory{
public:
    DepthFeatureFactory(const DepthFeatureParameters &param = DepthFeatureParameters());

    DepthFeature getDepthFeature(MicrosoftResearch::Cambridge::Sherwood::Random &random);

    void setParameters(const DepthFeatureParameters &params) {param_ = params;}

private:
    DepthFeatureParameters param_;
};

/*standard depth feature*/

class DepthFeature
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
        return (featureValue < NaN());
    }

    float GetResponse(MicrosoftResearch::Cambridge::Sherwood::IDataPointCollection& data, unsigned int dataIndex);

    bool Serialize(std::ostream &stream) const;
    bool Deserialize(std::istream &stream);

    bool SerializeChar(std::ostream &stream) const;


private:

    static float NaN(){
        return 1e5;
    }

    static float bg(){
        return 200;
    }

    cv::Point2i u_,v_;
    unsigned short zeroplane_;
};

#endif // DEPTHFEATURE_H
