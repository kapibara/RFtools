#ifndef DEPTHFEATURE_H
#define DEPTHFEATURE_H

#include "Interfaces.h"
#include "Random.h"

#include <iostream>
#include <ostream>
#include <opencv2/opencv.hpp>

class DepthFeature;


struct DepthFeatureParameters{

    friend std::ostream & operator<<(std::ostream &os, const DepthFeatureParameters& p){
        os << "UV limit: " << p.uvlimit_ << std::endl
           << "Zero plane: " << p.zeroplane_ << std::endl;

        return os;
    }

    DepthFeatureParameters(){
        uvlimit_ = 45;
        zeroplane_ = 300;
    }

    unsigned int uvlimit_;
    unsigned short zeroplane_;
};

/** this is an interface a depth feature factory should implement
 * in any case; however, it is implemented as a template
 * for performance reasons */
class IDepthFeatureFactory{
    virtual DepthFeature getDepthFeature(MicrosoftResearch::Cambridge::Sherwood::Random &random) = 0;
    virtual void setCurrentNode(int nodeIndex) = 0;
};


//implements IDepthFeatureFactory
class FullDepthFeatureFactory{
public:
    FullDepthFeatureFactory(const DepthFeatureParameters &param = DepthFeatureParameters()):param_(param)
    {/*nothing to implement*/}

    DepthFeature getDepthFeature(MicrosoftResearch::Cambridge::Sherwood::Random &random);
    void setCurrentNode(int nodeIndex)
    {/*nothing to implement*/}

    void setParameters(const DepthFeatureParameters &params) {param_ = params;}

    DepthFeatureParameters getParameters();

private:
    DepthFeatureParameters param_;
};

//implements IDepthFeatureFactory
class PartialDepthFeatureFactory{
public:
    PartialDepthFeatureFactory(const DepthFeatureParameters &param = DepthFeatureParameters()):param_(param)
    {/*nothing to implement*/}

    DepthFeature getDepthFeature(MicrosoftResearch::Cambridge::Sherwood::Random &random);
    void setCurrentNode(int nodeIndex)
    {/*nothing to implement*/}


    void setParameters(const DepthFeatureParameters &params) {param_ = params;}

    DepthFeatureParameters getParameters();

private:
    DepthFeatureParameters param_;
};




/*============Depth Features==============*/
class DepthFeature
{

public:

    DepthFeature(cv::Point2i u, cv::Point2i v, unsigned short zeroplane):u_(u),v_(v),zeroplane_(zeroplane)
    {
    }

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
        return 500;
    }

    cv::Point2i u_,v_;
    unsigned short zeroplane_;
};

#endif // DEPTHFEATURE_H
