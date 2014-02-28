#include "depthfeature.h"

#include "depthimagedb.h"

#include <cmath>

DepthFeatureFactory::DepthFeatureFactory()
{
    uvlimit_ = 10;
    zeroplane_ = 400;
}

DepthFeature DepthFeatureFactory::getDepthFeature(MicrosoftResearch::Cambridge::Sherwood::Random &random){
    cv::Point2i u = cv::Point2i(floor((random.NextDouble()*2*uvlimit_-uvlimit_)),floor((random.NextDouble()*2*uvlimit_-uvlimit_)));
    cv::Point2i v = cv::Point2i(floor((random.NextDouble()*2*uvlimit_-uvlimit_)),floor((random.NextDouble()*2*uvlimit_-uvlimit_)));

    return DepthFeature(u,v,zeroplane_);
}

float DepthFeature::GetResponse(MicrosoftResearch::Cambridge::Sherwood::IDataPointCollection &data, unsigned int dataIndex)
{
    DepthImageDB &db = dynamic_cast<DepthImageDB &>(data);
    cv::Mat I;
    cv::Point2i x,x1,x2;

    db.getDataPoint(dataIndex,I,x);

    unsigned short Ix = I.at<unsigned short>(x);

    if (isDepthValid(Ix)){

        x1 = x + u_*((double)zeroplane_/(double)Ix);
        x2 = x + v_*((double)zeroplane_/(double)Ix);

        if (x1.x < 0 | x1.x>=I.cols | x1.y < 0 |x1.y>=I.rows |
            x2.x < 0 | x2.x>=I.cols | x2.y < 0 |x2.y>=I.rows){


            return NaN();
        }else{

            if (isDepthValid(I.at<unsigned short>(x1)) & isDepthValid(I.at<unsigned short>(x2))){

                return I.at<unsigned short>(x1) - I.at<unsigned short>(x2);


            }else{

                return NaN();
            }
        }
    }else{

        return NaN();
    }
}

bool DepthFeature::Serialize(std::ostream &stream) const
{
    stream.write((const char*)(&(u_.x)),sizeof(u_.x));
    stream.write((const char*)(&(u_.y)),sizeof(u_.y));
    stream.write((const char*)(&(v_.x)),sizeof(v_.x));
    stream.write((const char*)(&(v_.y)),sizeof(v_.y));
    stream.write((const char*)(&(zeroplane_)),sizeof(zeroplane_));

    return true;
}

bool DepthFeature::SerializeChar(std::ostream &stream) const
{
    std::ostringstream ss;

    ss << "u(" << u_.x << ";" << u_.y << ")v("<< v_.x << ";"<< v_.y << ")zp" << zeroplane_ << std::endl;

    stream << ss.str() << std::endl;

    return true;
}

bool DepthFeature::Deserialize(std::istream &stream)
{
    stream.read((char*)(&(u_.x)),sizeof(u_.x));
    stream.read((char*)(&(u_.y)),sizeof(u_.y));
    stream.read((char*)(&(v_.x)),sizeof(v_.x));
    stream.read((char*)(&(v_.y)),sizeof(v_.y));
    stream.read((char*)(&(zeroplane_)),sizeof(zeroplane_));

    return true;
}
