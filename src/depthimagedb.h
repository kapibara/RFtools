#ifndef DEPTHIMAGEDB_H
#define DEPTHIMAGEDB_H

#include <opencv2/opencv.hpp>

/*DB interface required to compute depth features*/

class DepthImageDB: public MicrosoftResearch::Cambridge::Sherwood::IDataPointCollection
{
public:
    typedef unsigned int index_type;
    virtual bool getDataPoint(index_type i, cv::Mat &img, cv::Point2i &coordinate) = 0;
};

#endif // DEPTHIMAGEDB_H
