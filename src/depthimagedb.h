#ifndef DEPTHIMAGEDB_H
#define DEPTHIMAGEDB_H

#include <opencv2/opencv.hpp>
#include "Interfaces.h"
#include "arraylist.h"

/*DB interface required to compute depth features*/

class DepthImageDB: public MicrosoftResearch::Cambridge::Sherwood::IDataPointCollection
{
public:
    typedef unsigned int index_type;
    virtual bool getDataPoint(index_type i, cv::Mat &img, cv::Point2i &coordinate) = 0;
};

class GeneralStringParser
{
public:
    virtual std::string getFilename() = 0;
    virtual void setString(const std::string &str) = 0;
};

class SimpleParser: public GeneralStringParser
{
public:
    virtual void setString(const std::string &str){
        str_ = str;
    }
    virtual std::string getFilename(){
        return str_;
    }

private:
    std::string str_;
};


class DepthFileBasedImageDB: public DepthImageDB
{
    typedef unsigned short fileindex_type;

public:
    DepthFileBasedImageDB(const std::string &filename,
                          GeneralStringParser &parser,
                          const std::string &base = "",
                          bool constImgSize = false);

    bool getDataPoint(index_type i, cv::Mat &img, cv::Point2i &coordinate);
    virtual bool getDataPoint(index_type i, std::string &file, cv::Point2i &coordinate);

    fileindex_type getImageIdx(index_type i) const
    {
        return pointsIndex_[i].first;
    }

    std::string imageIdx2Filename(fileindex_type i) const
    {
        return files_[i];
    }

protected:
    typedef std::pair<fileindex_type,unsigned short> filebased_type; // index -> (filename,i)

    /*define this function to add aditional parts of the database*/
    virtual bool postprocessFile(const cv::Mat &image,GeneralStringParser &parser);
    void push_pixel(unsigned short index);

    filebased_type getIndex(index_type i) const{
        return pointsIndex_[i];
    }

    unsigned short point2index(cv::Point2i p,cv::Size size) const{
        return p.x+p.y*size.width;
    }

    cv::Point2i index2point(unsigned short i,cv::Size size) const{
        return cv::Point2i(i % size.width, i / size.width);
    }

    cv::Size imgSize_;
    bool constImgSize_;

private:
    /*reads files one by one*/
    void readFiles(const std::string &file, GeneralStringParser &parser);
    /*simple imlementation to push pixels to an array; called from postprocessFile()*/


    std::vector<std::string> files_;
    std::string path_;

    cv::Mat cache_;

    fileindex_type previous_;
    index_type elementCount_;

    ArrayList<filebased_type> pointsIndex_;
};

#endif // DEPTHIMAGEDB_H
