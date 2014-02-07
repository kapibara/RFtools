#ifndef INMEMDB_H
#define INMEMDB_H

#include "arraylist.h"
#include <opencv2/opencv.hpp>
#include <Interfaces.h>

class InMemDB: public MicrosoftResearch::Cambridge::Sherwood::IDataPointCollection
{
public:
    typedef unsigned short label_type;
    typedef unsigned int index_type;

    InMemDB(const std::string &file, const std::string &basepath="");
    bool getDataPoint(index_type i, cv::Mat &img, cv::Point2i &coordinate);
    bool getDataPoint(index_type i, std::string &file, cv::Point2i &coordinate);

    unsigned short getImageIdx(index_type i){

    }

    std::string imgIdx2Name(unsigned short img_idx){

    }

    std::string labelIdx2Name(label_type label_idx){

    }

    label_type getNumericalLabel(unsigned short i) const{

    }

    label_type getNumericalLabel(index_type i) const
    {

    }

    unsigned short classCount() const
    {

    }

    unsigned short imageCount() const
    {

    }

    unsigned int Count() const{

    }

private:

    typedef std::pair<cv::Mat,cv::Point2i> mat_type;
    typedef std::pair<unsigned short,unsigned short> filebased_type; // index -> (filename,i)

    ArrayList<filebased_type> pointsIndex_;
    ArrayList<mat_type> matrices_;

    std::vector<label_type> datalabels_;

    //maps numerical data label values to the string values
    std::map<std::string,label_type> labels_;
    typedef std::pair<std::string,label_type > labelmap_type;
};

#endif // INMEMDB_H
