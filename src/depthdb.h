#ifndef DEPTHDB_H
#define DEPTHDB_H

#include "Interfaces.h"
#include "arraylist.h"
#include "depthfeature.h"

#include <iostream>
#include <string>

#include <utility>

#include "opencv2/opencv.hpp"

class ClassificationDB: public MicrosoftResearch::Cambridge::Sherwood::IDataPointCollection
{
public:
    typedef unsigned short label_type;
    typedef unsigned int index_type;
    typedef unsigned short fileindex_type;

    virtual label_type getNumericalLabel(index_type i) const = 0;
    virtual bool getDataPoint(index_type i, cv::Mat &img, cv::Point2i &coordinate) = 0;
    virtual fileindex_type getImageIdx(index_type i) const = 0;
};

class DepthDBSubindex : public ClassificationDB
{
public:
    DepthDBSubindex(ClassificationDB &source, const std::vector<ClassificationDB::index_type> &subindex):source_(source),subindex_(subindex)
    {
        std::set<ClassificationDB::index_type> tmp;
        for(int i=0; i<subindex_.size(); i++){
            tmp.insert(getImageIdx(i));
        }

        imageids_.insert(imageids_.begin(),tmp.begin(),tmp.end());
    }

    unsigned int Count() const{
        return subindex_.size();
    }

    label_type getNumericalLabel(index_type i) const{
        return source_.getNumericalLabel(subindex_[i]);
    }

    fileindex_type imageCount() const{
        return imageids_.size();
    }


    fileindex_type getImageIndex(fileindex_type i) const{
        std::cout << imageids_.size() << ";" << i << std::endl;
        return imageids_[i];
    }

    bool getDataPoint(index_type i, cv::Mat &img, cv::Point2i &coordinate){
        return source_.getDataPoint(subindex_[i],img,coordinate);
    }

    fileindex_type getImageIdx(index_type i) const {
        return source_.getImageIdx(subindex_[i]);
    }

private:
    ClassificationDB &source_;
    const std::vector<ClassificationDB::index_type> &subindex_;
    std::vector<ClassificationDB::fileindex_type> imageids_;

};

class DepthDB  : public ClassificationDB
{

public:
    /*basepath --- database path; for portability
      file --- file containing all db files (1 file per line, relative to basepath*/
    DepthDB(const std::string &file, const std::string &basepath="", unsigned int maxCacheSize = 1);

    bool getDataPoint(index_type i, cv::Mat &img, cv::Point2i &coordinate);
    bool getDataPoint(index_type i, std::string &file, cv::Point2i &coordinate);

    fileindex_type getImageIdx(index_type i) const{
        return pointsIndex_[i].first;
    }

    std::string imgIdx2Name(fileindex_type img_idx){
        return files_[img_idx];
    }

    std::string labelIdx2Name(label_type label_idx){
        for(std::map<std::string,label_type>::iterator itor = labels_.begin();
            itor!=labels_.end();itor++){
            if(itor->second == label_idx){
                return itor->first;
            }
        }
        return "";
    }

    label_type getNumericalLabel(fileindex_type i) const{
        return datalabels_[i];
    }

    label_type getNumericalLabel(index_type i) const
    {
        return datalabels_[pointsIndex_[i].first];
    }

    unsigned short classCount() const
    {
        return classCount_;
    }

    unsigned short imageCount() const
    {
        return files_.size();
    }

    unsigned int Count() const{
        return pointsIndex_.size();
    }

private:

    void readFiles(const std::string &file);
    void push_pixels(const cv::Mat &mat);

    std::string path_;


    std::vector<int> isInCache_;

    // numerical class label for each file (NOT pixel)
    // has the same length as files_ and the same indexing
    std::vector<std::string> files_;
    std::vector<label_type> datalabels_;

    //maps numerical data label values to the string values
    std::map<std::string,label_type> labels_;
    typedef std::pair<std::string,label_type > labelmap_type;

    std::vector<cv::Mat> cache_;

    typedef std::pair<fileindex_type,unsigned short> filebased_type; // index -> (filename,i)
    ArrayList<filebased_type> pointsIndex_;

    int lastAccesed_;
    int previous_;

    unsigned short classCount_;
    unsigned short elementCount_;

    const unsigned short CACHESIZE;
};

#endif // DEPTHDB_H
