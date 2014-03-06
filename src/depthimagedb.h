#ifndef DEPTHIMAGEDB_H
#define DEPTHIMAGEDB_H

#include <opencv2/opencv.hpp>

#include <vector>

#include "Interfaces.h"
#include "arraylist.h"

/*DB interface required to compute depth features*/

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

class DepthImageDB: public MicrosoftResearch::Cambridge::Sherwood::IDataPointCollection
{
public:
    typedef unsigned int index_type;
    virtual bool getDataPoint(index_type i, cv::Mat &img, cv::Point2i &coordinate) = 0;
};

class DepthFileBasedImageDB: public DepthImageDB
{
public:
    typedef unsigned short fileindex_type;
    virtual bool getDataPoint(index_type i, std::string &file, cv::Point2i &coordinate) = 0;
    virtual bool getDataPoint(index_type i, cv::Mat &img, cv::Point2i &coordinate) = 0;
    virtual fileindex_type getImageIdx(index_type i) const = 0;
    virtual std::string imageIdx2Filename(fileindex_type i) const = 0;
    virtual fileindex_type imageCount() const = 0;
    virtual unsigned int clearCacheCallCount() = 0;
};



class Cache
{
public:
    unsigned int Count() const{
        return subindex_.size();
    }

    void addToCache(const std::string &file);

    bool getImage(fileindex_type imgindex, cv::Mat &image);

    fileindex_type imageCount() const
    {
        return files_.size();
    }

    unsigned int clearCacheCallCount(){
        unsigned int tmp = cachCallCount_;
        cachCallCount_ = 0;
        return tmp;
    }

private:
    std::vector<std::string> files_;
    unsigned int cachCallCount_;
    cv::Mat cached_;
    fileindex_type previous_;
};

class SubindexFileBasedImageDB: public DepthFileBasedImageDB
    {
    public:
    SubindexFileBasedImageDB(DepthFileBasedImageDB &source, const std::vector<index_type> &subindex):
        source_(source),subindex_(subindex)
    {
        for(int i=0; i<subindex_.size(); i++){
            imageids_.insert(std::make_pair(source_.getImageIdx(subindex_[i]),imageids_.size()));
        }
    }

    bool getDataPoint(index_type i, std::string &file, cv::Point2i &coordinate){
        return source_.getDataPoint(subindex_[i],file,coordinate);
    }

    bool getDataPoint(index_type i, cv::Mat &img, cv::Point2i &coordinate){
        return source_.getDataPoint(subindex_[i],img,coordinate);
    }

    std::string imageIdx2Filename(fileindex_type i) const{
       return source_.imageIdx2Filename(i);
    }

    fileindex_type imageCount() const{
        return imageids_.size();
    }

    fileindex_type getOriginalImageIdx(index_type i) const {
        return source_.getImageIdx(subindex_[i]);
    }

    fileindex_type getImageIdx(index_type i) const {
        return imageids_.at(source_.getImageIdx(subindex_[i]));
    }

    unsigned int clearCacheCallCount(){
        return source_.clearCacheCallCount();
    }

protected:
    DepthFileBasedImageDB &source_;
    std::vector<index_type> subindex_;

private:
    std::map<fileindex_type,fileindex_type> imageids_;
};


class DepthFileBasedImageDBImpl: public DepthFileBasedImageDB
{

public:
    DepthFileBasedImageDBImpl(const std::string &base = "",
                              bool constImgSize = false);

    virtual bool loadDB(const std::string &filename, GeneralStringParser &stringParser);

    bool getDataPoint(index_type i, cv::Mat &img, cv::Point2i &coordinate);
    bool getDataPoint(index_type i, std::string &file, cv::Point2i &coordinate);

    fileindex_type getImageIdx(index_type i) const
    {
        return pointsIndex_[i].first;
    }

    std::string imageIdx2Filename(fileindex_type i) const
    {
        return files_[i];
    }

    fileindex_type imageCount() const{
        return files_.size();
    }

    index_type Count() const{
        return pointsIndex_.size();
    }

    virtual unsigned int clearCacheCallCount(){
        unsigned int tmp = cachCallCount_;
        cachCallCount_ = 0;
        return tmp;
    }

protected:
    typedef std::pair<fileindex_type,unsigned short> filebased_type; // index -> (filename,i)

    /*redefine this function to add aditional parts of the database*/
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
    unsigned int cachCallCount_;

    Cache cache_;
    std::string path_;

    index_type elementCount_;

    ArrayList<filebased_type> pointsIndex_;
};

#endif // DEPTHIMAGEDB_H
