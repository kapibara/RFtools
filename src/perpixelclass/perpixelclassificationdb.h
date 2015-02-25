#ifndef PERPIXELCLASSIFICATIONDB_H
#define PERPIXELCLASSIFICATIONDB_H

#include "depthimagedb.h"
#include "classification/depthdb.h"
#include "imagecache.h"

#include "string2number.hpp"

class PerPixelClassificationDBParser: public GeneralStringParser
{
public:
   virtual std::string getFilename(){
        return strsplit_[0];
   }

   std::string getLabelsFilename(){
        return strsplit_[1];
   }

   virtual void setString(const std::string &str){
        strsplit_.clear();
        split(str,",",strsplit_);
    }

private:
    std::vector<std::string> strsplit_;
};

class PerPixelClassificationDB: public DepthFileBasedImageDBImpl, public ClassificationDB
{
public:
    PerPixelClassificationDB(const std::string &base = ""):DepthFileBasedImageDBImpl(base,true)
    {
        classCount_ = 0;
        lblimgcache_ = new InMemoryCache();
    }

    virtual label_type getNumericalLabel(DepthFileBasedImageDB::index_type i) const
    {
        filebased_type pair = getIndex(i);
        cv::Mat img;

        lblimgcache_->getImage(pair.first,img);

        cv::Point2i coordinate = index2point(pair.second,img.size());

        return img.at<char>(coordinate)-1; //labels from 1 to 16 -> should be from 0 - 15
    }

    virtual bool loadDB(const std::string &filename)
    {
        PerPixelClassificationDBParser parser;

        bool result =  DepthFileBasedImageDBImpl::loadDB(filename, parser, true);

        return result;
    }

    virtual std::string labelIndex2Name(label_type label_idx) const
    {
        return ""; //dont need
    }

    virtual label_type classCount() const
    {
        return classCount_;
    }

protected:

    virtual bool postprocessFile(const cv::Mat &image,GeneralStringParser &parser)
    {
        /*put all pixels into the array*/
        DepthFileBasedImageDBImpl::postprocessFile(image,parser);

        PerPixelClassificationDBParser &typedparser = dynamic_cast<PerPixelClassificationDBParser &>(parser);

        std::string lblimage = typedparser.getLabelsFilename();

        lblimgcache_->addToCache(lblimage);

        return true;
    }

    virtual void processHeader(const std::string &header)
    {
        classCount_ = str2num<int>(header);
    }


private:
    int classCount_;

    ICache *lblimgcache_;

};

#endif // PERPIXELCLASSIFICATIONDB_H
