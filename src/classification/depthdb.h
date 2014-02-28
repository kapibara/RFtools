#ifndef DEPTHDB_H
#define DEPTHDB_H

#include "Interfaces.h"
#include "arraylist.h"
#include "depthimagedb.h"

#include <iostream>
#include <string>

#include <utility>

#include "opencv2/opencv.hpp"

class LabelPerImageStringParser: public GeneralStringParser
{
public:
    void setString(const std::string &str);
    std::string getFilename(){
        return strsplit_[0];
    }

    std::string getLabel(){
        return strsplit_[1];
    }

private:
    std::vector<std::string> strsplit_;
};

class ClassificationDB
{
public:
    typedef unsigned short label_type;

    virtual label_type getNumericalLabel(DepthFileBasedImageDB::index_type i) const = 0;
    virtual std::string labelIndex2Name(label_type label_idx) const = 0;
    virtual label_type classCount() const = 0;
};

class DepthDBClassImage: public DepthFileBasedImageDBImpl ,public ClassificationDB
{

public:
    /*basepath --- database path; for portability
      file --- file containing all db files (1 file per line, relative to basepath*/
    DepthDBClassImage(const std::string &basepath="");

    bool loadDB(const std::string &filename);

    std::string labelIndex2Name(label_type label_idx) const{
        for(std::map<std::string,label_type>::const_iterator itor = labels_.begin();
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
        return datalabels_[getImageIdx(i)];
    }

    unsigned short classCount() const
    {
        return labels_.size();
    }

protected:
    virtual bool postprocessFile(const cv::Mat &image,GeneralStringParser &parser);

private:
    std::vector<label_type> datalabels_;

    //maps numerical data label values to the string values
    std::map<std::string,label_type> labels_;
    typedef std::pair<std::string,label_type > labelmap_type;
};

class DepthDBSubindex : public DepthFileBasedImageDB, public ClassificationDB
{
public:
    DepthDBSubindex(DepthFileBasedImageDB &source, const std::vector<index_type> &subindex):
        source_(source),subindex_(subindex)
    {
        for(int i=0; i<subindex_.size(); i++){
            imageids_.insert(std::make_pair(source_.getImageIdx(subindex_[i]),imageids_.size()));
        }
    }

    unsigned int Count() const{
        return subindex_.size();
    }

    bool getDataPoint(index_type i, std::string &file, cv::Point2i &coordinate){
        return source_.getDataPoint(i,file,coordinate);
    }

    bool getDataPoint(index_type i, cv::Mat &img, cv::Point2i &coordinate){
        return ((DepthImageDB &)source_).getDataPoint(i,img,coordinate);
    }

    label_type getNumericalLabel(index_type i) const{
        return dynamic_cast<ClassificationDB &>(source_).getNumericalLabel(subindex_[i]);
    }

    std::string labelIndex2Name(label_type label_idx) const{
            return dynamic_cast<ClassificationDB &>(source_).labelIndex2Name(label_idx);
    }

    label_type classCount() const{
        return dynamic_cast<ClassificationDB &>(source_).classCount();
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

private:
    DepthFileBasedImageDB &source_;
    std::vector<index_type> subindex_;
    std::map<fileindex_type,fileindex_type> imageids_;

};

#endif // DEPTHDB_H
