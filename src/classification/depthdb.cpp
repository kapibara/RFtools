#include "depthdb.h"

#include <fstream>
#include <iostream>
#include <map>

#include "split.h"

using namespace std;

void LabelPerImageStringParser::setString(const std::string &str){
    strsplit_.clear();
    split(str,",",strsplit_);
}


DepthDB::DepthDB(const std::string &file, const std::string &basepath, unsigned int maxCacheSize):
    CACHESIZE(maxCacheSize)
{
    path_ = basepath;
    elementCount_ = 0;
    lastAccesed_ = 0;
    previous_ = 0;

    try{
        readFiles(file);
    }catch(std::bad_alloc &e){
        std::cerr << "bad allocation expcetion caught" << std::endl;
    }
}

bool DepthDB::getDataPoint(unsigned int i, cv::Mat &img, cv::Point2i &coordinate)
{
    if (i >= pointsIndex_.size()){
        return false;
    }

    if (isInCache_[pointsIndex_[i].first]>=0){
        //the image is in cache
        img = cache_[isInCache_[pointsIndex_[i].first]];
    }else{
        //the images is not in cache
        img = cv::imread(files_[pointsIndex_[i].first],-1);

        //todo: reimplement cashing strategy - right now cash consists of only one image
        cache_[lastAccesed_] = img;
        isInCache_[pointsIndex_[i].first] = lastAccesed_;
    }

    int idx = pointsIndex_[i].second; //pixel index on the image
    coordinate.x = idx % img.cols;
    coordinate.y = idx / img.cols;

    return true;
}

bool DepthDB::getDataPoint(unsigned int i, std::string &file, cv::Point2i &coordinate)
{

    if (i >= pointsIndex_.size()){
        return false;
    }

    cv::Mat img;
    file = files_[pointsIndex_[i].first];

    if (isInCache_[pointsIndex_[i].first]>=0){
        //the image is in cache
        img = cache_[isInCache_[pointsIndex_[i].first]];
    }else{

        //the images is not in cache
        img = cv::imread(file,-1);

        //todo: reimplement cashing strategy - right now cash consists of only one image
        cache_[lastAccesed_] = img;
        isInCache_[previous_] = -1;
        previous_ = pointsIndex_[i].first;
        isInCache_[pointsIndex_[i].first] = lastAccesed_;

    }

    int idx = pointsIndex_[i].second; //pixel index on the image
    coordinate.x = idx % img.cols;
    coordinate.y = idx / img.cols;

    return true;
}

void DepthDB::readFiles(const std::string &file)
{
    ifstream input(file.c_str());
    cv::Mat tmpMat;
    string tmp;
    vector<string> sep;

    if (input.is_open()){

        while(!input.eof()){

            input >> tmp;

            sep.clear();
            split(tmp,std::string(","),sep);

            assert(sep.size() == 2);

            if (path_.length()>0){
                sep[0] = path_+sep[0];
            }

            std::cout << "reading file: " << sep[0]<< std::endl;

            files_.push_back(sep[0]);

            tmpMat = cv::imread(sep[0],-1);

            if (elementCount_ < CACHESIZE){
                cache_.push_back(tmpMat);
                previous_ = elementCount_;
                isInCache_.push_back(elementCount_);
            }else{
                isInCache_.push_back(-1);
            }
            std::pair<std::map<std::string,label_type>::iterator,bool> itor = labels_.insert(labelmap_type(sep[1],labels_.size()));

            datalabels_.push_back(itor.first->second);

            push_pixels(tmpMat);

            elementCount_++;
        }
    }

    classCount_ = labels_.size();
}

void DepthDB::push_pixels(const cv::Mat &mat){


    int rows = mat.rows,cols = mat.cols;
    const unsigned short *dataptr;

    if (mat.isContinuous()){
        cols *= rows;
        rows = 1;
    }

    for(int i=0; i<rows; i++){

        dataptr = mat.ptr<unsigned short>(i);

        for(int j=0; j<cols; j++){
            if (dataptr[j]>0){
                //if depth pixel has a value
                try{
                    pointsIndex_.push_back(filebased_type(files_.size()-1 ,j+i*cols));
                }catch(std::bad_alloc &e){
                    std::cerr << "bad allocation pointsIndex_:" << pointsIndex_.vector_size() << std::endl;
                }
            }
        }
    }


}
