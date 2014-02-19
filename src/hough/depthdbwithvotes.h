#ifndef DEPTHDBWITHVOTES_H
#define DEPTHDBWITHVOTES_H

#include "depthimagedb.h"
#include "classification/depthdb.h"

#include "string2number.hpp"
#include "split.h"

class StringParserWithOffsert: public GeneralStringParser
{
public:
   virtual std::string getFilename(){
        return strsplit_[0];
   }

   void getJoints(std::vector<cv::Point2i> &j){
        for(int i=0; i<strsplit_.size();i+=2){
            j.push_back(cv::Point2i(str2num<int>(strsplit_[i]),str2num<int>(strsplit_[i+1])));
        }
   }

   virtual void setString(const std::string &str){
        strsplit_.clear();
        split(str,",",strsplit_);
    }

private:
    std::vector<std::string> strsplit_;
};

class DepthDBWithVotes
{
public:
    virtual bool getDataPointVote(DepthFileBasedImageDB::index_type i, std::vector<cv::Point2i> &vote) = 0;
};

class DepthDBWithVotesImpl: public DepthFileBasedImageDBImpl, public DepthDBWithVotes
{
public:
    DepthDBWithVotesImpl(const std::string &basepath="");

    bool loadDB(const std::string &filename);

    bool getDataPointVote(index_type i, std::vector<cv::Point2i> &vote);

protected:
    bool postprocessFile(const cv::Mat &image, GeneralStringParser &parser);

private:

    std::vector<std::vector<cv::Point2i> > votes_; //stores joint locations for each image
};

#endif // DEPTHDBWITHVOTES_H