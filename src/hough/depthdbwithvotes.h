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
        for(int i=1; i<strsplit_.size();i+=2){
            j.push_back(cv::Point2i(std::floor(str2num<float>(strsplit_[i])),
                                    std::floor(str2num<float>(strsplit_[i+1]))));
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
    typedef unsigned char vote_class_count;
    virtual bool getDataPointVote(DepthFileBasedImageDB::index_type i, std::vector<cv::Point2i> &vote) = 0;
    virtual vote_class_count voteClassCount() = 0;
};

class DepthDBWithVotesImpl: public DepthFileBasedImageDBImpl, public DepthDBWithVotes
{
public:


    DepthDBWithVotesImpl(const std::string &basepath="");

    bool loadDB(const std::string &filename);

    bool getDataPointVote(index_type i, std::vector<cv::Point2i> &vote);

    vote_class_count voteClassCount(){
        return voteClassCount_;
    }

protected:
    bool postprocessFile(const cv::Mat &image, GeneralStringParser &parser);

private:

    vote_class_count voteClassCount_;
    std::vector<std::vector<cv::Point2i> > votes_; //stores joint locations for each image
};

class DepthDBWithVotesSubindex: public SubindexFileBasedImageDB, public DepthDBWithVotes
{
public:
    DepthDBWithVotesSubindex(DepthFileBasedImageDB &source, const std::vector<index_type> &subindex):
        SubindexFileBasedImageDB(source,subindex)
    {

    }

    bool getDataPointVote(index_type i, std::vector<cv::Point2i> &vote){
        return dynamic_cast<DepthDBWithVotes &>(source_).getDataPointVote(subindex_[i], vote);
    }

    vote_class_count voteClassCount(){
        return dynamic_cast<DepthDBWithVotes &>(source_).voteClassCount();
    }

};

#endif // DEPTHDBWITHVOTES_H
