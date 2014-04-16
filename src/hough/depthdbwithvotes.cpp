#include "depthdbwithvotes.h"

DepthDBWithVotesImpl::DepthDBWithVotesImpl(const std::string &basepath):
    DepthFileBasedImageDBImpl(basepath,true)
{
    voteClassCount_ = 0;
}

bool DepthDBWithVotesImpl::loadDB(const std::string &filename,  bool hasHeader)
{
    StringParserWithOffsert parser;


    bool result =  DepthFileBasedImageDBImpl::loadDB(filename, parser,hasHeader);

    if(!hasHeader){
        isRelative_.resize(voteClassCount_,true);
    }

    return result;
}


bool DepthDBWithVotesImpl::postprocessFile(const cv::Mat &mat, GeneralStringParser &parser)
{
    /*put all pixels into the array*/
    DepthFileBasedImageDBImpl::postprocessFile(mat,parser);

    StringParserWithOffsert &typedparser = dynamic_cast<StringParserWithOffsert &>(parser);
    std::vector<cv::Point2i> joints;

    /*add joint locations*/
    typedparser.getJoints(joints);
    votes_.push_back(joints);

    if (voteClassCount_<joints.size())
        voteClassCount_ = joints.size();

    return true;
}


void DepthDBWithVotesImpl::processHeader(const std::string &header)
{
    std::vector< std::string > splitResult;

    split(header,",",splitResult);

    std::cerr << splitResult.size() << std::endl;

    isRelative_.resize((splitResult.size()-1)/2,true);

    for(int i=1; i<splitResult.size(); i+=2){
        std::cerr << "vote index: " << (i-1)/2 << " splitResult[i]: " << splitResult[i] << std::endl;
        if(strcmp(splitResult[i].c_str(),"a") == 0){
            isRelative_[(i-1)/2] = false;
        }
    }
}


bool DepthDBWithVotesImpl::getDataPointVote(index_type i, std::vector<cv::Point2i> &vote)
{
    if (vote.size() != voteClassCount_){
        std::cerr << "vote container does not have size " << (int)voteClassCount_ << std::endl;
        std::cerr.flush();
    }

    filebased_type pair = getIndex(i);

    cv::Point2i x = index2point(pair.second,imgSize_);//we know that imgSize is constant

    for(int i=0; i<votes_[pair.first].size();i++){
        if (isRelative_[i])
            vote[i] = votes_[pair.first][i]-x;
        else
            vote[i] = votes_[pair.first][i];

    }

    return true;
}

