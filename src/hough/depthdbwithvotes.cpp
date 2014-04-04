#include "depthdbwithvotes.h"

DepthDBWithVotesImpl::DepthDBWithVotesImpl(const std::string &basepath):
    DepthFileBasedImageDBImpl(basepath,true)
{
    voteClassCount_ = 0;
}

bool DepthDBWithVotesImpl::loadDB(const std::string &filename)
{
    StringParserWithOffsert parser;
    bool result =  DepthFileBasedImageDBImpl::loadDB(filename, parser);
    isRelative_.resize(this->voteClassCount(),true);
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
