#include "depthdbwithvotes.h"

DepthDBWithVotesImpl::DepthDBWithVotesImpl(const std::string &basepath):
    DepthFileBasedImageDBImpl(basepath,true)
{

}

bool DepthDBWithVotesImpl::loadDB(const std::string &filename)
{
    return DepthFileBasedImageDBImpl::loadDB(filename, StringParserWithOffsert());
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

    return true;
}


bool DepthDBWithVotesImpl::getDataPointVote(index_type i, std::vector<cv::Point2i> &vote)
{
    filebased_type pair = getIndex(i);
    cv::Point2i x = index2point(pair.second,imgSize_);//we know that imgSize is constant

    for(int i=0; i<votes_[pair.first].size();i++){
        vote.push_back(votes_[pair.first][i]-x);
    }

    return true;
}
