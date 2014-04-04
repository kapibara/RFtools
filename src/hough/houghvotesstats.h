#ifndef HOUGHVOTESSTATS_H
#define HOUGHVOTESSTATS_H

#include <opencv2/opencv.hpp>
#include <hough/votesstats.h>

class HoughVotesStats
{
public:
    HoughVotesStats(const cv::Size &s, unsigned char voteClass);

    void Clear(){
        outOfBoundaries_ = 0;
        mat_.setTo(0);
    }

    bool Aggregate(const cv::Point2i &abs,const VotesStats& i);
    bool AggregateOld(const cv::Point2i &abs, const VotesStats& stats);

    void Aggregate(const cv::Point2i &abs, const cv::Point2i &vote);

    void Aggregate(const HoughVotesStats &stats);

    bool Serialize(std::ostream &out);
    bool Serialize(const std::string &filename);

    unsigned char voteClassCount()
    {
        return voteClass_;
    }

    unsigned int outOfBoundariesCount()
    {
        return outOfBoundaries_;
    }

    void setGT(const cv::Point2i &gt){
        gt_ = gt;
    }


private:

    cv::Mat mat_;
    unsigned char voteClass_;

    unsigned int outOfBoundaries_;
    cv::Point2i gt_;
};

#endif // HOUGHVOTESSTATS_H
