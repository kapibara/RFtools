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

    void Aggregate(const cv::Point2i &abs, const VotesStats& i);

    void Aggregate(const cv::Point2i &abs, const cv::Point2i &vote);

    void Aggregate(const HoughVotesStats &stats);

    bool Serialize(std::ostream &out);
    bool Serialize(const std::string &filename);


private:

    cv::Mat mat_;
    unsigned char voteClass_;

    unsigned int outOfBoundaries_;
};

#endif // HOUGHVOTESSTATS_H
