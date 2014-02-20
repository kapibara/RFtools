#ifndef HOUGHVOTESSTATS_H
#define HOUGHVOTESSTATS_H

#include <opencv2/opencv.hpp>
#include <hough/votesstats.h>

class HoughVotesStats
{
public:
    HoughVotesStats(const cv::Size &s, unsigned char voteClass);

    void Aggregate(const cv::Point2i &abs, const VotesStats& i);

    void Aggregate(const cv::Point2i &abs, const cv::Point2i &vote);

    bool Serialize(std::ostream &out);
    bool Serialize(std::string &filename);


private:

    cv::Mat mat_;
    unsigned char voteClass_;

    unsigned int outOfBoundaries_;
};

#endif // HOUGHVOTESSTATS_H
