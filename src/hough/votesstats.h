#ifndef VOTESSTATS_H
#define VOTESSTATS_H

#include "Interfaces.h"
#include "hough/depthdbwithvotes.h"

#include <vector>

class VotesStats
{
public:


    VotesStats(unsigned char voteClasses)
    {
        dthreashold2_ = 10;
        pointCount_=0;
        voteClasses_ = voteClasses;

        for(int i=0 ; i < voteClasses_; i++){
            votes_.push_back(std::vector<cv::Point2i>());
        }
    }

    void Clear()
    {
       for(int i=0 ; i < voteClasses_; i++){
           votes_[i].clear();
       }
    }

    void Aggregate(MicrosoftResearch::Cambridge::Sherwood::IDataPointCollection& data, unsigned int index);

    void Aggregate(const VotesStats& i);

    int Size(){
        return pointCount_;
    }

    int Classes(){
        return voteClasses_;
    }

    void Compress();

    double VoteVariance() const;

    virtual VotesStats DeepClone() const;

private:

    unsigned int norm2(int x, int y) const
    {
        return x*x+y*y;
    }

    double norm2(double x, double y) const
    {
        return x*x+y*y;
    }

    std::vector< std::vector<cv::Point2i> > votes_;

    int dthreashold2_;
    unsigned char voteClasses_;
    unsigned int pointCount_;

};

#endif // VOTESSTATS_H
