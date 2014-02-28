#ifndef VOTESSTATS_H
#define VOTESSTATS_H

#include "Interfaces.h"
#include "hough/depthdbwithvotes.h"

#include <vector>
#include <list>

#define ENABLE_OVERFLOW_CHECKS

class VotesStats
{
    typedef std::list<cv::Point2i> voteVector;
public:

    typedef voteVector::const_iterator const_iterator;
    typedef unsigned int element_count;

    VotesStats(unsigned char voteClasses = 0)
    {
        dthreashold2_ = 100*100;
        pointCount_=0;
        voteClasses_ = voteClasses;

        for(int i=0 ; i < voteClasses_; i++){
            votes_.push_back(voteVector());
        }
    }

    void Clear()
    {
       for(int i=0 ; i < voteClasses_; i++){
           votes_[i].clear();
       }
       pointCount_ = 0;
    }

    const_iterator begin(unsigned char voteClass) const{
        return votes_[voteClass].begin();
    }

    const_iterator end(unsigned char voteClass) const{
        return votes_[voteClass].end();
    }

    void Aggregate(MicrosoftResearch::Cambridge::Sherwood::IDataPointCollection& data, unsigned int index);

    void Aggregate(const VotesStats& i);

    element_count Size() const
    {
        return pointCount_;
    }

    int Classes() const
    {
        return voteClasses_;
    }

    void Compress();

    double VoteVariance() const;

    bool Serialize(std::ostream &stream) const;
    bool SerializeChar(std::ostream &stream) const;
    bool Deserialize(std::istream &stream);

    virtual VotesStats DeepClone() const
    {
        return VotesStats(*this);
    }

private:

    unsigned int norm2(int x, int y) const
    {
        return x*x+y*y;
    }

    double norm2(double x, double y) const
    {
        return x*x+y*y;
    }

    //very memory-unfriendly; in a cv::Mat, quantized?
    std::vector< voteVector > votes_;

    int dthreashold2_;
    unsigned char voteClasses_;
    element_count pointCount_;

};

#endif // VOTESSTATS_H
