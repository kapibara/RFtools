#ifndef VOTESSTATS_H
#define VOTESSTATS_H

#include "Interfaces.h"
#include "hough/depthdbwithvotes.h"

#include <vector>
#include <list>

//#define ENABLE_OVERFLOW_CHECKS

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
        variance_ = -1;

        for(int i=0 ; i < voteClasses_; i++){
            votes_.push_back(voteVector());
            mx_.push_back(0);
            my_.push_back(0);
            mx2_.push_back(0);
            my2_.push_back(0);
            votesCount_.push_back(0);
        }
    }

    void Clear()
    {
       for(int i=0 ; i < voteClasses_; i++){
           votes_[i].clear();
           mx_[i] = 0;
           my_[i] = 0;
           mx2_[i] = 0;
           my2_[i] = 0;
           votesCount_[i] = 0;
       }
       pointCount_ = 0;
       variance_ = -1;
    }

    const_iterator begin(unsigned char voteClass) const{
        return votes_[voteClass].begin();
    }

    const_iterator end(unsigned char voteClass) const{
        return votes_[voteClass].end();
    }

    void Aggregate(MicrosoftResearch::Cambridge::Sherwood::IDataPointCollection& data, unsigned int index);

    void Aggregate(const VotesStats& i);

    element_count Count() const
    {
        return pointCount_;
    }

    int Classes() const
    {
        return voteClasses_;
    }

    void Compress();

    double VoteVariance();

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
    std::vector<element_count> votesCount_;
    std::vector< double > mx_;
    std::vector< double > mx2_;
    std::vector< double > my_;
    std::vector< double > my2_;

    int dthreashold2_;
    unsigned char voteClasses_;
    element_count pointCount_;

    double variance_;

};

#endif // VOTESSTATS_H
