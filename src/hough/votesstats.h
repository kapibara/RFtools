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

    VotesStats(unsigned char voteClasses = 0):
        mx_(voteClasses,0),
        my_(voteClasses,0),
        mx2_(voteClasses,0),
        my2_(voteClasses,0),
/*        minx_(voteClasses,0),
        miny_(voteClasses,0),
        maxx_(voteClasses,0),
        maxy_(voteClasses,0),*/
        votesCount_(voteClasses,0),
        container_(voteClasses),
        votes_(voteClasses,voteVector())

    {
        dthreashold2_ = 100*100;
        pointCount_=0;
        voteClasses_ = voteClasses;
        variance_ = -1;
        aggregationValid_ = true;
        fullStats_ = true;

    }

    void Clear()
    {

       mx_.assign(mx_.size(),0);
       my_.assign(my_.size(),0);
       mx2_.assign(mx2_.size(),0);
       my2_.assign(my2_.size(),0);
/*       minx_.assign(minx_.size(),0);
       miny_.assign(miny_.size(),0);
       maxx_.assign(maxx_.size(),0);
       maxy_.assign(maxy_.size(),0);*/
       votesCount_.assign(votesCount_.size(),0);

       for(int i=0 ; i < voteClasses_; i++){
           votes_[i].clear();

       }
       pointCount_ = 0;
       variance_ = -1;
       aggregationValid_ = true;
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

    void FullStats(bool compute)
    {
        fullStats_ = compute;
    }

    void toMatrices();

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
    std::vector< cv::Mat > matrixStats_;
/*    std::vector< int> minx_;
    std::vector< int> miny_;
    std::vector< int > maxx_;
    std::vector< int > maxy_;*/
    std::vector<cv::Point2i> container_;

    int dthreashold2_;
    unsigned char voteClasses_;
    element_count pointCount_;
    bool fullStats_;

    double variance_;
    bool aggregationValid_;

};

#endif // VOTESSTATS_H
