#ifndef HOUGHTRAININGCONTEXT_H
#define HOUGHTRAININGCONTEXT_H

#include "Interfaces.h"

#include "depthfeature.h"
#include "hough/votesstats.h"

template <class F>
class HoughTrainingContext: public MicrosoftResearch::Cambridge::Sherwood::ITrainingContext<F,VotesStats>
{
public:
    HoughTrainingContext(unsigned char nClasses_,DepthFeatureFactory &factory):factory_(factory)
    {
        nClasses_ = VoteState.;
    }

    virtual F GetRandomFeature(MicrosoftResearch::Cambridge::Sherwood::Random& random)
    {
        return factory_.getDepthFeature(random);
    }

    virtual VotesStats GetStatisticsAggregator()
    {
        return VotesStats(nClasses_);
    }

    virtual double ComputeInformationGain(const VotesStats& parent, const VotesStats& leftChild, const VotesStats& rightChild)
    {
        double lvv = leftChild.VoteVariance();
        double rvv = rightChild.VoteVariance();

        if(parent.Size()<=1){
            return 0;
        }
        /*computing parent.VoteVariance() is not optimal*/

        return (parent.VoteVariance() - (((double)leftChild.Size())/parent.Size()*lvv +((double)rightChild.Size())/parent.Size()*rvv));
    }

    virtual bool ShouldTerminate(const VotesStats& parent, const VotesStats& leftChild, const VotesStats& rightChild, double gain)
    {
        return gain < 0.01;
    }

private:
    DepthFeatureFactory &factory_;
    unsigned char nClasses_;
};

#endif // HOUGHTRAININGCONTEXT_H
