#ifndef HOUGHTRAININGCONTEXT_H
#define HOUGHTRAININGCONTEXT_H

#include "Interfaces.h"

#include "depthfeature.h"
#include "hough/votesstats.h"

template <class F>
class HoughTrainingContext: public MicrosoftResearch::Cambridge::Sherwood::ITrainingContext<F,VotesStats>
{
public:
    HoughTrainingContext(unsigned char nClasses,DepthFeatureFactory &factory):factory_(factory)
    {
        nClasses_ = nClasses;
    }

    virtual F GetRandomFeature(MicrosoftResearch::Cambridge::Sherwood::Random& random)
    {
        return factory_.getDepthFeature(random);
    }

    virtual VotesStats GetStatisticsAggregator()
    {
        return VotesStats(nClasses_);
    }

    virtual double ComputeInformationGain(VotesStats& parent, VotesStats& leftChild, VotesStats& rightChild)
    {
        double lvv = leftChild.VoteVariance();
        double rvv = rightChild.VoteVariance();


        if(parent.Count() <=1){
            return 0;
        }
        /*computing parent.VoteVariance() is not optimal*/

        return ((parent.VoteVariance() - lvv) - rvv);
    }

    virtual bool ShouldTerminate(const VotesStats& parent, const VotesStats& leftChild, const VotesStats& rightChild, double gain)
    {
     /*    std::cerr << "vote variance: " << parent.VoteVariance() << std::endl;
        std::cerr << "lvv: " << leftChild.VoteVariance() << std::endl;
        std::cerr << "rvv: " << rightChild.VoteVariance() << std::endl;
        std::cerr << "lsize: " << leftChild.Size() << std::endl;
        std::cerr << "rsize: " << rightChild.Size() << std::endl;
        std::cerr << "gain: " << gain << std::endl;*/

        return gain < 0.01 | (leftChild.Count() < 100) | (rightChild.Count() < 100);
    }

private:
    DepthFeatureFactory &factory_;
    unsigned char nClasses_;
};

#endif // HOUGHTRAININGCONTEXT_H
