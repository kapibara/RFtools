#ifndef POOLEDHOUGHTRAININGCONTEXT_H
#define POOLEDHOUGHTRAININGCONTEXT_H

#include "featurepool.h"
#include "votesstats.h"
#include "featureaccomulator.h"

namespace MRS = MicrosoftResearch::Cambridge::Sherwood;


class PooledHoughTrainingContext: public MRS::ITrainingContext<DepthFeature,VotesStats>
{
public:

    PooledHoughTrainingContext(unsigned char nClasses, FeaturePool &pool):pool_(pool)
    {
        nClasses_ = nClasses;
        currentNode_ = 0;
        accomulator_ = 0;
    }

    DepthFeature GetRandomFeature(MicrosoftResearch::Cambridge::Sherwood::Random& random);

    void setFeatureAccomulator(FeatureAccomulator *ptr){
        accomulator_ = ptr;
    }

    VotesStats GetStatisticsAggregator()
    {
        return VotesStats(nClasses_);
    }

    double ComputeInformationGain(VotesStats& parent, VotesStats& leftChild, VotesStats& rightChild)
    {
        double lvv = leftChild.VoteVariance();
        double rvv = rightChild.VoteVariance();


        if(parent.Count() <=1){
            return 0;
        }
        /*computing parent.VoteVariance() is not optimal*/

        return ((parent.VoteVariance() - lvv) - rvv);
    }

    bool ShouldTerminate(const VotesStats& parent, const VotesStats& leftChild, const VotesStats& rightChild, double gain)
    {
     /*    std::cerr << "vote variance: " << parent.VoteVariance() << std::endl;
        std::cerr << "lvv: " << leftChild.VoteVariance() << std::endl;
        std::cerr << "rvv: " << rightChild.VoteVariance() << std::endl;
        std::cerr << "lsize: " << leftChild.Size() << std::endl;
        std::cerr << "rsize: " << rightChild.Size() << std::endl;
        std::cerr << "gain: " << gain << std::endl;*/

        return gain < 0.01 | (leftChild.Count() < 100) | (rightChild.Count() < 100);
    }

    virtual void setCurrentNode(int nodeIndex) {
        currentNode_ = nodeIndex;
        if(accomulator_!=0){
            accomulator_->setCurrentNode(nodeIndex);
        }
    }

    virtual void collectStats(const DepthFeature&feature, float threashold, double gain)
    {
        if(accomulator_!=0){
            accomulator_->addCurrentParameters(feature,threashold,gain);
        }

    }

private:
    FeaturePool &pool_;
    FeatureAccomulator *accomulator_;
    unsigned char nClasses_;
    int currentNode_;
};

#endif // POOLEDHOUGHTRAININGCONTEXT_H
