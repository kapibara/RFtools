#ifndef HOUGHTRAININGCONTEXT_H
#define HOUGHTRAININGCONTEXT_H

#include "Interfaces.h"

#include "depthfeature.h"
#include "hough/votesstats.h"
#include "featureaccomulator.h"

#include <ostream>


template <class FeatureFactory>
class HoughTrainingContext: public MicrosoftResearch::Cambridge::Sherwood::ITrainingContext<DepthFeature,VotesStats>
{
public:
    HoughTrainingContext(unsigned char nClasses,FeatureFactory &factory,unsigned int thr2 = 300*300):factory_(factory)
    {
        nClasses_ = nClasses;
        accomulator_ = 0;
        thr2_ = thr2;
    }

    DepthFeature GetRandomFeature(MicrosoftResearch::Cambridge::Sherwood::Random& random)
    {
        return factory_.getDepthFeature(random);
    }

    void setFeatureAccomulator(FeatureAccomulator *ptr){
        accomulator_ = ptr;
    }
    VotesStats GetStatisticsAggregator()
    {
        return VotesStats(nClasses_,thr2_);
    }

    double ComputeInformationGain(VotesStats& parent, VotesStats& leftChild, VotesStats& rightChild)
    {
        double lvv = leftChild.VoteVariance();
        double rvv = rightChild.VoteVariance();


        if(parent.Count() <=1){
            return 0;
        }


        return ((parent.VoteVariance() - lvv) - rvv);
    }

    bool ShouldTerminate(const VotesStats& parent, const VotesStats& leftChild, const VotesStats& rightChild, double gain)
    {
        return gain < 0.01 | (leftChild.Count() < 100) | (rightChild.Count() < 100);
    }

    void setCurrentNode(int nodeIndex)
    {
        currentNode_ = nodeIndex;
        if(accomulator_!=0){
            accomulator_->setCurrentNode(nodeIndex);
        }
        factory_.setCurrentNode(nodeIndex);
    }

    void collectStats(const DepthFeature &feature, float threashold, double gain)
    {
        if(accomulator_!=0){
            accomulator_->addCurrentParameters(feature,threashold,gain);
        }
    }


private:
    FeatureFactory &factory_;

    int currentNode_;
    FeatureAccomulator *accomulator_;

    unsigned char nClasses_;
    unsigned int thr2_;
};

#endif // HOUGHTRAININGCONTEXT_H
