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
    HoughTrainingContext(unsigned char nClasses,FeatureFactory &factory):factory_(factory)
    {
        nClasses_ = nClasses;
        accomulator_ = 0;
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
        std::cerr << "gain: " << gain << std::endl;
        if (out_!=0){
            out_->write((const char *)&currentNode_, sizeof(currentNode_));
            out_->write((const char *)&gain, sizeof(gain));
        }*/

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

//    std::ostream *out_;

    unsigned char nClasses_;
};

#endif // HOUGHTRAININGCONTEXT_H
