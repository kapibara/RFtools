#ifndef HOUGHTRAININGCONTEXT_H
#define HOUGHTRAININGCONTEXT_H

#include "Interfaces.h"

#include "depthfeature.h"
#include "hough/votesstats.h"
#include "featureaccomulator.h"

#include <ostream>


template <class Stats, class FeatureFactory>
class RegTrainingContext: public MicrosoftResearch::Cambridge::Sherwood::ITrainingContext<DepthFeature,Stats>
{
public:
    enum GainType {Variance, DiffEntropy};

    RegTrainingContext(unsigned char nClasses, FeatureFactory &factory,const std::string &gtype = "variance", unsigned int thr2 = 300*300):factory_(factory)
    {
        nClasses_ = nClasses;
        accomulator_ = 0;
        type_ = RegTrainingContext::Variance;
        setGainType(gtype);
        thr2_ = thr2;
    }

    DepthFeature GetRandomFeature(MicrosoftResearch::Cambridge::Sherwood::Random& random)
    {
        return factory_.getDepthFeature(random);
    }

    void setFeatureAccomulator(FeatureAccomulator *ptr){
        accomulator_ = ptr;
    }

    Stats GetStatisticsAggregator()
    {
        return Stats(nClasses_,thr2_);
    }

    void setGainType(GainType type){
        type_ = type;
    }

    bool setGainType(const std::string &type){
        if (type.compare("variance")){
            setGainType(RegTrainingContext::Variance);
            return true;
        }

        if (type.compare("diffentropy")){
            setGainType(RegTrainingContext::DiffEntropy);
            return true;
        }

        std::cerr << "WARNING: setGainType failed - unrecognized gain type" << std::endl;

        return false;
    }

    double ComputeInformationGain(Stats& parent, Stats& leftChild, Stats& rightChild)
    {
        if(parent.Count() <=1){
            return 0;
        }

        double w1 = ((double)leftChild.Count())/parent.Count();
        double w2 = ((double)rightChild.Count())/parent.Count();
        double lvv,rvv,pvv;

        if (type_ == RegTrainingContext::Variance){
            lvv = leftChild.VoteVariance();
            rvv = rightChild.VoteVariance();
            pvv = parent.VoteVariance();

        }
        else{
            lvv = leftChild.LogEntropy();
            rvv = rightChild.LogEntropy();
            pvv = parent.LogEntropy();
        }

        return ((pvv - w1*lvv) - w2*rvv);
//        return ((pvv - lvv) - rvv);
    }

    bool ShouldTerminate(const Stats& parent, const Stats& leftChild, const Stats& rightChild, double gain)
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
    GainType type_;

    int currentNode_;
    FeatureAccomulator *accomulator_;

    unsigned char nClasses_;
    unsigned int thr2_;
};

#endif // HOUGHTRAININGCONTEXT_H
