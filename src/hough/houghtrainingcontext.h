#ifndef HOUGHTRAININGCONTEXT_H
#define HOUGHTRAININGCONTEXT_H

#include "Interfaces.h"

#include "depthfeature.h"

template <class F, class S>
class HoughTrainingContext: public MicrosoftResearch::Cambridge::Sherwood::ITrainingContext<F,S>
{
public:
    HoughTrainingContext(DepthFeatureFactory &factory):factory_(factory)
    {
        nClasses_ = nClasses;
    }

    virtual F GetRandomFeature(MicrosoftResearch::Cambridge::Sherwood::Random& random)
    {
        return factory_.getDepthFeature(random);
    }

    virtual S GetStatisticsAggregator()
    {
        return 0;
    }


    virtual double ComputeInformationGain(const S& parent, const S& leftChild, const S& rightChild)
    {
// to implement
    }

    virtual bool ShouldTerminate(const S& parent, const S& leftChild, const S& rightChild, double gain)
    {
        return gain < 0.01;
    }

private:
    DepthFeatureFactory &factory_;
};

#endif // HOUGHTRAININGCONTEXT_H
