#ifndef TRAININGCONTEXT_H
#define TRAININGCONTEXT_H

#include "Interfaces.h"
#include "depthfeature.h"
#include "classstats.h"

#include <iostream>

template <class F, class S>
class TrainingContext : public MicrosoftResearch::Cambridge::Sherwood::ITrainingContext<F,S>
{
public:
    TrainingContext(unsigned short nClasses,DepthFeatureFactory &factory):factory_(factory)
    {
        nClasses_ = nClasses;
    }

    virtual F GetRandomFeature(MicrosoftResearch::Cambridge::Sherwood::Random& random)
    {
        return factory_.getDepthFeature();
    }

    virtual S GetStatisticsAggregator()
    {
        return ClassStats(nClasses_);
    }


    virtual double ComputeInformationGain(const S& parent, const S& leftChild, const S& rightChild)
    {
        ClassStats &p = (ClassStats &)parent, &l = (ClassStats&)leftChild, &r = (ClassStats &)rightChild;

        double entropyBefore = p.Entropy();

        unsigned int nTotalSamples = p.SampleCount();

        if (nTotalSamples <= 1)
          return 0.0;

        double entropyAfter = ((l.SampleCount() / (double) nTotalSamples) * l.Entropy() + (r.SampleCount() / (double) nTotalSamples) * r.Entropy());

        return entropyBefore - entropyAfter;
    }

    virtual bool ShouldTerminate(const S& parent, const S& leftChild, const S& rightChild, double gain)
    {
        std::cerr << "gain: " << gain << std::endl;

        return gain < 0.01;
    }

private:

    unsigned short nClasses_;
    DepthFeatureFactory &factory_;

};



#endif // TRAININGCONTEXT_H
