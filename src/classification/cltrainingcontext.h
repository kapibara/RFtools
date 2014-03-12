#ifndef TRAININGCONTEXT_H
#define TRAININGCONTEXT_H

#include "Interfaces.h"
#include "depthfeature.h"
#include "classstats.h"

#include <iostream>

class ClTrainingContext : public MicrosoftResearch::Cambridge::Sherwood::ITrainingContext<DepthFeature,ClassStats>
{
public:
    ClTrainingContext(unsigned short nClasses,DepthFeatureFactory &factory):factory_(factory)
    {
        nClasses_ = nClasses;
    }

    virtual DepthFeature GetRandomFeature(MicrosoftResearch::Cambridge::Sherwood::Random& random)
    {
        return factory_.getDepthFeature(random);
    }

    virtual ClassStats GetStatisticsAggregator()
    {
        return ClassStats(nClasses_);
    }


    virtual double ComputeInformationGain(ClassStats& p, ClassStats& l, ClassStats& r)
    {

        double entropyBefore = p.Entropy();

        unsigned int nTotalSamples = l.SampleCount()+r.SampleCount();

        if( (l.SampleCount()+r.SampleCount())!= p.SampleCount()){
            std::cerr << "error: different sample count values" << std::endl;
        }

        if (nTotalSamples <= 1)
          return 0.0;

        double entropyAfter = ((l.SampleCount() / (double) nTotalSamples) * l.Entropy() + (r.SampleCount() / (double) nTotalSamples) * r.Entropy());

        return entropyBefore - entropyAfter;
    }

    virtual bool ShouldTerminate(const ClassStats& parent, const ClassStats& leftChild, const ClassStats& rightChild, double gain)
    {
        bool result = (gain < 0.01) | (leftChild.SampleCount()<200)|(rightChild.SampleCount()<200);
/*        if (result){
            std::cerr << "gain: " << gain << std::endl;
            std::cerr << "left child stats: " << leftChild.SampleCount() << std::endl;
            std::cerr << "right child stats:" << rightChild.SampleCount() << std::endl;
        }*/
        return result;
    }

private:

    unsigned short nClasses_;
    DepthFeatureFactory &factory_;

};



#endif // TRAININGCONTEXT_H
