#ifndef STUBHOUGHTRAININGCONTEXT_H
#define STUBHOUGHTRAININGCONTEXT_H

#include "stubstats.h"
#include "depthfeature.h"
#include "Interfaces.h"
#include "TrainingParameters.h"

namespace MRS = MicrosoftResearch::Cambridge::Sherwood;

class StubTrainingContext:public  MRS::ITrainingContext<DepthFeature,StubStats>
{
public:
    StubTrainingContext(DepthFeatureFactory &factory, const MRS::TrainingParameters &params, std::ostream &output): out_(output), factory_(factory)
    {

            output.write((const char *)&(params.NumberOfCandidateFeatures), sizeof(params.NumberOfCandidateFeatures));
            output.write((const char *)&(params.MaxDecisionLevels), sizeof(params.MaxDecisionLevels));

    }

    DepthFeature GetRandomFeature(MicrosoftResearch::Cambridge::Sherwood::Random& random);

    StubStats GetStatisticsAggregator()
    {
        return StubStats();
    }

    double ComputeInformationGain(StubStats& parent, StubStats& leftChild, StubStats& rightChild)
    {
        return 1;
    }

    bool ShouldTerminate(const StubStats& parent, const StubStats& leftChild, const StubStats& rightChild, double gain)
    {
        //always return false;
        return false;
    }

private:
    DepthFeatureFactory &factory_;
    std::ostream &out_;
};

#endif // STUBHOUGHTRAININGCONTEXT_H
