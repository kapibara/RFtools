#include "pooledhoughtrainingcontext.h"

DepthFeature PooledHoughTrainingContext::GetRandomFeature(MicrosoftResearch::Cambridge::Sherwood::Random& random)
{
    return pool_.getFeature(currentNode_);
}
