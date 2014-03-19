#include "houghtrainingcontext.h"

DepthFeature HoughTrainingContext::GetRandomFeature(MicrosoftResearch::Cambridge::Sherwood::Random& random)
{
    return factory_.getDepthFeature(random);
}
