#include "stubtrainingcontext.h"

DepthFeature StubTrainingContext::GetRandomFeature(MicrosoftResearch::Cambridge::Sherwood::Random& random)
{
    DepthFeature f = factory_.getDepthFeature(random);

    f.Serialize(out_);

    return f;
}
