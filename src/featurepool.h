#ifndef FEATUREPOOLTRAININGCONTEXT_H
#define FEATUREPOOLTRAININGCONTEXT_H

#include "Interfaces.h"
#include "depthfeature.h"
#include "arraylist.h"

#include <istream>

class FeaturePool
{
public:
    FeaturePool(std::istream &input): in_(input)
    {
        input.read((char *)fpnode_,sizeof(fpnode_));
        input.read((char *)maxDecisionLevel_, sizeof(maxDecisionLevel_));
        readFeatures();
        currentNode_ = 0;
        currentFeature_ = 0;
    }

    DepthFeature getFeature(unsigned int nodeIndex);

private:

    void readFeatures();

    std::istream &in_;
    int fpnode_;
    int maxDecisionLevel_;
    int currentNode_;
    int currentFeature_;
    ArrayList<DepthFeature *> features_;
};


#endif // FEATUREPOOLTRAININGCONTEXT_H
