#ifndef FEATUREPOOLTRAININGCONTEXT_H
#define FEATUREPOOLTRAININGCONTEXT_H

#include "Interfaces.h"
#include "depthfeature.h"
#include "arraylist.h"

#include <istream>
#include <vector>

class FeaturePool
{
public:
    FeaturePool(std::istream &input): in_(input)
    {
        in_.read((char *)&fpnode_,sizeof(fpnode_));
        in_.read((char *)&maxDecisionLevel_, sizeof(maxDecisionLevel_));
        readFeatures();
        currentNode_ = 0;
        currentFeature_ = 0;
    }

    int featuresPerNode(){
        return fpnode_;
    }

    DepthFeature getFeature(unsigned int nodeIndex);

private:

    void readFeatures();

    std::istream &in_;
    int fpnode_;
    int maxDecisionLevel_;
    int currentNode_;
    int currentFeature_;
    std::vector<DepthFeature *> features_;
};


#endif // FEATUREPOOLTRAININGCONTEXT_H
