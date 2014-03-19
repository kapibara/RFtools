#include "featurepool.h"

#include "math.h"

DepthFeature FeaturePool::getFeature(unsigned int nodeIndex)
{
    if(nodeIndex != currentNode_){
        currentNode_ = nodeIndex;
        if (currentNode_ != fpnode_){
            std::cerr << "not all features from the node " << nodeIndex << " are checked" << std::endl;
        }
        currentFeature_ = 0;
    }

    int feature = currentNode_;
    currentNode_++;

    return features_[currentNode_][feature];
}


void FeaturePool::readFeatures()
{
    unsigned int nodeCount = std::pow((double)2,maxDecisionLevel_)-1;
    DepthFeature *f;

    for(unsigned int i=0; i< nodeCount; i++){
        f = new DepthFeature[fpnode_];
        for(int j=0; j<fpnode_; j++){
            f[j].Deserialize(in_);
        }
        features_.push_back(f);
    }
}
