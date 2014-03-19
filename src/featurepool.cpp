#include "featurepool.h"

#include "math.h"

DepthFeature FeaturePool::getFeature(unsigned int nodeIndex)
{
    if(nodeIndex != currentNode_){
        currentNode_ = nodeIndex;
        if (currentFeature_ != fpnode_){
            std::cerr << "not all features from the node " << nodeIndex << " are checked" << std::endl;
            std::cerr << "last feature: " << currentFeature_
                      << "feature count: " << fpnode_
                      << std::endl;
        }
        currentFeature_ = 0;
    }

    int feature = currentFeature_;
    currentFeature_++;

    return features_[currentNode_][feature];
}


void FeaturePool::readFeatures()
{
    unsigned int nodeCount = std::pow((double)2,maxDecisionLevel_)-1;
    int currentNode;
    DepthFeature *f;
    features_.resize(nodeCount);

    for(unsigned int i=0; i< nodeCount; i++){
        in_.read((char *)&currentNode,sizeof(currentNode));
        f = new DepthFeature[fpnode_];
        for(int j=0; j<fpnode_; j++){
            f[j].Deserialize(in_);
        }
        features_[currentNode] = f;
    }
}
