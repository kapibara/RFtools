#include "featurepool.h"

#include "math.h"

DepthFeature FeaturePool::getFeature(unsigned int nodeIndex)
{
    std::cerr << "FeaturePool::getFeature" << std::endl;

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

    std::cerr << "node index: " << currentNode_ << " feature: " << feature << std::endl;
    std::cerr << "max  index: " << features_.size() << std::endl;

    DepthFeature d = features_[currentNode_][feature];

    std::cerr << "FeaturePool::getFeature ended" << std::endl;

    return d;
}


void FeaturePool::readFeatures()
{
    unsigned int nodeCount = std::pow((double)2,maxDecisionLevel_)-1;
    int currentNode;
    DepthFeature *f;
    features_.resize(nodeCount);

    for(unsigned int i=0; i< nodeCount; i++){
        in_.read((char *)&currentNode,sizeof(currentNode));
        std::cerr << "current node: " << currentNode << std::endl;
        f = new DepthFeature[fpnode_];
        for(int j=0; j<fpnode_; j++){
            f[j].Deserialize(in_);
        }
        features_[currentNode] = f;
    }
}
