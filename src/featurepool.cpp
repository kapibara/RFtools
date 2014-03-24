#include "featurepool.h"

#include "math.h"

DepthFeature FeaturePool::getDepthFeature(MicrosoftResearch::Cambridge::Sherwood::Random &random)
{
    int feature = currentFeature_;
    currentFeature_++;

    std::cerr << "node index: " << currentNode_ << " feature: " << feature << std::endl;
    std::cerr << "max  index: " << features_.size() << std::endl;

    DepthFeature d = features_[currentNode_][feature];

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
