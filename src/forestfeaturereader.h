#ifndef FORESTFEATUREREADER_H
#define FORESTFEATUREREADER_H

#include "Forest.h"
#include "depthfeature.h"
#include "regression/votesstatst.h"
#define VoteDim 3
typedef float VoteType;

typedef VotesStatsT<VoteType,VoteDim> Stats;

using namespace MicrosoftResearch::Cambridge::Sherwood;


class ForestFeatureReader
{
public:
    ForestFeatureReader(std::istream &input)
    {
        forest_ = Forest<DepthFeature, Stats>::Deserialize(input);
    }

    DepthFeature getFeature(int tree, int node)
    {
        return forest_->GetTree(tree).GetNode(node).Feature;
    }


private:

    std::auto_ptr<Forest<DepthFeature,Stats> > forest_;
};

#endif // FORESTFEATUREREADER_H
