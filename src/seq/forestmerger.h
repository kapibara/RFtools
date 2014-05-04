#ifndef FORESTMERGER_H
#define FORESTMERGER_H

#include "Forest.h"
#include "Node.h"
#include "TrainingParameters.h"

#include "depthfeature.h"
#include "regression/aggregatedleafs.h"
#include <auto_ptr.h>

using namespace MicrosoftResearch::Cambridge::Sherwood;

template<class DepthFeature, class Stats, typename ElemType, int S>
class AggregatedLeafsMerger
{

    typedef AggregatedLeafs<DepthFeature,Stats,ElemType,S> Leafs;
public:
    AggregatedLeafsMerger()
    {

    }

    void SetWeights(double w1, double w2)
    {
        w1_ = w1; w2_ = w2;
    }

    Leafs mergeAggregatedLeafs(const Leafs &al1,const Leafs &al2)
    {
        Leafs result;

        for (int t = 0; t< al1.TreeCount(); t++)
        {
            al1.begin(t);
            al2.begin(t);

            for(int n = 0; n<al1.MaxNodesCount())
        }

        return forest;
    }

private:
    double w1_;
    double w2_;


};

template< class Stats>
class ForestMerger
{
    typedef DepthFeature F;
    typedef Stats S;

public:
    ForestMerger()
    {

    }

    void SetWeights(double w1, double w2)
    {
        w1_ = w1; w2_ = w2;
    }

    std::auto_ptr<Forest<DepthFeature,Stats> > mergeForests(const Forest<DepthFeature,Stats> &f1,const Forest<DepthFeature,Stats> &f2, TrainingParameters param)
    {
        std::auto_ptr<Forest<F,S> > forest = std::auto_ptr<Forest<F,S> >(new Forest<F,S>());

        for (int t = 0; t < f1.TreeCount(); t++)
        {
            std::auto_ptr<Tree<F, S> > tree = mergeTrees(f1.GetTree(t),f2.GetTree(),param);
            forest->AddTree(tree);
        }

        return forest;
    }

private:
    //stats in fact does not matter-> merged later

    std::auto_ptr<Tree<F,S> > mergeTrees(const Tree<F,S> &t1,const Tree<F,S> &t2, TrainingParameters param)
    {
        std::auto_ptr<Tree<F, S> > tree = std::auto_ptr<Tree<F, S> >(new Tree<F,S>(param.MaxDecisionLevels));

        for(int i=0; i<tree->NodeCount(); i++){
            if(t1->GetNode(i).isSplit() && t2->GetNode(i).isSplit())
            {
                F f1 = t1->GetNode(i).Feature;
                F f2 = t2->GetNode(i).Feature;
                float thr1 =  t1->GetNode(i).Threshold;
                float thr2 =  t2->GetNode(i).Threshold;

                tree->GetNode(i).InitializeSplit(linearCombination(f1, f2, w1, w2),thr1*w1_ + thr2*w2_,Stats());
            }
            if(t1->GetNode(i).isSplit() & ~t2->GetNode(i).isSplit() ){
                tree->GetNode(i).InitializeSplit(f1,thr1,Stats());
            }
            if(t2->GetNode(i).isSplit() & ~t1->GetNode(i).isSplit() ){
                tree->GetNode(i).InitializeSplit(f2,thr2,Stats());
            }
            if((t1->GetNode(i).isLeaf() & t2->GetNode(i).isLeaf()) ||
               (t1->GetNode(i).isLeaf() & t2->GetNode(i).isNull()) ||
               (t2->GetNode(i).isLeaf() & t1->GetNode(i).isNull())){
                tree->GetNode(i).InitializeLeaf(Stats());
            }
        }
    }

    double w1_;
    double w2_;
};

#endif // FORESTMERGER_H