#ifndef FORESTMERGER_H
#define FORESTMERGER_H

#include "Forest.h"
#include "Node.h"
#include "TrainingParameters.h"

#include "depthfeature.h"
#include "regression/aggregatedleafs.h"
#include <auto_ptr.h>

using namespace MicrosoftResearch::Cambridge::Sherwood;

template<typename ElemType, int S>
class AggregatedLeafsMerger
{
    typedef AggregatedLeafs<DepthFeature,VotesStatsT<ElemType,S> ,ElemType, S> Leafs;

public:
    AggregatedLeafsMerger()
    {
        w1_ = 1; w2_ = 0;
    }

    void SetWeights(double w1, double w2)
    {
        w1_ = w1; w2_ = w2;
    }

    std::auto_ptr<Leafs> mergeAggregatedLeafs(const Leafs &al1,const Leafs &al2)
    {
        std::auto_ptr<Leafs> result = std::auto_ptr<Leafs>(new Leafs(al1.TreeCount(),al1.MaxNodesCount(),al1.VotesCount()));
        float w1 = w1_,w2 = w2_;

        for (int t = 0; t< al1.TreeCount(); t++)
        {
            typename Leafs::const_iterator i1 = al1.begin(t);
            typename Leafs::const_iterator i2 = al2.begin(t);

            for(int n = 0; n<al1.MaxNodesCount(); n++){
                w1 = w1_*i1->OriCount(0)/(w1_*i1->OriCount(0) + w2_*i2->OriCount(0));//cheat
                w2 = w2_*i2->OriCount(0)/(w1_*i1->OriCount(0) + w2_*i2->OriCount(0));//cheat

                if(!i1->IsEmpty()){
                    result->get(t,n).AddVotes((*i1),w1);
                    result->operator [](t)++;
                }
                if(!i2->IsEmpty()){
                    result->get(t,n).AddVotes((*i2),w2);
                    result->operator [](t)++;
                }
                if(!i1->IsEmpty() & !i2->IsEmpty() ){
                    result->operator [](t)--;
                }
                i1++;
                i2++;
            }
        }


        return result;
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

    std::auto_ptr<Forest<DepthFeature,Stats> > mergeForests(const Forest<DepthFeature,Stats> &f1,const Forest<DepthFeature,Stats> &f2)
    {
        std::auto_ptr<Forest<F,S> > forest = std::auto_ptr<Forest<F,S> >(new Forest<F,S>());

        for (int t = 0; t < f1.TreeCount(); t++)
        {
            std::auto_ptr<Tree<F, S> > tree = mergeTrees(f1.GetTree(t),f2.GetTree(t));
            forest->AddTree(tree);
        }

        return forest;
    }

private:
    //stats in fact does not matter-> merged later

    std::auto_ptr<Tree<F,S> > mergeTrees(const Tree<F,S> &t1,const Tree<F,S> &t2)
    {
        std::auto_ptr<Tree<F, S> > tree = std::auto_ptr<Tree<F, S> >(new Tree<F,S>(t1.DecisionLevels()));

        for(int i=0; i< tree->NodeCount(); i++){

            F f1 = t1.GetNode(i).Feature;
            F f2 = t2.GetNode(i).Feature;
            float thr1 =  t1.GetNode(i).Threshold;
            float thr2 =  t2.GetNode(i).Threshold;

            if(t1.GetNode(i).IsSplit() && t2.GetNode(i).IsSplit())
            {
                tree->GetNode(i).InitializeSplit(linearCombination(f1, f2, w1_, w2_),thr1*w1_ + thr2*w2_,Stats());
            }
            if(t1.GetNode(i).IsSplit() & ~t2.GetNode(i).IsSplit() ){
                tree->GetNode(i).InitializeSplit(f1,thr1,Stats());
            }
            if(t2.GetNode(i).IsSplit() & ~t1.GetNode(i).IsSplit() ){
                tree->GetNode(i).InitializeSplit(f2,thr2,Stats());
            }
            if((t1.GetNode(i).IsLeaf() & t2.GetNode(i).IsLeaf()) ||
               (t1.GetNode(i).IsLeaf() & t2.GetNode(i).IsNull()) ||
               (t2.GetNode(i).IsLeaf() & t1.GetNode(i).IsNull())){
                tree->GetNode(i).InitializeLeaf(Stats());
            }
        }

        return tree;
    }

    double w1_;
    double w2_;
};

#endif // FORESTMERGER_H
