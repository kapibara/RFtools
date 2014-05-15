#ifndef AGGREGATEDLEAFS_H
#define AGGREGATEDLEAFS_H

#include "Forest.h"

#include "votesaggregator.h"

#include "meanshift.h"

#include <vector>

using namespace MicrosoftResearch::Cambridge::Sherwood;

template<class DepthFeature, class Stats, typename ElemType, int S>
class AggregatedLeafs
{
public:
    typedef typename std::vector<VotesAggregator<ElemType,S> >::iterator iterator;
    typedef typename std::vector<VotesAggregator<ElemType,S> >::const_iterator const_iterator;

    AggregatedLeafs()
    {
        varThr_ = -1;
        wThr_ = -1;
        sizeThr_ = -1;
    }

    AggregatedLeafs(int tcount, int maxNodes, int votesCount)
    {
        varThr_ = -1;
        wThr_ = -1;
        sizeThr_ = -1;
        maxNodes_ = maxNodes;
        votesCount_= votesCount;
        leafs_.resize(tcount);
        leafcount_.assign(tcount,0);
        VotesAggregator<ElemType,S> tmp(votesCount);

        for(int i=0; i<tcount; i++){
            leafs_[i] = new std::vector<VotesAggregator<ElemType,S> >();
            leafs_[i]->assign(maxNodes,tmp);
        }

    }

    iterator begin(int elemIdx)
    {
        return leafs_[elemIdx]->begin();
    }

    iterator end(int elemIdx)
    {
        return leafs_[elemIdx]->begin();
    }

    const_iterator begin(int elemIdx) const
    {
        return leafs_[elemIdx]->begin();
    }

    const_iterator end(int elemIdx) const
    {
        return leafs_[elemIdx]->begin();
    }

    void SetVarThreashold(double thr)
    {
        varThr_ = thr;
    }
    void SetSmallWeightsThreashold(double thr)
    {
        wThr_ = thr;
    }
    void SetNodeSizeThreashold(int thr)
    {
        sizeThr_ = thr;
    }

    int TreeCount() const
    {
        return leafs_.size();
    }

    int LeafCount(int elemIdx) const
    {

        return leafcount_[elemIdx];
    }

    int &operator[](int elemIdx)
    {

        return leafcount_[elemIdx];
    }

    int operator[](int elemIdx) const
    {

        return leafcount_[elemIdx];
    }

    int MaxNodesCount() const
    {
        return maxNodes_;
    }

    int VotesCount() const
    {
        return votesCount_;
    }

    void Denormalize(const std::vector<float> &mean, const std::vector<float> &var)
    {
        for(int t = 0 ; t < leafs_.size(); t++)
        {
            for(int j=0; j< leafs_[t]->size(); j++){
                get(t,j).Denormalize(mean,var);
            }
        }
    }

    void Normalize(const std::vector<float> &mean, const std::vector<float> &var)
    {
        for(int t = 0 ; t < leafs_.size(); t++)
        {
            for(int j=0; j< leafs_[t]->size(); j++){
                get(t,j).Normalize(mean,var);
            }
        }
    }

    void Build(Forest<DepthFeature,Stats> &forest, mean_shift::MeanShift &mshift, int votesCount){

        votesCount_ = votesCount;
        std::vector<VotesAggregator<ElemType,S> > *current;
        VotesAggregator<ElemType,S> tmp(votesCount_);
        maxNodes_ = forest.GetTree(0).NodeCount(); //all trees have the same number of nodes

        leafcount_.assign(forest.TreeCount(),0);

        for(int t = 0 ; t < forest.TreeCount(); t++)
        {
            current = new std::vector<VotesAggregator<ElemType,S> >();
            current->assign(forest.GetTree(t).NodeCount(),tmp);
            leafcount_[t] = aggregateTree(forest.GetTree(t), *current, mshift);
            leafs_.push_back(current);
        }
    }

    void Serialize(std::ostream &out)
    {
        int treeCount = leafs_.size();
        out.write((const char *)&treeCount, sizeof(treeCount));
        out.write((const char *)&maxNodes_, sizeof(maxNodes_));
        out.write((const char *)&votesCount_, sizeof(votesCount_));
        int nodeCount;

        for(int t=0; t<leafs_.size(); t++){
            nodeCount = leafcount_[t];

            out.write((const char *)&nodeCount, sizeof(nodeCount));

            for(int i=0; i<leafs_[t]->size(); i++){
                if(!leafs_[t]->operator[](i).IsEmpty()){
                //write leafs index
                    out.write((const char *)&i,sizeof(i));
                    leafs_[t]->operator[](i).Serialize(out);
                }
            }
        }
    }

    void Serialize(std::ostream &out,Forest<DepthFeature,Stats> &forest)
    {
        int treeCount = leafs_.size();
        out.write((const char *)&treeCount, sizeof(treeCount));
        out.write((const char *)&maxNodes_, sizeof(maxNodes_));
        out.write((const char *)&votesCount_, sizeof(votesCount_));
        int nodeCount;

        for(int t=0; t<leafs_.size(); t++){
            nodeCount = leafcount_[t];
            out.write((const char *)&nodeCount, sizeof(nodeCount));

            for(int i=0; i<leafs_[t]->size(); i++){
                if(!leafs_[t]->operator[](i).IsEmpty()){
                //write leafs index

                    out.write((const char *)&i,sizeof(i));
                    leafs_[t]->operator[](i).Serialize(out);
                    forest.GetTree(t).GetNode(i).TrainingDataStatistics.Serialize(out);
                }
            }
        }
    }

    void Deserialize(std::istream &in, bool foreststats = false)
    {
        int treeCount;
        in.read(( char *)&treeCount, sizeof(treeCount));
        in.read(( char *)&maxNodes_, sizeof(maxNodes_));
        in.read(( char *)&votesCount_, sizeof(votesCount_));
        leafcount_.resize(treeCount);

        std::vector<VotesAggregator<ElemType,S> > *current;
        VotesAggregator<ElemType,S> tmp(votesCount_);

        int nodeIdx;
        for(int t=0; t<treeCount; t++){
            in.read((char *)&(leafcount_[t]), sizeof(leafcount_[t]));

            current = new std::vector<VotesAggregator<ElemType,S> >();
            current->assign(maxNodes_,tmp);
            for(int i=0; i<leafcount_[t]; i++){
                in.read((char *)&nodeIdx,sizeof(nodeIdx));
                current->operator [](nodeIdx).Deserialize(in);
                if(foreststats){
                    tmp.Deserialize(in);
                }
            }
            leafs_.push_back(current);
        }
    }

    VotesAggregator<ElemType,S> &get(int t, unsigned int index)
    {


        return (leafs_[t])->operator[](index);
    }


private:

    int aggregateTree(Tree<DepthFeature,Stats> &tree, std::vector<VotesAggregator<ElemType,S> > &aggLeafs, mean_shift::MeanShift &mshift)
    {
        int leafcount = 0;
        for(int i=0; i< tree.NodeCount();i++){
            if(tree.GetNode(i).IsLeaf()){
                if(varThr_< 0 ||
                    tree.GetNode(i).TrainingDataStatistics.VoteVariance() < varThr_){
                    if (sizeThr_ <0 ||
                        tree.GetNode(i).TrainingDataStatistics.Count() < sizeThr_){
                        leafcount++;
                        aggLeafs[i].AggregateVotes(tree.GetNode(i).TrainingDataStatistics,mshift);
                        aggLeafs[i].FilterSmallWeights(wThr_);
                    }else{
                        std::cerr << "threshold the size: node " << i << std::endl;
                    }
                }else{
                    std::cerr << "threshold the variance: node " << i << std::endl;
                }

            }
        }

        return leafcount;
    }

    std::vector< std::vector<VotesAggregator<ElemType,S> > *> leafs_;
    std::vector< int> leafcount_;
    int maxNodes_;
    int votesCount_;
    double varThr_,wThr_;
    int sizeThr_;
};


#endif // AGGREGATEDLEAFS_H
