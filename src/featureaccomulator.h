#ifndef FEATUREACCOMULATOR_H
#define FEATUREACCOMULATOR_H

#include "depthfeature.h"

#include <istream>
#include <vector>
#include <queue>

class FeatureAccomulator
{
public:
    FeatureAccomulator(std::ostream &out,int statsSize,int count = 10):out_(out)
    {
        count_ = count;
        nodeIndex_ = 0;
        gainStats_.resize(statsSize);
        thrStats_.resize(statsSize);
        last_ = 0;
        out_.write((const char *)&count_,sizeof(count_));
    }

    void setCurrentNode(int index)
    {
        if(accomulator_.size()>0){
            serializePQ();
        }
        nodeIndex_ = index;
    }

    bool addCurrentParameters(const DepthFeature &f, float threashold, double gain);


private:

    struct FeatureGainType{
        DepthFeature f_;
        float threashold_;
        double gain_;

        FeatureGainType(const DepthFeature &f, float threashold, double gain):
            f_(f),
            threashold_(threashold),
            gain_(gain)
        {

        }

        bool operator<(const FeatureGainType &other) const{
            return gain_ < other.gain_;
        }
    };

    void serializePQ(){

        out_.write((const char *)&nodeIndex_,sizeof(nodeIndex_));
        while(!accomulator_.empty()){
            accomulator_.top().f_.Serialize(out_);
            out_.write((const char *)&(accomulator_.top().threashold_),sizeof(float));
            out_.write((const char *)&(accomulator_.top().gain_),sizeof(double));
            accomulator_.pop();
        }

        {
            int size = gainStats_.size();
            out_.write((const char *)&(size),sizeof(size));
            for(int i=0; i< size; i++){
                out_.write((const char *)&(gainStats_[i]),sizeof(double));
            }
            size = thrStats_.size();
            out_.write((const char *)&(size),sizeof(size));
            for(int i=0; i< size; i++){
                out_.write((const char *)&(thrStats_[i]),sizeof(float));
            }
            last_ = 0;
        }
    }

    int count_;
    int nodeIndex_;
    std::ostream &out_;
    std::priority_queue<FeatureGainType> accomulator_;

    std::vector<double> gainStats_;
    std::vector<float> thrStats_;
    int last_;
};

#endif // FEATUREACCOMULATOR_H
