#include "featureaccomulator.h"


bool FeatureAccomulator::addCurrentParameters(const DepthFeature &f, float threashold, double gain)
{
    if (accomulator_.size()>= count_){
        if (-gain < accomulator_.top().gain_){
            if(accomulator_.size() >= count_){
                //preserve the constant size of accomulator...
                accomulator_.pop();
            }

            //make gain negative, because we want to remove the smallest always
            accomulator_.push(FeatureGainType(f,threashold,-gain));

        }
    }
    else
    {
        accomulator_.push(FeatureGainType(f,threashold,-gain));
    }

    //collect all stats
    {
        gainStats_[last_]=gain;
        thrStats_[last_]=threashold;
        last_++;
    }

}
