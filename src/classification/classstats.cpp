#include "classstats.h"

#include "depthdb.h"

#include <iostream>
#include <cstring>

ClassStats::ClassStats(unsigned short clCount)
{
    binCount_ = clCount;
    if (clCount>0)
        bins_ = new bintype[clCount];
    else
        bins_ = 0;

    Clear();
}

ClassStats::ClassStats(const ClassStats &obj)
{

    binCount_ = obj.binCount_;

    if (binCount_>0)
        bins_  = new bintype[binCount_];
    else
        bins_ = 0;

    sampleCount_ = obj.sampleCount_;

    memcpy(bins_, obj.bins_, sizeof(bintype)*binCount_);
}

ClassStats::~ClassStats()
{

    if (bins_!=0)
        delete [] bins_;


}

unsigned char ClassStats::ClassDecision() const
{

    if(binCount_>0){
        unsigned short idx = 0;

        for (int b = 1; b < binCount_; b++){
            if (bins_[b] >  bins_[idx]){
                idx = b;
            }
        }

        return idx;
    }else{
        return -1;
    }
}

bool ClassStats::Serialize(std::ostream &stream) const{

    stream.write((const char *)(&binCount_),sizeof(binCount_));
    stream.write((const char *)bins_,sizeof(bintype)*binCount_);

    return true;
}

bool ClassStats::Deserialize(std::istream &stream){

    unsigned char tmp;
    stream.read((char *)(&tmp),sizeof(binCount_));

    if(binCount_ != tmp){
        if(binCount_>0){
            delete [] bins_;
        }
        bins_ = new bintype[tmp];
        binCount_ = tmp;
    }
    stream.read((char *)bins_,sizeof(bintype)*binCount_);

    sampleCount_=0;
    for(int i=0; i<binCount_; i++){
        sampleCount_+=bins_[i];
    }

    return true;
}

bool ClassStats::SerializeChar(std::ostream &stream) const{
    std::ostringstream ss;

    for(int i=0; i<binCount_; i++){
        ss << bins_[i] << ";";
    }
    stream << ss.str() << std::endl;
    return true;
}

ClassStats & ClassStats::operator=(const ClassStats & obj){


    if (binCount_ != obj.binCount_){
        binCount_ = obj.binCount_;
        if (bins_!=0){
            delete [] bins_;
        }
        bins_ = new bintype[binCount_];
    }
    memcpy(bins_, obj.bins_, sizeof(bintype)*binCount_);

    sampleCount_ = obj.sampleCount_;

    return *this;
}

void ClassStats::Clear()
{
  for (int b = 0; b < binCount_; b++)
    bins_[b] = 0;
  sampleCount_=0;
}

void ClassStats::Aggregate(const MicrosoftResearch::Cambridge::Sherwood::IDataPointCollection& data, unsigned int index)
{
    const ClassificationDB& db = dynamic_cast<const ClassificationDB&>(data);

    Aggregate(db.getNumericalLabel(index));
}

void ClassStats::Aggregate(bintype i)
{
#ifdef ENABLE_OVERFLOW_CHECKS
    if(i>=binCount_){
        std::cerr << "stats overflow" << std::endl;
        std::cerr.flush();
        //throw std::invalid_argument("ClassStats: index i should be less then number of classes");
    }
#endif

    bins_[i]++;
    sampleCount_++;

#ifdef ENABLE_OVERFLOW_CHECKS
    if(sampleCount_ == 0 ){
        std::cerr << "stats overflow" << std::endl;
        std::cerr.flush();
        //throw std::overflow_error("ClassStats: sampleCount_ overflow");
    }
#endif
}

double ClassStats::Entropy() const
{
    if (sampleCount_==0)
        return 0.0;


    double result = 0.0;
    for (int b = 0; b < binCount_; b++)
    {
      double p = (double)bins_[b] / (double)sampleCount_;
      result -= (p == 0.0) ? 0.0 : p * log(p)/log(2.0);
    }

    return result;
}

void ClassStats::Aggregate(const ClassStats& aggregator)
{

  if(aggregator.binCount_ != binCount_ && binCount_!=0){
      // do nothing
      std::cerr << "binCount_ does not correspond" <<std::endl;
      std::cerr << "binCount_:" << (int)binCount_ << "aggregator.binCount_:" << (int)aggregator.binCount_ <<std::endl;
  }

  if (binCount_==0){
        binCount_ = aggregator.binCount_;
        sampleCount_ = aggregator.sampleCount_;
        bins_ = new bintype[binCount_];
        memcpy(bins_, aggregator.bins_, sizeof(bintype)*binCount_);
        return;
  }

  for (int b = 0; b < binCount_; b++){
    bins_[b] += aggregator.bins_[b];
  }

#ifdef ENABLE_OVERFLOW_CHECKS
    if ((sampleCount_+aggregator.sampleCount_)<sampleCount_){
        std::cerr << "stats overflow" << std::endl;
        std::cerr.flush();
        //throw std::overflow_error("ClassStats: sampleCount_ overflow");
    }
#endif

  sampleCount_ += aggregator.sampleCount_;
}

ClassStats ClassStats::DeepClone() const
{
  return ClassStats(*this);
}
