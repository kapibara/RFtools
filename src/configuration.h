#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <istream>

#include "depthfeature.h"
#include "TrainingParameters.h"

#include <boost/property_tree/ptree.hpp>

namespace MRF = MicrosoftResearch::Cambridge::Sherwood;
namespace bpt = boost::property_tree;

class Configuration
{
public:
    enum Factory {FullFeaturesFactory, PartialFeaturesFactory, FeaturePool, Unknown};

    struct ForestParam{
        ForestParam(const std::string &ff,
                    const std::string &leafsFile): ff_(ff),lf_(leafsFile)
        {

        }

        std::string ff_;
        std::string lf_;
        std::vector<float> bounds_;
        std::vector<float> mean_;
        std::vector<float> std_;
    };

    Configuration();
    Configuration(std::istream &config);

    bool testOnly() const{return testOnly_;}
    bool serializeInfo() const{return serializeInfo_;}
    bool useSubsampler() const{return subsamplerRate_>0;}
    bool discardHighVar() const{return varianceThr_>0;}
    bool databaseHasHeader() const{return dbHasHeader_;}
    bool testOnTrain() const{return testOnTrain_;}
    bool testOnTest() const{return testOnTest_;}
    float testTrainSplit() const{return testTrainSplit_;}
    float subsamplerRate() const{return subsamplerRate_;}
    float nodeVarThr() const{return varianceThr_;}
    int nodeSizeThr() const{return sizeThr_;}
    int voteDistThr() const{return voteDistThr_;}

    float meanShiftR() const{return r_;}
    int meanShiftMaxIter() const{return maxIter_;}
    float smallWeightThr() const{return weightThr_;}
    int maxNN() const {return maxNN_;}

    std::string gainType() const{return gainType_;}
    std::string cacheFolderName() const{return cacheFolderName_;}
    std::string databaseFile() const{return dbFile_;}
    std::string featuresFile() const{return featuresLocation_;}
    std::string otherForestFile() const {return otherForestFile_;}

    /*default function; for backward compatibility*/
    std::string forestFile() const{return forests_[0].ff_;}
    std::string leafsFile() const{return forests_[0].lf_;}

    int forestsCount() const {return forests_.size();}

    const ForestParam &forest(int idx) const {return forests_[idx];}

    MRF::TrainingParameters forestParameters() const{return forestParam_;}
    DepthFeatureParameters featureParameters() const{return dfParams_;}
    Factory factoryType() const{return factory_;}

private:

    Factory parseFactoryName(const std::string &string);

    void readForestsList(bpt::ptree &props);
    void readFloatVector(const std::string &in,std::vector<float> &out);

    bool testOnly_;
    bool serializeInfo_;
    bool dbHasHeader_;
    bool testOnTrain_;
    bool testOnTest_;

    Factory factory_;
    std::string forestType_;
    std::string otherForestFile_;
    std::string dbFile_;
    std::string featuresLocation_;
    std::string cacheFolderName_;
    std::string gainType_;

    std::vector<ForestParam> forests_;

    float subsamplerRate_;
    float varianceThr_;

    int candidatesCount_;
    int voteDistThr_;
    int sizeThr_;
    float testTrainSplit_;

    float r_;
    int maxIter_;
    int maxNN_;
    float weightThr_;

    DepthFeatureParameters  dfParams_;
    MRF::TrainingParameters forestParam_;
};

#endif // CONFIGURATION_H
