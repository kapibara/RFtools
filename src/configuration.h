#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <istream>

#include "depthfeature.h"
#include "TrainingParameters.h"

namespace MRF = MicrosoftResearch::Cambridge::Sherwood;

class Configuration
{
public:
    enum Factory {FullFeaturesFactory, PartialFeaturesFactory, FeaturePool, Unknown};

    Configuration();
    Configuration(std::istream &config);

    bool testOnly(){return testOnly_;}
    bool serializeInfo() {return serializeInfo_;}
    bool useSubsampler() {return subsamplerRate_>0;}
    bool discardHighVar() {return varianceThr_>0;}
    float testTrainSplit() {return testTrainSplit_;}
    float subsamplerRate() {return subsamplerRate_;}
    float nodeVarThr() {return varianceThr_;}

    std::string cacheFolderName() {return cacheFolderName_;}
    std::string databaseFile() {return dbFile_;}
    std::string forestFile() {return forestLocation_;}
    std::string featuresFile() {return featuresLocation_;}

    MRF::TrainingParameters forestParameters() {return forestParam_;}
    DepthFeatureParameters featureParameters() {return dfParams_;}
    Factory factoryType() {return factory_;}

private:

    Factory parseFactoryName(const std::string &string);

    bool testOnly_;
    bool serializeInfo_;


    Factory factory_;
    std::string forestType_;
    std::string forestLocation_;
    std::string dbFile_;
    std::string featuresLocation_;
    std::string cacheFolderName_;
    float subsamplerRate_;
    float varianceThr_;

    int candidatesCount_;
    float testTrainSplit_;

    DepthFeatureParameters  dfParams_;
    MRF::TrainingParameters forestParam_;
};

#endif // CONFIGURATION_H
