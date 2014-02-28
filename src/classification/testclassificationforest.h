#ifndef TESTCLASSIFICATIONFOREST_H
#define TESTCLASSIFICATIONFOREST_H

#include "TrainingParameters.h"
#include "depthfeature.h"
#include "localcache.h"
#include "depthdb.h"

#include <vector>

using namespace MicrosoftResearch::Cambridge::Sherwood;

struct ExtendedTrainingParameters
{
    friend std::ostream & operator<<(std::ostream &os, const ExtendedTrainingParameters& p){
        os << "Depth: " << p.paramForest_.MaxDecisionLevels << std::endl
           << "Number of trees: " << p.paramForest_.NumberOfTrees << std::endl
           << "F number: " << p.paramForest_.NumberOfCandidateFeatures << std::endl
           << "T number: " << p.paramForest_.NumberOfCandidateThresholdsPerFeature << std::endl
           << "Verbose: " << (int)p.paramForest_.Verbose << std::endl
           << "UV limit: " << p.paramFeatures_.uvlimit_ << std::endl
           << "Zero level: " << p.paramFeatures_.zeroplane_ << std::endl;

        return os;

    }

    ExtendedTrainingParameters(){
//do nothing
    }

    TrainingParameters paramForest_;
    DepthFeatureParameters paramFeatures_;
};

class TestClassificationForest
{
public:
    TestClassificationForest();

    void addParameterSet(const ExtendedTrainingParameters &param);

    void test(DepthDBClassImage *db);

private:
    std::vector<ExtendedTrainingParameters> params_;

};

#endif // TESTCLASSIFICATIONFOREST_H
