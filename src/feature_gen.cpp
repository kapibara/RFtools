
#include <iostream>

#include "depthfeature.h"
#include "localcache.h"
#include "parameter.h"
#include "stubtrainingcontext.h"
#include "configuration.h"

#include <TrainingParameters.h>
#include <Random.h>

using namespace MicrosoftResearch::Cambridge::Sherwood;

int main(int argc, char** argv)
{
    std::cout << "config: " << argv[1] << std::endl;
    std::cout << "reading config data" << std::endl;

    std::ifstream in(argv[1]);
    Configuration config(in);
    in.close();


    LocalCache cache(config.cacheFolderName(),"/home/kuznetso/tmp");

    if(!cache.init()){
        std::cerr << "failed to initialize temporary directory" << std::endl;
        exit(-1);
    }

    std::ostream &log = cache.log();

    try{

        TrainingParameters trainingParameters = config.forestParameters();

        log << "decision levels: " << trainingParameters.MaxDecisionLevels << std::endl
            << "trees: " << trainingParameters.NumberOfTrees << std::endl
            << "features sampled: " << trainingParameters.NumberOfCandidateFeatures << std::endl
            << "threashold per feature: " << trainingParameters.NumberOfCandidateThresholdsPerFeature <<std::endl
            << std::endl;

        DepthFeatureParameters dfparam = config.featureParameters();

        log << dfparam;

        PartialDepthFeatureFactory factory(dfparam);

        log << "factory used: PartialDepthFeatureFactory" << std::endl;

        std::ostream &featureStream = cache.openBinStream("features");

        StubTrainingContext<PartialDepthFeatureFactory> context(factory,config.forestParameters(),featureStream);
        Random random;

        int nodeCount = std::pow((double)2,trainingParameters.MaxDecisionLevels)-1;

        log << "generating features for " << nodeCount << " nodes" << std::endl;

        for(int i=0; i<nodeCount; i++){
            context.setCurrentNode(i);
            for(int f=0; f<trainingParameters.NumberOfCandidateFeatures;f++){
                //sample random features
                context.GetRandomFeature(random);
            }
        }

        log << "finished generating" << std::endl;


    }catch(std::exception e){
        std::cerr << "exception caught: " << e.what() << std::endl;
        std::cerr.flush();
    }
    //that's it, log will close the features file
}
