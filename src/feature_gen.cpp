
#include <iostream>

#include "depthfeature.h"
#include "localcache.h"
#include "parameter.h"
#include "stubtrainingcontext.h"

#include <TrainingParameters.h>
#include <Random.h>

using namespace MicrosoftResearch::Cambridge::Sherwood;

int main(int argc, char** argv)
{
    LocalCache cache("Generator","/home/kuznetso/tmp");

    if(!cache.init()){
        std::cerr << "failed to initialize temporary directory" << std::endl;
        exit(-1);
    }

    std::ostream &log = cache.log();

    try{

        Parameter<int> T(1, "No. of trees in the forest.");
        Parameter<int> D(10, "Maximum tree levels.");
        Parameter<int> F(1500, "No. of candidate feature response functions per split node.");
        Parameter<int> L(10, "No. of candidate thresholds per feature response function.");
        Parameter<bool> verbose(true,"Enables verbose progress indication.");

        log << T << D << F << L << std::endl;

        TrainingParameters trainingParameters;
        trainingParameters.MaxDecisionLevels = D.value()-1;
        trainingParameters.NumberOfCandidateFeatures = F.value();
        trainingParameters.NumberOfCandidateThresholdsPerFeature = L.value();
        trainingParameters.NumberOfTrees = T.value();
        trainingParameters.Verbose = verbose.value();

        DepthFeatureParameters featureParams;
        featureParams.uvlimit_ = 30;
        featureParams.zeroplane_ = 300;

        log << featureParams;

        PartialDepthFeatureFactory factory(featureParams);

        log << "factory used: PartialDepthFeatureFactory" << std::endl;

        std::ostream &featureStream = cache.openBinStream("features");

        StubTrainingContext<PartialDepthFeatureFactory> context(factory,trainingParameters,featureStream);
        Random random;

        int nodeCount = std::pow((double)2,D.value()-1)-1;

        log << "generating features for " << nodeCount << " nodes" << std::endl;

        for(int i=0; i<nodeCount; i++){
            context.setCurrentNode(i);
            for(int f=0; f<F.value();f++){
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
