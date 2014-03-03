
#include <iostream>
#include <fstream>
#include <ctime>

#ifdef WIN32
#include <windows.h>
#endif

#include "classification/depthdb.h"
#include "classification/cltrainingcontext.h"

#include "TrainingParameters.h"
#include "Forest.h"
#include "ForestTrainer.h"
#include "parameter.h"
#include "rfutils.h"
#include "localcache.h"
#include "classification/classstats.h"
#include "classification/imagepixelstats.h"

#include <climits>
#include "split.h"

using namespace MicrosoftResearch::Cambridge::Sherwood;

int main(int argc, char **argv)
{
    for(int i=0; i<3; i++){
        LocalCache cache(argc,argv);
        cache.init();
    }

    exit(-1);

if (argc<2){
    std::cout << "exec <db file>" << std::endl;
    exit(-1);
}

LocalCache cache(argc,argv);

if(!cache.init()){
    std::cerr << "failed to initialize temporary directory" << std::endl;
    exit(-1);
}

std::ostream &log = cache.log();

try{
   std::auto_ptr<Forest<DepthFeature, ClassStats> > forest;

   std::auto_ptr<DepthDBClassImage> db(new DepthDBClassImage());
   std::auto_ptr<DepthFileBasedImageDB> test;
   std::auto_ptr<DepthFileBasedImageDB> train;
   Random  random;
   std::vector<std::vector<int> > leafIndicesPerTree;

   db->loadDB(argv[1]);

   if (argc==2){
        log << "starting training ... " << std::endl;
        RFUtils::splitRandom<DepthDBSubindex ,DepthFileBasedImageDB>(random,*db,train,test);

        log << "Train samples:" << train->Count() << std::endl;
        log << "Test samples:" << test->Count() << std::endl;

        log << "db loaded ... " << std::endl;


        Parameter<int> T(1, "No. of trees in the forest.");
        Parameter<int> D(2, "Maximum tree levels.");
        Parameter<int> F(100, "No. of candidate feature response functions per split node.");
        Parameter<int> L(10, "No. of candidate thresholds per feature response function.");
        Parameter<bool> verbose(true,"Enables verbose progress indication.");

        log << "Parameters" << std::endl;
        log << T << ";" << D << ";" << F << ";" << L << std::endl;

        TrainingParameters trainingParameters;
        trainingParameters.MaxDecisionLevels = D.value()-1;
        trainingParameters.NumberOfCandidateFeatures = F.value();
        trainingParameters.NumberOfCandidateThresholdsPerFeature = L.value();
        trainingParameters.NumberOfTrees = T.value();
        trainingParameters.Verbose = verbose.value();

        log << "class count: "  << db->classCount() << " element count: " << db->Count()<<std::endl;

        DepthFeatureFactory factory;
        ClTrainingContext<DepthFeature, ClassStats> context(db->classCount(),factory);

        log << "start forest training ... " << std::endl;

        ProgressStream progress(log,Verbosity::Verbose);

        forest = ForestTrainer<DepthFeature, ClassStats>::TrainForest (
                random, trainingParameters, context, *train, &progress );

        log << "Forest trained: " << forest->GetTree(0).NodeCount() << std::endl;

        std::ofstream out(cache.base() + std::string("testout"));
        forest->Serialize(out);

        log << "Forest saved" << std::endl;

    }
    else if (argc>2){
        log << "loading the forest ... " << std::endl;

        std::ifstream in("testout");
        forest = Forest<DepthFeature, ClassStats>::Deserialize(in);

        log << "Forest deserialized" << std::endl;

        test = db;
   }

    ClassificationDB *testascldb = dynamic_cast<ClassificationDB *>(test.get());

    ClassStats clStatsPixel(testascldb->classCount());

    double result = RFUtils::testClassificationForest<DepthFeature,ClassStats>(*forest,clStatsPixel,*test,cache,true);

    log << "Error: " << result << std::endl;

    log << "Statistics computed" << std::endl;

    }catch(std::exception &e){
        log << "exception caught 2: " << e.what() << std::endl;
    }
}
