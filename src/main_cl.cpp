
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

#include <time.h>
#include "split.h"


using namespace MicrosoftResearch::Cambridge::Sherwood;

int main(int argc, char **argv)
{

if (argc<2){
    std::cout << "exec <db file>" << std::endl;
    exit(-1);
}

LocalCache cache(argc,argv,"/home/kuznetso/tmp");

if(!cache.init()){
    std::cerr << "failed to initialize temporary directory" << std::endl;
    exit(-1);
}

try{
   std::auto_ptr<Forest<DepthFeature, ClassStats> > forest;

   DepthDBClassImage * db = new DepthDBClassImage();
   DepthFileBasedImageDB *test;
   DepthFileBasedImageDB *train;
   Random  random;

   db->loadDB(argv[1]);

   if (argc==2){
        std::cout << "starting training ... " << std::endl;
        //RFUtils::splitRandom<DepthDBSubindex ,DepthFileBasedImageDB>(random,*db,train,test);


//        std::cout << "Train samples:" << train->Count() << std::endl;
//        std::cout << "Test samples:" << test->Count() << std::endl;

        std::cerr << "db loaded ... " << std::endl;
        std::cerr.flush();

        Parameter<int> T(1, "No. of trees in the forest.");
        Parameter<int> D(18, "Maximum tree levels.");
        Parameter<int> F(100, "No. of candidate feature response functions per split node.");
        Parameter<int> L(10, "No. of candidate thresholds per feature response function.");
        Parameter<bool> verbose(true,"Enables verbose progress indication.");

        TrainingParameters trainingParameters;
        trainingParameters.MaxDecisionLevels = D.value()-1;
        trainingParameters.NumberOfCandidateFeatures = F.value();
        trainingParameters.NumberOfCandidateThresholdsPerFeature = L.value();
        trainingParameters.NumberOfTrees = T.value();
        trainingParameters.Verbose = verbose.value();

        std::cerr <<"class count: "  << db->classCount() << " element count: " << db->Count()<<std::endl;

        DepthFeatureFactory factory;
        ClTrainingContext context(db->classCount(),factory);

        std::cerr << "start forest training ... ; depth " << D.value() << std::endl;
        std::cerr.flush();

        time_t start,end;

        time(&start);

        forest = ForestTrainer<DepthFeature, ClassStats>::TrainForest (
                random, trainingParameters, context, *db );

        time(&end);
        double dif = difftime (end,start);

        std::cerr << "Forest trained: " << forest->GetTree(0).NodeCount() << std::endl;
        std::cerr << dif << " seconds ellapsed" << std::endl;
        std::cerr.flush();

        std::string fname = cache.base() + std::string("testout");
        std::ofstream out(fname.c_str());

        forest->Serialize(out);

        std::cerr << "Forest saved" << std::endl;
        std::cerr.flush();

    }
    else if (argc>2){
        std::cout << "loading the forest ... " << std::endl;

        std::ifstream in(argv[2]);
        forest = Forest<DepthFeature, ClassStats>::Deserialize(in);

        std::cerr << "Forest deserialized" << std::endl;
        std::cerr.flush();

   }

    ClassStats clStatsPixel(db->classCount());

    double result = RFUtils::testClassificationForest<DepthFeature,ClassStats>(*forest,clStatsPixel,*db,cache,true);

    std::cerr << "Error: " << result << std::endl;

    std::cerr << "Statistics computed" << std::endl;
    std::cerr.flush();

    for(int i=0;i<db->classCount();i++){
        std::cerr << db->labelIndex2Name(i) << "-" << (int)i << std::endl;
    }

    }catch(std::exception &e){
        std::cerr << "exception caught 2: " << e.what() << std::endl;
        std::cerr.flush();
    }
}
