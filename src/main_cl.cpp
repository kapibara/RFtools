
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

LocalCache cache("DepthCL","/home/kuznetso/tmp");

if(!cache.init()){
    std::cerr << "failed to initialize temporary directory" << std::endl;
    exit(-1);
}

std::ostream &log = cache.log();

try{
   std::auto_ptr<Forest<DepthFeature, ClassStats> > forest;

   DepthDBClassImage * db = new DepthDBClassImage();
   std::auto_ptr<DepthDBSubindex> test;
   std::auto_ptr<DepthDBSubindex> train;
   Random  random;

   log << "input file: " << argv[1] << std::endl;

   db->loadDB(argv[1]);

   if (argc==2){

        log << "starting training ... " << std::endl;
        RFUtils::splitRandom<DepthDBSubindex ,DepthFileBasedImageDB>(random,*db,train,test);

        log << "Train samples:" << train->Count() << std::endl;
        log << "Test samples:" << test->Count() << std::endl;

        log << "db loaded ... " << std::endl;


        Parameter<int> T(1, "No. of trees in the forest.");
        Parameter<int> D(2, "Maximum tree levels.");
        Parameter<int> F(200, "No. of candidate feature response functions per split node.");
        Parameter<int> L(20, "No. of candidate thresholds per feature response function.");
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

        DepthFeatureParameters param;
        param.uvlimit_ = 25;
        param.zeroplane_ = 300;
        DepthFeatureFactory factory(param);
        ClTrainingContext context(db->classCount(),factory);

        log << param;

        log << "start forest training ... " << std::endl;

        time_t start,end;
        ProgressStream ps(log,Verbose);

        time(&start);

        forest = ForestTrainer<DepthFeature, ClassStats>::TrainForest (
                random, trainingParameters, context, *train, &ps );

        time(&end);
        double dif = difftime (end,start);

        log << "Forest trained: " << forest->GetTree(0).NodeCount() << std::endl;
        log << "Time: " << dif << std::endl;

        std::string fname = cache.base() + std::string("testout");
        std::ofstream out(fname.c_str());

        forest->Serialize(out);

        out.close();

        log << "Forest saved" << std::endl;

    }
    else if (argc>2){
        log << "loading the forest ... " << std::endl;

        std::ifstream in(argv[2]);
        forest = Forest<DepthFeature, ClassStats>::Deserialize(in);

        log << "Forest deserialized" << std::endl;

   }

    ClassStats clStatsPixel(db->classCount());

    double result = RFUtils::testClassificationForest<DepthFeature,ClassStats>(*forest,clStatsPixel,*test,cache,true);

    log << "Error: " << result << std::endl;

    log << "Statistics computed" << std::endl;

    for(int i=0;i<db->classCount();i++){
        std::cerr << db->labelIndex2Name(i) << "-" << (int)i << std::endl;
    }

    }catch(std::exception &e){
        log << "exception caught 2: " << e.what() << std::endl;
    }
}
