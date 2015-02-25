
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>

#include "ProgressStream.h"
#include "Forest.h"
#include "TrainingParameters.h"

#include "perpixelclass/perpixelclassificationdb.h"
#include "classification/cltrainingcontext.h"
#include "classification/classstats.h"
#include "parallel/paralleltreetrainer.h"
#include "depthfeature.h"
#include "featureaccomulator.h"
#include "configuration.h"
#include "localcache.h"
#include "copyfile.h"


using namespace MicrosoftResearch::Cambridge::Sherwood;

int main(int argc, char **argv)
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

    log << "copying config file" << std::endl;
    copyfile(argv[1],cache.base() + "config.xml");
    Random random;

    PerPixelClassificationDB db;

    if (config.useSubsampler()){
        log << "using subsampler - rate: " << config.subsamplerRate() << std::endl;
        db.setSubsampler(new RandomSubsampler(config.subsamplerRate(),random));
    }


    db.loadDB(config.databaseFile());

    ProgressStream progress(log,Verbose);

    log << "loading from: " << config.databaseFile() << std::endl;
    log << "number of images: " << db.imageCount() << std::endl;
    log << "number of points: " << db.Count() << std::endl;
    log << "number of vote classes: " << (int)db.classCount() << std::endl;
    log << "discard high variance: " << config.discardHighVar() << " with thr: " << config.nodeVarThr() << std::endl;

    log << "train a forest" << std::endl;
    DepthFeatureParameters featureParams = config.featureParameters();
    log << featureParams;

    TrainingParameters trainingParameters = config.forestParameters();

    log << "decision levels: " << trainingParameters.MaxDecisionLevels << std::endl
        << "trees: " << trainingParameters.NumberOfTrees << std::endl
        << "features sampled: " << trainingParameters.NumberOfCandidateFeatures << std::endl
        << "threashold per feature: " << trainingParameters.NumberOfCandidateThresholdsPerFeature << std::endl;


    FullDepthFeatureFactory *fff = new FullDepthFeatureFactory(config.featureParameters());
    ClTrainingContext *context = new ClTrainingContext(db.classCount(),*fff);

    time_t start,end;
    std::auto_ptr<Forest<DepthFeature, ClassStats> > forest;

    time(&start);

    forest = TreeBasedParallelTrainer<DepthFeature, ClassStats>::TrainForest (random, trainingParameters, *context, db,&progress);

    time(&end);
    double dif = difftime (end,start);

    log << "time: " << dif << std::endl;
    log << "forest trained" << std::endl;

    std::ostream &out = cache.openBinStream("forest");
    forest->Serialize(out);

    log << "forest serialized" << std::endl;

    return 0;

}
