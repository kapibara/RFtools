#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>


#include "regression/depthdbreg.h"
#include "regression/regtrainingcontext.h"
#include "configuration.h"
#include "copyfile.h"
#include "Tree.h"
#include "Random.h"
#include "Forest.h"
#include "ForestTrainer.h"
#include "depthfeature.h"

#define VoteDim 3
typedef float VoteType;

typedef VotesStatsT<VoteType,VoteDim> Stats;
typedef RegTrainingContext<Stats,FullDepthFeatureFactory> Context;

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





    DepthDBWithVotesSImpl<VoteType, VoteDim> db;

    if (config.useSubsampler()){
        log << "using subsampler - rate: " << config.subsamplerRate() << std::endl;
        db.setSubsampler(new RandomSubsampler(config.subsamplerRate(),random));
    }

    db.setCalibParam(config.calibration());
    log << "calibration paramerers: " << config.calibration() << std::endl;
    log << "loading db" << std::endl;
    db.loadDB(config.databaseFile(),config.databaseHasHeader());

    ProgressStream progress(log,Verbose);

    log << "loading from: " << config.databaseFile() << std::endl;
    log << "number of images: " << db.imageCount() << std::endl;
    log << "number of points: " << db.Count() << std::endl;
    log << "number of vote classes: " << (int)db.voteClassCount() << std::endl;
    log << "discard high variance: " << config.discardHighVar() << " with thr: " << config.nodeVarThr() << std::endl;

    log << "train a forest" << std::endl;
        //some work here
    DepthFeatureParameters featureParams = config.featureParameters();
    log << featureParams;

    TrainingParameters trainingParameters = config.forestParameters();

    log << "decision levels: " << trainingParameters.MaxDecisionLevels << std::endl
        << "trees: " << trainingParameters.NumberOfTrees << std::endl
        << "features sampled: " << trainingParameters.NumberOfCandidateFeatures << std::endl
        << "threashold per feature: " << trainingParameters.NumberOfCandidateThresholdsPerFeature << std::endl;

    FeatureAccomulator *accomulator = 0;
    TrainingParameters tp = config.forestParameters();

    FullDepthFeatureFactory *fff = new FullDepthFeatureFactory(config.featureParameters());
    Context *context = new Context(db.voteClassCount(),*fff,config.gainType(),config.voteDistThr()*config.voteDistThr());
    context->setFeatureAccomulator(accomulator);

    time_t start,end;
    std::auto_ptr<Forest<DepthFeature, Stats> > forest;

    time(&start);

    forest = ForestTrainer<DepthFeature, Stats>::TrainForest (random, trainingParameters, *context, db,&progress);

    time(&end);
    double dif = difftime (end,start);

    log << "time: " << dif << std::endl;
    log << "forest trained" << std::endl;

    std::ostream &out = cache.openBinStream("forest");
    forest->Serialize(out);

    log << "forest serialized" << std::endl;

    return 0;
}


