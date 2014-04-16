#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "Tree.h"
#include "Random.h"
#include "Forest.h"
#include "ForestTrainer.h"
#include "depthfeature.h"
#include "parameter.h"
#include "localcache.h"
#include "TrainingParameters.h"
#include "stubtrainingcontext.h"
#include "regression/votesstatst.h"
#include "regression/depthdbreg.h"
#include "regression/regtrainingcontext.h"
#include "stubstats.h"
#include "string2number.hpp"
#include "rfutils.h"
#include "featurepool.h"
#include "nodedistributionimagestats.h"
#include "configuration.h"
#include "copyfile.h"

#include <time.h>

#define VoteDim 3
typedef float VoteType;

typedef VotesStatsT<VoteType,VoteDim> Stats;

ITrainingContext<DepthFeature,Stats>  *createTrainingContext(Configuration &config,
                                                                  LocalCache &cache,
                                                                  uchar classCount)
{
    ITrainingContext<DepthFeature,Stats> *context = 0;
    FeatureAccomulator *accomulator = 0;
    TrainingParameters tp = config.forestParameters();

    if(config.serializeInfo()){
        std::ostream &featureOutput = cache.openBinStream("accomulatedFeatures");
        accomulator = new FeatureAccomulator(featureOutput,tp.NumberOfCandidateFeatures*tp.NumberOfCandidateThresholdsPerFeature);
    }

    switch(config.factoryType()){
    case Configuration::FeaturePool:
    {
        cache.log() << "feature factory: FeaturePool" << std::endl;
        std::ifstream in(config.featuresFile().c_str());
        FeaturePool *fp = new FeaturePool(in);
        context =  new RegTrainingContext<Stats,FeaturePool>(classCount,*fp,config.voteDistThr()*config.voteDistThr());
        break;
    }
    case Configuration::FullFeaturesFactory:
    {
        cache.log() << "feature factory: FullDepthFeatureFactory" << std::endl;
        FullDepthFeatureFactory *fff = new FullDepthFeatureFactory(config.featureParameters());
        context = new RegTrainingContext<Stats,FullDepthFeatureFactory>(classCount,*fff,config.voteDistThr()*config.voteDistThr());
        if (config.serializeInfo()){
            ((RegTrainingContext<Stats,FullDepthFeatureFactory> *)context)->setFeatureAccomulator(accomulator);
        }
        break;
    }
    case Configuration::PartialFeaturesFactory:
    {
        cache.log() << "feature factory: PartialDepthFeatureFactory" << std::endl;
        PartialDepthFeatureFactory *pff = new PartialDepthFeatureFactory(config.featureParameters());
        context = new RegTrainingContext<Stats,PartialDepthFeatureFactory>(classCount,*pff,config.voteDistThr()*config.voteDistThr());
        if (config.serializeInfo()){
            ((RegTrainingContext<Stats,PartialDepthFeatureFactory> *)context)->setFeatureAccomulator(accomulator);
        }
        break;
    }
    default:
        cache.log() << "Unknown feature factory type; nothing to return" << std::endl;
    }

    return context;
}

int main(int argc, char **argv)
{
    std::cout << "config: " << argv[1] << std::endl;
    std::cout << "reading config data" << std::endl;

    std::ifstream in(argv[1]);
    Configuration config(in);
    in.close();

    std::cout << "configuration loaded" << std::endl;

    LocalCache cache(config.cacheFolderName(),"/home/kuznetso/tmp");

    if(!cache.init()){
        std::cerr << "failed to initialize temporary directory" << std::endl;
        exit(-1);
    }

    std::ostream &log = cache.log();

    log << "copying config file" << std::endl;
    copyfile(argv[1],cache.base() + "config.xml");

    log << "config name: " << argv[1] << std::endl;

    try{
        DepthDBWithVotesSImpl<VoteType,VoteDim>  db;
        Random random;
        time_t start,end;
        ProgressStream progress(log,Verbose);

        std::auto_ptr<Forest<DepthFeature, Stats> > forest;

        if (config.useSubsampler()){
            log << "using subsampler - rate: " << config.subsamplerRate() << std::endl;
            db.setSubsampler(new RandomSubsampler(config.subsamplerRate(),random));
        }
        db.loadDB(config.databaseFile(),config.databaseHasHeader());

        log << "loading from: " << config.databaseFile() << std::endl;
        log << "number of images: " << db.imageCount() << std::endl;
        log << "number of points: " << db.Count() << std::endl;
        log << "number of vote classes: " << (int)db.voteClassCount() << std::endl;
        log << "test samples: " << 1-config.testTrainSplit() << std::endl;
        log << "discard high variance: " << config.discardHighVar() << " with thr: " << config.nodeVarThr() << std::endl;

        std::auto_ptr<DepthDBWithVotesSubindex> test;
        std::auto_ptr<DepthDBWithVotesSubindex> train;
        RFUtils::splitRandom<DepthDBWithVotesSubindex>(random,db,train,test,config.testTrainSplit());

        log << "train set size: " << train->Count() << std::endl;
        log << "test set size: " << test->Count() << std::endl;

        if(!config.testOnly()){

            log << "train a forest" << std::endl;
            //some work here
            DepthFeatureParameters featureParams = config.featureParameters();

            log << featureParams;

            TrainingParameters trainingParameters = config.forestParameters();

            log << "decision levels: " << trainingParameters.MaxDecisionLevels << std::endl
                << "trees: " << trainingParameters.NumberOfTrees << std::endl
                << "features sampled: " << trainingParameters.NumberOfCandidateFeatures << std::endl
                << "threashold per feature: " << trainingParameters.NumberOfCandidateThresholdsPerFeature <<std::endl
                << std::endl;

            ITrainingContext<DepthFeature,Stats> *context = createTrainingContext(config,cache,db.voteClassCount());

            log << "training images used: " << train->imageCount() << std::endl;

            time(&start);

            forest = ForestTrainer<DepthFeature, Stats>::TrainForest (
                random, trainingParameters, *context, db,&progress);

            time(&end);
            double dif = difftime (end,start);

            log << "time: " << dif << std::endl;
            log << "forest trained" << std::endl;

            std::ostream &out = cache.openBinStream("forest");
            forest->Serialize(out);

            log << "forest serialized" << std::endl;
        }

        else{

            log << "load a forest from: " << config.forestFile().c_str() << std::endl;

            std::ifstream in(config.forestFile().c_str(),std::ios_base::binary);

            forest = Forest<DepthFeature, Stats>::Deserialize(in);

            log << "forest deserialized" << std::endl;
        }

        std::ostream &log = cache.log();

        std::vector<std::vector<int> > leafIndicesPerTree;
        forest->Apply(db,leafIndicesPerTree,&progress);

        std::ostream &leafIds = cache.openBinStream("leafIds");
        serializeVector<int>(leafIds,leafIndicesPerTree[0]);

        log << "forest applied" << std::endl;

        std::vector<int> imgIds(db.Count(),0);
        std::vector<int> x(db.Count(),0);
        std::vector<int> y(db.Count(),0);
        std::string tmpstr;
        cv::Point2i current;

        for(int i=0; i<db.Count(); i++){
            db.getDataPoint(i,tmpstr,current);
            imgIds[i] = db.getImageIdx(i);
            x[i] = current.x;
            y[i] = current.y;
        }

        std::ostream &ids = cache.openBinStream("imgIds");
        serializeVector<int>(ids,imgIds);

        std::ostream &xvals = cache.openBinStream("xVals");
        serializeVector<int>(xvals,x);

        std::ostream &yvals = cache.openBinStream("yVals");
        serializeVector<int>(yvals,y);

        log << "indices saved" << std::endl;


    }catch(std::exception e){
        std::cerr << "exception caught: " << e.what() << std::endl;
        std::cerr.flush();
    }
}

