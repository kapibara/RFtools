#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "Tree.h"
#include "Random.h"
#include "Forest.h"
#include "ForestTrainer.h"
#include "depthfeature.h"
#include "hough/houghvotesstats.h"
#include "parameter.h"
#include "localcache.h"
#include "TrainingParameters.h"
#include "stubtrainingcontext.h"
#include "hough/depthdbwithvotes.h"
#include "hough/houghtrainingcontext.h"
#include "stubstats.h"
#include "string2number.hpp"
#include "rfutils.h"
#include "featurepool.h"
#include "nodedistributionimagestats.h"
#include "configuration.h"
#include "copyfile.h"

#include <time.h>

//#define TRAIN_TEST_RANDOM

using namespace MicrosoftResearch::Cambridge::Sherwood;

void applyForest(Forest<DepthFeature, VotesStats> &forest,
                 ProgressStream &progress,
                 DepthDBWithVotesSubindex &db,
                 const Configuration &config,
                 const std::string &suffix,
                 LocalCache &cache)
{
    std::ostream &log = cache.log();

    std::vector<std::vector<int> > leafIndicesPerTree;
    forest.Apply(db,leafIndicesPerTree,&progress);

    std::ostream &leafIds = cache.openBinStream("leafIds"+suffix);
    serializeVector<int>(leafIds,leafIndicesPerTree[0]);

    log << "forest applied" << std::endl;

    cv::Point2i current;
    cv::Size imgSize(480,640);
    std::string tmpstr;
    std::vector< std::vector <HoughVotesStats> > fullStats;

    for (int v = 0; v < db.voteClassCount(); v++){
        fullStats.push_back(std::vector<HoughVotesStats>());

        for(int i=0; i< db.imageCount(); i++){
            if (db.getRelative(v)){
                fullStats.back().push_back(HoughVotesStats(imgSize,v));
            }
            else{
                fullStats.back().push_back(HoughVotesStats(imgSize,v,cv::Point2i(240,320)));
            }
       }
    }


    log << "full stats vector created" << std::endl;
    log << "image count: " << db.imageCount() << std::endl;

    std::vector<int> imgIds(db.Count(),0);
    std::vector<int> x(db.Count(),0);
    std::vector<int> y(db.Count(),0);
    std::vector<double> variance(db.Count(),0);

    VotesStats *trStats;

    for(int i=0; i<db.Count(); i++){
        db.getDataPoint(i,tmpstr,current);
        imgIds[i] = db.getOriginalImageIdx(i);
        x[i] = current.x;
        y[i] = current.y;
        for(int t=0; t<forest.TreeCount(); t++)
        {

            if (leafIndicesPerTree[t][i]>=0){
                trStats = &(forest.GetTree(t).GetNode(leafIndicesPerTree[t][i]).TrainingDataStatistics);
                trStats->FinalizeDistribution();
                variance[i] = trStats->VoteVariance();
                if(!config.discardHighVar() || variance[i] < config.nodeVarThr()){
                    for(int v = 0; v < db.voteClassCount(); v++){
                        if(db.getRelative(v)){
                            fullStats[v][db.getImageIdx(i)].Aggregate(current,*trStats);
                        }else{
                            fullStats[v][db.getImageIdx(i)].Aggregate(cv::Point2i(0,0),*trStats);
                        }

                    }
                }
            }
        }
    }

    std::ostream &agforest = cache.openBinStream("aggregatedForest"+suffix);
    forest.Serialize(agforest);

    std::ostream &vars = cache.openBinStream("vars"+suffix);
    serializeVector<double>(vars,variance);

    std::ostream &ids = cache.openBinStream("imgIds"+suffix);
    serializeVector<int>(ids,imgIds);

    std::ostream &xvals = cache.openBinStream("xVals"+suffix);
    serializeVector<int>(xvals,x);

    std::ostream &yvals = cache.openBinStream("yVals"+suffix);
    serializeVector<int>(yvals,y);


    log << "statistic aggregated" << std::endl;

    std::string filename;

    std::vector<bool> seen(db.imageCount(),false);
    std::vector<cv::Point2i> votes(db.voteClassCount());
    cv::Point2i p;

    std::ofstream *stream;

    for(int i=0; i<db.Count(); i++){
        std::cerr.flush();
        if(!seen[db.getImageIdx(i)])
        {
            seen[db.getImageIdx(i)] = true;
            log << "i: " << i <<
                   " filename: " << db.imageIdx2Filename(db.getOriginalImageIdx(i)) <<
                   " imindex: "  << db.getOriginalImageIdx(i) << std::endl;

            db.getDataPointVote(i,votes);
            db.getDataPoint(i,filename,p);

            for(int v = 0; v < db.voteClassCount(); v++){

                if (db.getRelative(v)){
                    fullStats[v][db.getImageIdx(i)].setGT(votes[v] + p);
                }
                else{
                    fullStats[v][db.getImageIdx(i)].setGT(votes[v]);
                }

                filename = "img_" + num2str<int>(db.getOriginalImageIdx(i))+ "_" + num2str<int>(v)+suffix;
                stream = (std::ofstream *)&cache.openBinStream(filename);
                fullStats[v][db.getImageIdx(i)].Serialize(*stream);
                stream->close();

                fullStats[v][db.getImageIdx(i)].Serialize(cache.base() + filename + std::string(".png"));

            }
        }
    }

    log << "results serialized" << std::endl;

}

ITrainingContext<DepthFeature,VotesStats>  *createTrainingContext(Configuration &config,
                                                                  LocalCache &cache,
                                                                  uchar classCount)
{
    ITrainingContext<DepthFeature,VotesStats> *context = 0;
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
        context =  new HoughTrainingContext<FeaturePool>(classCount,*fp,config.voteDistThr()*config.voteDistThr());
        break;
    }
    case Configuration::FullFeaturesFactory:
    {
        cache.log() << "feature factory: FullDepthFeatureFactory" << std::endl;
        FullDepthFeatureFactory *fff = new FullDepthFeatureFactory(config.featureParameters());
        context = new HoughTrainingContext<FullDepthFeatureFactory>(classCount,*fff,config.voteDistThr()*config.voteDistThr());
        if (config.serializeInfo()){
            ((HoughTrainingContext<FullDepthFeatureFactory> *)context)->setFeatureAccomulator(accomulator);
        }
        break;
    }
    case Configuration::PartialFeaturesFactory:
    {
        cache.log() << "feature factory: PartialDepthFeatureFactory" << std::endl;
        PartialDepthFeatureFactory *pff = new PartialDepthFeatureFactory(config.featureParameters());
        context = new HoughTrainingContext<PartialDepthFeatureFactory>(classCount,*pff,config.voteDistThr()*config.voteDistThr());
        if (config.serializeInfo()){
            ((HoughTrainingContext<PartialDepthFeatureFactory> *)context)->setFeatureAccomulator(accomulator);
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
        DepthDBWithVotesImpl db;
        Random random;
        time_t start,end;
        ProgressStream progress(log,Verbose);

        std::auto_ptr<Forest<DepthFeature, VotesStats> > forest;

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

            ITrainingContext<DepthFeature,VotesStats> *context = createTrainingContext(config,cache,db.voteClassCount());

            log << "training images used: " << train->imageCount() << std::endl;

            time(&start);

            forest = ForestTrainer<DepthFeature, VotesStats>::TrainForest (
                random, trainingParameters, *context, *train ,&progress);

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

            forest = Forest<DepthFeature, VotesStats>::Deserialize(in);

            log << "forest deserialized" << std::endl;
        }

        if (config.testOnTrain()){
            log << "applying forest on training data" << std::endl;
            applyForest(*forest,progress,*train,config,"_train",cache);
        }
        if (config.testOnTest()){
            log << "applying forest on test data" << std::endl;
            applyForest(*forest,progress,*test,config,"_test",cache);
        }


    }catch(std::exception e){
        std::cerr << "exception caught: " << e.what() << std::endl;
        std::cerr.flush();
    }
}
