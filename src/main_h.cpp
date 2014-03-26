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
#include "hough/houghtrainingcontext.h"
#include "stubstats.h"
//#include "classification/classstats.h"
#include "string2number.hpp"
#include "rfutils.h"
#include "featurepool.h"
#include "nodedistributionimagestats.h"
#include "configuration.h"

#include <time.h>

//#define TRAIN_TEST_RANDOM

using namespace MicrosoftResearch::Cambridge::Sherwood;

ITrainingContext<DepthFeature,VotesStats>  *createTrainingContext(Configuration &config, LocalCache &cache, uchar classCount)
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
        std::ifstream in(config.featuresFile().c_str());
        FeaturePool *fp = new FeaturePool(in);
        context =  new HoughTrainingContext<FeaturePool>(classCount,*fp);
        break;
    }
    case Configuration::FullFeaturesFactory:
    {
        FullDepthFeatureFactory *fff = new FullDepthFeatureFactory(config.featureParameters());
        context = new HoughTrainingContext<FullDepthFeatureFactory>(classCount,*fff);
        if (config.serializeInfo()){
            ((HoughTrainingContext<FullDepthFeatureFactory> *)context)->setFeatureAccomulator(accomulator);
        }
        break;
    }
    case Configuration::PartialFeaturesFactory:
    {
        PartialDepthFeatureFactory *pff = new PartialDepthFeatureFactory(config.featureParameters());
        context = new HoughTrainingContext<PartialDepthFeatureFactory>(classCount,*pff);
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

    std::ifstream in("/home/kuznetso/Projects/CPP/DepthRF/propertiesTemplate.xml");
    Configuration config(in);
    in.close();

    std::cout << "configuration loaded" << std::endl;


    LocalCache cache(config.cacheFolderName(),"/home/kuznetso/tmp");

    if(!cache.init()){
        std::cerr << "failed to initialize temporary directory" << std::endl;
        exit(-1);
    }

    std::ostream &log = cache.log();

    log << "config name: " << argv[1] << std::endl;

    try{
        DepthDBWithVotesImpl db;
        Random random;
        time_t start,end;
        ProgressStream progress(log,Verbose);

        std::auto_ptr<Forest<DepthFeature, VotesStats> > forest;

        db.loadDB(config.databaseFile());

        log << "loading from: " << config.databaseFile() << std::endl;
        log << "number of images: " << db.imageCount() << std::endl;
        log << "number of points: " << db.Count() << std::endl;
        log << "number of vote classes: " << (int)db.voteClassCount() << std::endl;

        std::auto_ptr<DepthDBWithVotesSubindex> test;
        std::auto_ptr<DepthDBWithVotesSubindex> train;
        RFUtils::splitRandom<DepthDBWithVotesSubindex>(random,db,train,test,config.testTrainSplit());

        log << "train set size: " << train->Count() << std::endl;
        log << "test set size: " << test->Count() << std::endl;

        if(!config.testOnly()){
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

            std::ifstream in(config.forestFile().c_str(),std::ios_base::binary);

            forest = Forest<DepthFeature, VotesStats>::Deserialize(in);

            log << "forest deserialized" << std::endl;
        }

        std::vector<std::vector<int> > leafIndicesPerTree;
        forest->Apply(*test,leafIndicesPerTree,&progress);

        log << "forest applied" << std::endl;

        cv::Point2i current;
        std::string tmpstr;
        std::vector< std::vector <HoughVotesStats> > fullStats;

        for (int v = 0; v < test->voteClassCount(); v++){
            fullStats.push_back(std::vector<HoughVotesStats>());

            for(int i=0; i< test->imageCount(); i++){
                fullStats.back().push_back(HoughVotesStats(cv::Size(240,320),v));
            }
        }

        log << "full stats vector created" << std::endl;
        log << "image count: " << test->imageCount() << std::endl;

        for(int i=0; i<test->Count(); i++){
            test->getDataPoint(i,tmpstr,current);
            for(int t=0; t<forest->TreeCount(); t++)
            {
                if (leafIndicesPerTree[t][i]>0)
                    for(int v = 0; v < test->voteClassCount(); v++){

                        if (fullStats[v][test->getImageIdx(i)].Aggregate(current,forest->GetTree(t).GetNode(leafIndicesPerTree[t][i]).TrainingDataStatistics)){
                            std::cerr << "node: " << leafIndicesPerTree[t][i] << std::endl;
                        }

                    }
            }
        }

        log << "statistic aggregated" << std::endl;

        std::string filename;

        std::vector<bool> seen(test->imageCount(),false);
        std::vector<cv::Point2i> votes(test->voteClassCount());
        cv::Point2i p;

        std::ofstream *stream;

        for(int i=0; i<test->Count(); i++){
            if(!seen[test->getImageIdx(i)])
            {
                seen[test->getImageIdx(i)] = true;
                log << "i: " << i <<
                       " filename: " << test->imageIdx2Filename(test->getOriginalImageIdx(i)) <<
                       " imindex: "  << test->getImageIdx(i) << std::endl;

                test->getDataPointVote(i,votes);
                test->getDataPoint(i,filename,p);
                for(int v = 0; v < test->voteClassCount(); v++){
                    fullStats[v][test->getImageIdx(i)].setGT(votes[v]+p);
                    filename = "img_" + num2str<int>(i)+ "_" + num2str<int>(v);
                    stream = (std::ofstream *)&cache.openBinStream(filename);
                    fullStats[v][test->getImageIdx(i)].Serialize(*stream);
                    stream->close();
                    fullStats[v][test->getImageIdx(i)].Serialize(cache.base() + filename + std::string(".png"));
                }
            }
        }

        log << "results serialized" << std::endl;

    }catch(std::exception e){
        std::cerr << "exception caught: " << e.what() << std::endl;
        std::cerr.flush();
    }
}
