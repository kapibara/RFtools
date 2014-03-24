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
#include "hough/pooledhoughtrainingcontext.h"
#include "nodedistributionimagestats.h"

#include <time.h>

//#define FEATURE_GENERATION

#define TRAIN_TEST_RANDOM

using namespace MicrosoftResearch::Cambridge::Sherwood;

int main(int argc, char **argv)
{
    std::cout << "starting the program" << std::endl;

    LocalCache cache("DepthHOUGH","/home/kuznetso/tmp");

    std::cout << "arg1: " << argv[1] << std::endl;

    if(!cache.init()){
        std::cerr << "failed to initialize temporary directory" << std::endl;
        exit(-1);
    }

    std::ostream &log = cache.log();


#ifndef FEATURE_GENERATION
#ifndef TRAIN_TEST_RANDOM
    // /home/kuznetso/tmp/DepthHOUGH/20_16_19_13/ --- full parameter set...
    std::string filename = "/home/kuznetso/tmp/DepthHOUGH/21_17_52_31/features";
    std::ifstream in(filename.c_str());
    log << "reading features from " <<  filename << std::endl;
    FeaturePool pool(in);
#endif
#endif

    try{
        DepthDBWithVotesImpl db;
        Random random;
        time_t start,end;
        ProgressStream progress(log,Verbose);
#ifdef FEATURE_GENERATION
        std::auto_ptr<Forest<DepthFeature, StubStats> > forest;
#else
        std::auto_ptr<Forest<DepthFeature, VotesStats> > forest;
#endif


        if (argc<2){
            std::cout << "exec <db file> <feature file>" << std::endl;
        }

        db.loadDB(argv[1]);

        log << "loading from: " << argv[1] << std::endl;
        log << "number of images: " << db.imageCount() << std::endl;
        log << "number of points: " << db.Count() << std::endl;
        log << "number of vote classes: " << (int)db.voteClassCount() << std::endl;

#ifndef FEATURE_GENERATION
        std::auto_ptr<DepthDBWithVotesSubindex> test;
        std::auto_ptr<DepthDBWithVotesSubindex> train;
        RFUtils::splitRandom<DepthDBWithVotesSubindex>(random,db,train,test,0.5);

        log << "train set size: " << train->Count() << std::endl;
        log << "test set size: " << test->Count() << std::endl;
#endif

        if (argc<3){

            DepthFeatureParameters featureParams;
            featureParams.uvlimit_ = 30;
            featureParams.zeroplane_ = 300;

            log << featureParams;

            Parameter<int> T(1, "No. of trees in the forest.");
            Parameter<int> D(10, "Maximum tree levels.");
            Parameter<int> F(500, "No. of candidate feature response functions per split node.");
            Parameter<int> L(10, "No. of candidate thresholds per feature response function.");
            Parameter<bool> verbose(true,"Enables verbose progress indication.");

            log << T << D << F << L << std::endl;

            TrainingParameters trainingParameters;
            trainingParameters.MaxDecisionLevels = D.value()-1;
            trainingParameters.NumberOfCandidateFeatures = F.value();
            trainingParameters.NumberOfCandidateThresholdsPerFeature = L.value();
            trainingParameters.NumberOfTrees = T.value();
            trainingParameters.Verbose = verbose.value();



#ifdef FEATURE_GENERATION
            DepthFeatureFactory factory(featureParams);
            std::ostream &features = cache.openBinStream("features");
            StubTrainingContext context(factory, trainingParameters, features);
#else
#ifndef TRAIN_TEST_RANDOM
            PooledHoughTrainingContext context(db.voteClassCount(),pool);
            std::ostream &featureOutput = cache.openBinStream("accomulatedFeatures");
            FeatureAccomulator accomulator(featureOutput,trainingParameters.NumberOfCandidateFeatures*trainingParameters.NumberOfCandidateThresholdsPerFeature);
            context.setFeatureAccomulator(&accomulator);
#else
            DepthFeatureFactory factory(featureParams);
            std::ostream &featureOutput = cache.openBinStream("accomulatedFeatures");
            FeatureAccomulator accomulator(featureOutput,trainingParameters.NumberOfCandidateFeatures*trainingParameters.NumberOfCandidateThresholdsPerFeature);
            HoughTrainingContext context(db.voteClassCount(),factory);
            context.setFeatureAccomulator(&accomulator);
#endif
#endif

            time(&start);
#ifdef FEATURE_GENERATION
            forest = ForestTrainer<DepthFeature, StubStats>::TrainForest (
                random, trainingParameters, context, db ,&progress);
#else
            forest = ForestTrainer<DepthFeature, VotesStats>::TrainForest (
                random, trainingParameters, context, *train ,&progress);

#endif
            time(&end);
            double dif = difftime (end,start);

            log << "time: " << dif << std::endl;
            log << "forest trained" << std::endl;

            std::ostream &out = cache.openBinStream("forest");
            forest->Serialize(out);

            log << "forest serialized" << std::endl;
        }
#ifndef FEATURE_GENERATION
        else{

            std::ifstream in(argv[2],std::ios_base::binary);

            forest = Forest<DepthFeature, VotesStats>::Deserialize(in);

            log << "forest deserialized" << std::endl;
        }

        std::vector<std::vector<int> > leafIndicesPerTree;
        forest->Apply(*test,leafIndicesPerTree,&progress);

        log << "forest applied" << std::endl;

        cv::Point2i current;
        std::string tmpstr;
//        int badNode = 1022;
        std::vector< std::vector <HoughVotesStats> > fullStats;
//        std::vector<NodeDistributionImagestats> nodeDist;

        for (int v = 0; v < test->voteClassCount(); v++){
            fullStats.push_back(std::vector<HoughVotesStats>());

            for(int i=0; i< test->imageCount(); i++){
                fullStats.back().push_back(HoughVotesStats(cv::Size(640,480),v));
            }
        }
/*
        for(int i=0; i< test->imageCount(); i++){
            nodeDist.push_back(NodeDistributionImagestats(cv::Size(480,640),badNode));
        }
*/
        log << "full stats vector created" << std::endl;
        log << "image count: " << test->imageCount() << std::endl;




        for(int i=0; i<test->Count(); i++){
            test->getDataPoint(i,tmpstr,current);
            std::cerr << "point: " << current << std::endl;
            for(int t=0; t<forest->TreeCount(); t++)
            {
                if (leafIndicesPerTree[t][i]>0)
//                    nodeDist[test->getImageIdx(i)].addPoint(current,leafIndicesPerTree[t][i]);
                    for(int v = 0; v < test->voteClassCount(); v++){

                        if (fullStats[v][test->getImageIdx(i)].Aggregate(current,forest->GetTree(t).GetNode(leafIndicesPerTree[t][i]).TrainingDataStatistics)){
                            std::cerr << "node: " << leafIndicesPerTree[t][i] << std::endl;
                        }

                    }
            }
        }

        log << "statistic aggregated" << std::endl;

        std::string filename;
/*        for(int i=0; i<nodeDist.size();i++){

            filename = cache.base() + "pixDist" + num2str<int>(i) + ".png";
            std::cerr << "serializing: " << filename << std::endl;
            nodeDist[i].Serialize(filename);
        }

        log << "pixDist serialized" << std::endl; */

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
#endif
    }catch(std::exception e){
        std::cerr << "exception caught: " << e.what() << std::endl;
        std::cerr.flush();
    }
}
