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
#include "hough/houghtrainingcontext.h"
#include "classification/classstats.h"
#include "string2number.hpp"
#include "rfutils.h"

using namespace MicrosoftResearch::Cambridge::Sherwood;

int main(int argc, char **argv)
{

    std::cout << "starting the program" << std::endl;

    LocalCache cache("DepthHOUGH","/home/kuznetso/tmp");

    if(!cache.init()){
        std::cerr << "failed to initialize temporary directory" << std::endl;
        exit(-1);
    }

    std::ostream &log = cache.log();

    try{
        DepthDBWithVotesImpl db;
        Random random;
        ProgressStream progress(log,Verbose);
        std::auto_ptr<Forest<DepthFeature, VotesStats> > forest;
        std::auto_ptr<DepthDBWithVotesSubindex> test;
        std::auto_ptr<DepthDBWithVotesSubindex> train;

        if (argc<2){
            std::cout << "exec <db file>" << std::endl;
        }

        db.loadDB(argv[1]);

        log << "number of images: " << db.imageCount() << std::endl;
        log << "number of points: " << db.Count() << std::endl;
        log << "number of vote classes: " << (int)db.voteClassCount() << std::endl;

        RFUtils::splitRandom<DepthDBWithVotesSubindex>(random,db,train,test);

        log << "train set size: " << train->Count() << std::endl;
        log << "test set size: " << test->Count() << std::endl;

        if (argc<3){

            DepthFeatureParameters featureParams;
            featureParams.uvlimit_ = 30;
            featureParams.zeroplane_ = 300;

            log << featureParams;

            Parameter<int> T(1, "No. of trees in the forest.");
            Parameter<int> D(8, "Maximum tree levels.");
            Parameter<int> F(300, "No. of candidate feature response functions per split node.");
            Parameter<int> L(20, "No. of candidate thresholds per feature response function.");
            Parameter<bool> verbose(true,"Enables verbose progress indication.");

            TrainingParameters trainingParameters;
            trainingParameters.MaxDecisionLevels = D.value()-1;
            trainingParameters.NumberOfCandidateFeatures = F.value();
            trainingParameters.NumberOfCandidateThresholdsPerFeature = L.value();
            trainingParameters.NumberOfTrees = T.value();
            trainingParameters.Verbose = verbose.value();

            DepthFeatureFactory factory(featureParams);
            HoughTrainingContext<DepthFeature> context(train->voteClassCount(),factory);

            forest = ForestTrainer<DepthFeature, VotesStats>::TrainForest (
                random, trainingParameters, context, db ,&progress);

            log << "forest trained" << std::endl;

            std::ostream &out = cache.openBinStream("forest");
            forest->Serialize(out);

            log << "forest serialized" << std::endl;
        }else{

            std::ifstream in(argv[2],std::ios_base::binary);
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

                        fullStats[v][test->getImageIdx(i)].Aggregate(current,forest->GetTree(t).GetNode(leafIndicesPerTree[t][i]).TrainingDataStatistics);

                    }
            }
        }

        log << "statistic aggregated" << std::endl;

        std::vector<bool> seen(test->imageCount(),false);
        std::vector<cv::Point2i> votes;
        cv::Point2i p;
        std::string filename;
        std::ofstream *stream;

        for(int i=0; i<test->Count(); i++){
            if(!seen[test->getImageIdx(i)])
            {
                seen[test->getImageIdx(i)] = true;
                log << "i: " << i <<
                       "filename: " << test->imageIdx2Filename(test->getOriginalImageIdx(i)) <<
                       "imindex: "  << std::endl;

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
                votes.clear();
            }
        }

        log << "results serialized" << std::endl;


    }catch(std::exception e){
        std::cerr << "exception caught" << e.what() << std::endl;
        std::cerr.flush();
    }
}
