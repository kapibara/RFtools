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
#include "TrainingParameters.h"
#include "hough/houghtrainingcontext.h"
#include "string2number.hpp"

using namespace MicrosoftResearch::Cambridge::Sherwood;

int main(int argc, char **argv)
{


    if (argc<2){
        std::cout << "exec <db file>" << std::endl;
    }

    std::cout << "starting the program" << std::endl;

    try{
        DepthDBWithVotesImpl db;
        Random random;
        std::vector<cv::Point2i> p;


        db.loadDB(argv[1]);

        std::cout << "number of images: " << db.imageCount() << std::endl;

        std::cout << "number of points:" << db.Count() << std::endl;
        std::cout << "number of vote classes: " << (int)db.voteClassCount() << std::endl;

        Parameter<int> T(1, "No. of trees in the forest.");
        Parameter<int> D(5, "Maximum tree levels.");
        Parameter<int> F(100, "No. of candidate feature response functions per split node.");
        Parameter<int> L(10, "No. of candidate thresholds per feature response function.");
        Parameter<bool> verbose(true,"Enables verbose progress indication.");

        TrainingParameters trainingParameters;
        trainingParameters.MaxDecisionLevels = D.value()-1;
        trainingParameters.NumberOfCandidateFeatures = F.value();
        trainingParameters.NumberOfCandidateThresholdsPerFeature = L.value();
        trainingParameters.NumberOfTrees = T.value();
        trainingParameters.Verbose = verbose.value();

        DepthFeatureFactory factory;
        HoughTrainingContext<DepthFeature> context(db.voteClassCount(),factory);

        std::auto_ptr<Forest<DepthFeature, VotesStats> > forest;
        forest = ForestTrainer<DepthFeature, VotesStats>::TrainForest (
                random, trainingParameters, context, db );

        std::cout << "forest trained" << std::endl;

        std::ofstream out("hforest",std::ios_base::binary);

        forest->Serialize(out);
        std::cout << "forest serialized" << std::endl;

        out.close();

        std::vector<std::vector<int> > leafIndicesPerTree;
        forest->Apply(db,leafIndicesPerTree);

        std::cout << "apply forest" << std::endl;



        HoughVotesStats fullStats(cv::Size(240,320),0);
        cv::Point2i current;
        std::string tmpstr;
        DepthDBWithVotesImpl::fileindex_type file = db.getImageIdx(0);

        std::cout << "file: " << db.imageIdx2Filename(file) <<std::endl;

        for(int i=0; i<db.Count(); i++){

            if(db.getImageIdx(i)!=file){
                break;
            }

            db.getDataPoint(i,tmpstr,current);

            for(int t=0; t<forest->TreeCount(); t++){
                if (leafIndicesPerTree[t][i]>0)
                    fullStats.Aggregate(current,forest->GetTree(t).GetNode(leafIndicesPerTree[t][i]).TrainingDataStatistics);
            }
        }

        std::cout << "statistic aggregated" << std::endl;

        std::ofstream outstats("fullstats",std::ios_base::binary);
        fullStats.Serialize(outstats);
        outstats.close();



    }catch(std::exception e){
        std::cout << "exception caught" << e.what() << std::endl;
    }
}
