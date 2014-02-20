
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
//#include "depthfeature.h"
#include "classification/classstats.h"
#include "classification/imagepixelstats.h"

#include <climits>
using namespace MicrosoftResearch::Cambridge::Sherwood;

int main(int argc, char **argv)
{
if (argc<2){
    std::cout << "exec <db file>" << std::endl;
}

try{
   std::auto_ptr<Forest<DepthFeature, ClassStats> > forest;

   DepthDBClassImage db;
   DepthFileBasedImageDB *test;
   Random  random;

   db.loadDB(argv[1]);

   std::cout << "DB: "<< db.classCount() << std::endl;

   if (argc==2){
        std::cout << "starting training ... " << std::endl;
        std::vector<DepthImageDB::index_type> train_ind;
        std::vector<DepthImageDB::index_type> test_ind;

        DepthImageDB::index_type pi=0;
        DepthFileBasedImageDB::fileindex_type fi = 0;

        for(int i=0; i<db.imageCount(); i++){
            if(random.NextDouble()<0.5){
                fi = db.getImageIdx(pi);
                while(db.getImageIdx(pi) == fi) //optimal eval expected
                {
                    train_ind.push_back(pi); pi++;
                    if (pi>= db.Count()) break;
                }
            }
            else {
                fi = db.getImageIdx(pi);
                while(db.getImageIdx(pi) == fi ) //optimal eval expected
                {
                    test_ind.push_back(pi); pi++;
                    if (pi>= db.Count()) break;
                }
            }
        }

        DepthDBSubindex train(db,train_ind);
        test = new DepthDBSubindex(db,test_ind);

        std::cout << "Train samples:" << train.Count() << std::endl;
        std::cout << "Test samples:" << test->Count() << std::endl;

        std::cerr << "db loaded ... " << std::endl;
        std::cerr.flush();

        Parameter<int> T(1, "No. of trees in the forest.");
        Parameter<int> D(2, "Maximum tree levels.");
        Parameter<int> F(100, "No. of candidate feature response functions per split node.");
        Parameter<int> L(10, "No. of candidate thresholds per feature response function.");
        Parameter<bool> verbose(true,"Enables verbose progress indication.");

        TrainingParameters trainingParameters;
        trainingParameters.MaxDecisionLevels = D.value()-1;
        trainingParameters.NumberOfCandidateFeatures = F.value();
        trainingParameters.NumberOfCandidateThresholdsPerFeature = L.value();
        trainingParameters.NumberOfTrees = T.value();
        trainingParameters.Verbose = verbose.value();

        std::cerr <<"class count: "  << db.classCount() << " element count: " << db.Count()<<std::endl;

        DepthFeatureFactory factory;
        ClTrainingContext<DepthFeature, ClassStats> context(db.classCount(),factory);

        std::cerr << "start forest training ... " << std::endl;
        std::cerr.flush();

        forest = ForestTrainer<DepthFeature, ClassStats>::TrainForest (
                random, trainingParameters, context, train );

        std::cerr << "Forest trained: " << forest->GetTree(0).NodeCount() << std::endl;
        std::cerr.flush();

        std::ofstream out("testout");
        forest->Serialize(out);

        std::cerr << "Forest saved" << std::endl;
        std::cerr.flush();

    }
    else if (argc>2){
        std::cout << "loading the forest ... " << std::endl;

        std::ifstream in("testout");
        forest = Forest<DepthFeature, ClassStats>::Deserialize(in);

        std::cerr << "Forest deserialized" << std::endl;
        std::cerr.flush();

        test = &db;
    }

    std::vector<std::vector<int> > leafIndicesPerTree;
    try{

        forest->Apply(*test, leafIndicesPerTree);
    }
    catch(std::exception &e){
        std::cerr << "exception caught:" << e.what() << std::endl;
        std::cerr.flush();
    }

    std::cerr << "Forest applied on the test set" << std::endl;
    std::cerr.flush();

    ClassStats clStatsPixel(db.classCount());
    std::vector<ClassStats> clStatsImage(test->imageCount(),ClassStats(db.classCount()));


    for(int i=0; i<test->Count(); i++)
    {

        //clStats for a pixel
        clStatsPixel.Clear();

        std::cerr << "aggregate data from forest for " << i << "th sample" << std::endl;

        for(int t=0; t< forest->TreeCount(); t++)
        {
            std::cerr << "leafIndicesPerTree[t][i]" <<leafIndicesPerTree[t][i] << std::endl;
            if(leafIndicesPerTree[t][i]!=-1)

                clStatsPixel.Aggregate(forest->GetTree(t).GetNode(leafIndicesPerTree[t][i]).TrainingDataStatistics);
        }

        std::cerr << "classCount_ " << (int)clStatsPixel.ClassCount() << std::endl;
        std::cerr << "aggregate data for image " << std::endl;
        std::cerr.flush();

        if (clStatsPixel.ClassCount()>0){
            std::cerr << "test->getImageIdx(i)" << test->getImageIdx(i) << std::endl;
            std::cerr << "class decision: " << (int) (clStatsPixel.ClassDecision()) << std::endl;
        //add statistics to the corresponding image statistics
//            clStatsImage[test->getImageIdx(i)].Aggregate(clStatsPixel.ClassDecision());
        }else
            std::cerr << "pixel cannot be classified" << std::endl;

    }

    std::cout << "Statistics computed" << std::endl;
    std::cout.flush();

    }catch(std::exception &e){
        std::cerr << "exception caught 2: " << e.what() << std::endl;
        std::cerr.flush();
    }
}
