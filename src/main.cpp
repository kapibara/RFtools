
#include <iostream>
#include <fstream>
#include <ctime>

#ifdef WIN32
#include <windows.h>
#endif

#include "depthdb.h"
#include "trainingcontext.h"

#include "TrainingParameters.h"
#include "Forest.h"
#include "ForestTrainer.h"
#include "parameter.h"
#include "inmemdb.h"
#include "imagepixelstats.h"

using namespace MicrosoftResearch::Cambridge::Sherwood;

typedef std::vector<unsigned int>::size_type DataPointIndex;

int main(int argc, char **argv)
{
    std::cout << "starting training ... " << std::endl;

    if (argc<2){
        std::cout << "exec <text file>" << std::endl;
        exit(-1);
    }
/*
    std::cout << "testing partitioning" << std::endl;

    std::vector<float> keys;
    std::vector<unsigned int> indices;
    float invalid = 100;

    for(int i=0; i<5; i++){
        keys.push_back(invalid);
        indices.push_back(i);
    }

    keys[3] = invalid;

    for (int i=0; i<5; i++){
        std::cout << keys[i] << " ";
    }

    std::pair<DataPointIndex,DataPointIndex> result = Tree<int, int>::Partition(keys, indices, 0, 5 , 1.0/3.0, invalid);

    std::cout << "result: " << result.first << ";" << result.second << std::endl;

    for (int i=0; i<5; i++){
        std::cout << (float)keys[i] << " ";
    }

    std::cout << std::endl;
    std::cout.flush();

    char in;

    std::cin >> in;

    return 0;*/

    try{
        DepthDB db(argv[1]);
        Random  random;
        std::vector<ClassificationDB::index_type> train_ind;
        std::vector<ClassificationDB::index_type> test_ind;
        ClassificationDB::index_type pi=0;
        ClassificationDB::fileindex_type fi = 0;

        for(int i=0; i<db.imageCount();i++){
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
        DepthDBSubindex test(db,test_ind);

        std::cout << "Train samples:" << train.Count() << std::endl;
        std::cout << "Test samples:" << test.Count() << std::endl;

        std::cerr << "db loaded ... " << std::endl;
        std::cerr.flush();

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

        std::cerr <<"class count: "  << db.classCount() << " element count: " << db.Count()<<std::endl;

        DepthFeatureFactory factory(random);
        TrainingContext<DepthFeature, ClassStats> context(db.classCount(),factory);


        std::cerr << "start forest training ... " << std::endl;
        std::cerr.flush();

        std::auto_ptr<Forest<DepthFeature, ClassStats> > forest = ForestTrainer<DepthFeature, ClassStats>::TrainForest (
                random, trainingParameters, context, train );

        std::cerr << "Forest trained: " << forest->GetTree(0).NodeCount() << std::endl;
        std::cerr.flush();

        std::ofstream out("testout");
        forest->Serialize(out);

        std::cerr << "Forest saved" << std::endl;
        std::cerr.flush();


        std::cout << test.Count()-1 << std::endl;
        //last image at test
        ClassificationDB::fileindex_type last_image = test.getImageIdx(test.Count()-1);

        std::cerr << "Last image stored: " << db.imgIdx2Name(last_image) << std::endl;

        std::vector<std::vector<int> > leafIndicesPerTree;

        try{

            forest->Apply(test, leafIndicesPerTree);
        }
        catch(std::exception &e){
            std::cerr << "exception caught:" << e.what() << std::endl;
            std::cerr.flush();
        }

        std::cout << "Forest applied on the test set" << std::endl;

        ClassStats clStatsPixel;
        std::vector<ClassStats> clStatsImage(db.imageCount());
        ImagePixelStats stats1;
        std::ostringstream s;
        cv::Mat stub;
        cv::Point2i xy;

        for(int i=0; i<test.Count(); i++)
        {
            //clStats for a pixel
            clStatsPixel.Clear();

            for(int t=0; t< forest->TreeCount(); t++)
            {
                clStatsPixel.Aggregate(forest->GetTree(t).GetNode(leafIndicesPerTree[t][i]).TrainingDataStatistics);
            }

            //add statistics to the corresponding image statistics
            clStatsImage[test.getImageIdx(i)].Aggregate(clStatsPixel);

            if(test.getImageIdx(i)==last_image){
                test.getDataPoint(i,stub,xy);
                stats1.Aggregate(xy,clStatsPixel);
            }

        }

        clStatsImage[1].SerializeChar(s);

        std::ofstream result10("out5.txt");
        stats1.Serialize("testimg.png");
        stats1.Serialize(result10);
        result10.close();
        std::ofstream result("out.txt");

        for(int i=0; i< db.imageCount(); i++){
            clStatsImage[i].SerializeChar(result);
        }

        std::cout << "Statistics computed" << std::endl;

        result.close();

        std::cout.flush();

    }catch(std::exception &e){
        std::cerr << "exception caught" << std::endl;
        std::cerr.flush();
    }
}
