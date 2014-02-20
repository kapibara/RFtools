#ifndef RFUTILS_H
#define RFUTILS_H

#include <Forest.h>
#include <Random.h>
#include "depthimagedb.h"
#include "classification/depthdb.h"

using namespace MicrosoftResearch::Cambridge::Sherwood;

class RFUtils
{
public:

    template<class OutputDB>
    static std::pair<std::auto_ptr<OutputDB>, std::auto_ptr<OutputDB> > splitRandom(Random &random, DepthFileBasedImageDB &db)
    {
        std::vector<DepthFileBasedImageDB::index_type> train_ind;
        std::vector<DepthFileBasedImageDB::index_type> test_ind;

        DepthFileBasedImageDB::index_type pi=0;
        DepthFileBasedImageDB::fileindex_type fi = 0;

        for(int i=0; i< db.imageCount(); i++){
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

        std::auto_ptr<OutputDB> train(new OutputDB(db,train_ind));
        std::auto_ptr<OutputDB> test(new OutputDB(db,train_ind));

        return std::make_pair(train,test);
    }

    template<class F,class S>
    static double testClassificationForest(Forest<F, S> &forest, S instance, DepthFileBasedImageDB &test)
    {
        std::vector<std::vector<int> > leafIndicesPerTree;

        forest.Apply(test, leafIndicesPerTree);

        S stats = instance;
        std::vector<S> perImageStats(test.imageCount(),instance);


        for(int i=0; i<test.Count(); i++)
        {

            stats.Clear();

            for(int t=0; t< forest.TreeCount(); t++)
            {
                if(leafIndicesPerTree[t][i]!=-1)

                    stats.Aggregate(forest->GetTree(t).GetNode(leafIndicesPerTree[t][i]).TrainingDataStatistics);
            }
            std::cerr << "aggregate data for image " << std::endl;
            std::cerr.flush();

            //add statistics to the corresponding image statistics
            perImageStats[test.getImageIdx(i)].Aggregate(stats);

        }
    }
};

#endif // RFUTILS_H
