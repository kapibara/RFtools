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

    template<class OutputDB,class OutputDBPClass>
    static void splitRandom(Random &random,
                            DepthFileBasedImageDB &db,
                            std::auto_ptr<OutputDBPClass> &train,
                            std::auto_ptr<OutputDBPClass> &test)
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

        train.reset(new OutputDB(db,train_ind));
        test.reset(new OutputDB(db,test_ind));
    }

    template<class F,class S>
    static double testClassificationForest(Forest<F, S> &forest,
                                           S instance,
                                           DepthFileBasedImageDB &test)
    {
        std::vector<std::vector<int> > leafIndicesPerTree;

        forest.Apply(test, leafIndicesPerTree);

        std::cerr << "forest applied" << std::endl;

        S stats = instance;
        std::vector<S> perImageStats(test.imageCount(),instance);

        for(int i=0; i<test.Count(); i++)
        {

            stats.Clear();

            for(int t=0; t< forest.TreeCount(); t++)
            {
                if(leafIndicesPerTree[t][i]>0)

                    stats.Aggregate(forest.GetTree(t).GetNode(leafIndicesPerTree[t][i]).TrainingDataStatistics);
            }

            perImageStats[test.getImageIdx(i)].Aggregate(stats.ClassDecision());

        }

        std::cerr << "statistics collected" << std::endl;

        double result = 0;
        ClassificationDB &testcl = dynamic_cast<ClassificationDB &>(test);
        std::vector<bool> seen(test.imageCount(),false);

        for(int i=0; i< seen.size(); i++){
            seen[i] = false;
        }
        std::cerr << "computing statistics for :" << test.imageCount() << std::endl;

        for(int i=0; i<test.Count(); i++)
        {
 //           std::cerr<< "image index" << test.getImageIdx(i) << std::endl;
            if(!seen[test.getImageIdx(i)]){
                std::cerr << "not seen:" << test.getImageIdx(i) << std::endl;
                if(testcl.getNumericalLabel(i)!= perImageStats[test.getImageIdx(i)].ClassDecision()){
                    std::cerr << "false decision" << std::endl;
                    std::cerr << "gt: " << testcl.getNumericalLabel(i)
                              << "per im stats: " <<  (int)perImageStats[test.getImageIdx(i)].ClassDecision()
                              << std::endl;
                   result+=1;//increase error count;
                }
                seen[test.getImageIdx(i)] = true;
            }
        }

        std::cerr << "error computed" << std::endl;

        return result/test.imageCount();
    }
};

#endif // RFUTILS_H
