#ifndef RFUTILS_H
#define RFUTILS_H

#include <Forest.h>
#include <Random.h>
#include "depthimagedb.h"
#include "classification/depthdb.h"
#include "localcache.h"
#include "string2number.hpp"

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
    static double testClassificationForest(Forest<F,S> &forest,
                                           S instance,
                                           DepthFileBasedImageDB &test, LocalCache &cache, bool saveBad)
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
        std::string filename;
        std::ofstream stream;

        for(int i=0; i<test.Count(); i++)
        {
            if(!seen[test.getImageIdx(i)]){
                seen[test.getImageIdx(i)] = true;
                std::cout << "i: " << i << " num label: " << testcl.getNumericalLabel(i)
                          << "decision: " << (int) perImageStats[test.getImageIdx(i)].ClassDecision() << std::endl;
                if(testcl.getNumericalLabel(i)!= perImageStats[test.getImageIdx(i)].ClassDecision()){
                    result+=1;//increase error count;

                    if(saveBad){
                        stream.open(cache.base() +
                                    testcl.labelIndex2Name(testcl.getNumericalLabel(i)) +
                                    num2str<int>(i),std::ios_base::binary);
                        perImageStats[test.getImageIdx(i)].Serialize(stream);
                        stream.close();
                    }
                }
            }
        }

        std::cerr << "error computed" << std::endl;

        return result/test.imageCount();
    }

};

#endif // RFUTILS_H
