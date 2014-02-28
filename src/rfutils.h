#ifndef RFUTILS_H
#define RFUTILS_H

#include <Forest.h>
#include <Random.h>
#include "depthimagedb.h"
#include "classification/depthdb.h"
#include "localcache.h"
#include "string2number.hpp"
#include "classification/imagepixelstats.h"

using namespace MicrosoftResearch::Cambridge::Sherwood;

class RFUtils
{
public:

    template<class OutputDB,class OutputDBPClass>
    static void splitRandom(Random &random,
                            DepthFileBasedImageDB &db,
                            std::auto_ptr<OutputDB> &train,
                            std::auto_ptr<OutputDB> &test)
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
    static bool serializeLeafs(std::ostream &stream, Tree<F,S> &tree)
    {
        int count = 0;
        for(int i=0; i< tree.NodeCount(); i++){
            if (tree.GetNode(i).IsLeaf()){
                tree.GetNode(i).Serialize(stream);
                count++;
            }
        }
        std::cout << count << " leafs serialized" << std::endl;
    }

    template<class F,class S>
    static double testClassificationForest(Forest<F,S> &forest,
                                           S instance,
                                           DepthFileBasedImageDB &test, LocalCache &cache, bool save)
    {
        std::vector<std::vector<int> > leafIndicesPerTree;

        forest.Apply(test, leafIndicesPerTree);

        cache.log() << "forest applied" << std::endl;


        S stats = instance;
        std::vector<ImagePixelStats> perImageStatsImg(test.imageCount());
        std::vector<S> perImageStats(test.imageCount(),instance);
        cv::Point2i p;
        std::string stubname;

        for(int i=0; i<test.Count(); i++)
        {

            stats.Clear();
            test.getDataPoint(i,stubname,p);

            for(int t=0; t< forest.TreeCount(); t++)
            {


                if(leafIndicesPerTree[t][i]>0){

                    stats.Aggregate(forest.GetTree(t).GetNode(leafIndicesPerTree[t][i]).TrainingDataStatistics);

                    if(!forest.GetTree(t).GetNode(leafIndicesPerTree[t][i]).IsLeaf()){
                        cache.log() << "bad leafIndicesPerTree;" << std::endl;
                    }
                }
            }

            if(stats.SampleCount()>0){
                perImageStatsImg[test.getImageIdx(i)].Aggregate(p,stats);
                perImageStats[test.getImageIdx(i)].Aggregate(stats.ClassDecision());
            }

        }

        cache.log() << "statistics collected" << std::endl;

        double result = 0;
        ClassificationDB &testcl = dynamic_cast<ClassificationDB &>(test);
        std::vector<bool> seen(test.imageCount(),false);
        std::string filename;
        std::ofstream *stream;

        for(int i=0; i< seen.size(); i++){
            seen[i] = false;
        }
        cache.log() << "computing statistics for :" << test.imageCount() << std::endl;

        for(int i=0; i<test.Count(); i++)
        {
            if(!seen[test.getImageIdx(i)]){
                seen[test.getImageIdx(i)] = true;

                if(save){
                    filename = testcl.labelIndex2Name(testcl.getNumericalLabel(i)) +
                               num2str<int>(i);
                    stream = (std::ofstream *)&cache.openBinStream(filename);
                    perImageStatsImg[test.getImageIdx(i)].Serialize(*stream);
                    stream->close(); //not to have too many of them opened
                    perImageStatsImg[test.getImageIdx(i)].Serialize(cache.base() + filename+".png");
                }
                 if(testcl.getNumericalLabel(i)!= perImageStats[test.getImageIdx(i)].ClassDecision()){
                    result+=1;//increase error count;
                }
            }
        }

        cache.log() << "error computed" << std::endl;

        return result/test.imageCount();
    }

};

#endif // RFUTILS_H
