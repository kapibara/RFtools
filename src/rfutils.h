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
                            OutputDBPClass *train,
                            OutputDBPClass *test)
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
                                           DepthFileBasedImageDB &test, LocalCache &cache, bool saveBad)
    {
        std::vector<std::vector<int> > leafIndicesPerTree;

        forest.Apply(test, leafIndicesPerTree);

        std::cerr << "forest applied" << std::endl;

        S stats = instance;
        std::vector<ImagePixelStats> perImageStatsImg(test.imageCount());
        std::vector<S> perImageStats(test.imageCount(),instance);
        std::vector<cv::Point2i> badPixels(test.imageCount());
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
                        std::cerr << "bad leafIndicesPerTree;" << std::endl;
                    }
                }
            }

            if(stats.SampleCount()>0){
                perImageStatsImg[test.getImageIdx(i)].Aggregate(p,stats);
                perImageStats[test.getImageIdx(i)].Aggregate(stats.ClassDecision());
            }

        }

        std::cerr << "statistics collected" << std::endl;

        double result = 0;
        ClassificationDB &testcl = dynamic_cast<ClassificationDB &>(test);
        std::vector<bool> seen(test.imageCount(),false);
        std::string filename;
        std::ofstream stream;

        for(int i=0; i< seen.size(); i++){
            seen[i] = false;
        }
        std::cerr << "computing statistics for :" << test.imageCount() << std::endl;

        for(int i=0; i<test.Count(); i++)
        {
 //           std::cerr<< "image index" << test.getImageIdx(i) << std::endl;
            if(!seen[test.getImageIdx(i)]){
                seen[test.getImageIdx(i)] = true;

                if(saveBad){
                    filename = cache.base() +
                               testcl.labelIndex2Name(testcl.getNumericalLabel(i)) +
                               num2str<int>(i);
                    stream.open(filename.c_str(),std::ios_base::binary);
                    perImageStatsImg[test.getImageIdx(i)].Serialize(stream);
                    stream.close();
                    perImageStatsImg[test.getImageIdx(i)].Serialize(filename+".png");
                }
                 if(testcl.getNumericalLabel(i)!= perImageStats[test.getImageIdx(i)].ClassDecision()){
                    result+=1;//increase error count;
                }
            }
        }

        std::cerr << "error computed" << std::endl;

        return result/test.imageCount();
    }

};

#endif // RFUTILS_H
