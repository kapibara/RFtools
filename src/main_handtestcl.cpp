
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>

#include "ProgressStream.h"
#include "Forest.h"
#include "TrainingParameters.h"

#include "perpixelclass/perpixelclassificationdb.h"
#include "classification/cltrainingcontext.h"
#include "classification/classstats.h"
#include "classification/imagepixelstats.h"
#include "parallel/paralleltreetrainer.h"

#include "depthfeature.h"
#include "featureaccomulator.h"
#include "configuration.h"
#include "localcache.h"
#include "copyfile.h"

#include "subsampler.h"

using namespace MicrosoftResearch::Cambridge::Sherwood;

int main(int argc, char **argv)
{
    std::cout << "config: " << argv[1] << std::endl;
    std::cout << "reading config data" << std::endl;

    std::ifstream configin(argv[1]);
    Configuration config(configin);
    configin.close();

    LocalCache cache(config.cacheFolderName(),"/home/kuznetso/tmp/");
    if(!cache.init()){
        std::cerr << "failed to initialize temporary directory" << std::endl;
        exit(-1);
    }

    std::ostream &log = cache.log();

    log << "copying config file" << std::endl;
    copyfile(argv[1],cache.base() + "config.xml");

    PerPixelClassificationDB db;
    Random random;

    if (config.useSubsampler()){
        log << "using subsampler - rate: " << config.subsamplerRate() << std::endl;
        db.setSubsampler(new RandomSubsampler(config.subsamplerRate(),random));
    }

    db.loadDB(config.databaseFile());

    log << "number of classes: " << db.classCount() << std::endl;

    ProgressStream progress(log,Verbose);

    std::auto_ptr<Forest<DepthFeature, ClassStats> > forest;

    log << "load a forest from: " << config.forestFile().c_str() << std::endl;

    std::ifstream in(config.forestFile().c_str(),std::ios_base::binary);

    forest = Forest<DepthFeature, ClassStats>::Deserialize(in);

    log << "forest deserialized" << std::endl;

    log << "depth: " << forest->GetTree(0).DecisionLevels() << " tree count: " << forest->TreeCount() << std::endl;

    log << "testing" << std::endl;

    std::vector<std::vector<int> > leafIndicesPerTree;
    time_t start,end;

    time(&start);
    forest->Apply(db,leafIndicesPerTree,&progress);
    time(&end);

    log << "forest applied" << std::endl;

    double dif = difftime (end,start);

    log << "time: " << dif << std::endl;

    log << "aggregating class labels" << std::endl;

    std::vector<ImagePixelStats> imstats;
    ImagePixelStats tmp(db.classCount());
    PerPixelClassificationDB::fileindex_type imindex;
    std::string s;
    cv::Point2i p;

    imstats.resize(db.imageCount(),tmp);

    for(int i = 0; i<db.Count(); i++){
        imindex = db.getImageIdx(i);
        db.getDataPoint(i,s,p);
        for(int t=0; t<1; t++){
            imstats[imindex].Aggregate(p,forest->GetTree(t).GetNode(leafIndicesPerTree[t][i]).TrainingDataStatistics);
        }
    }

    log << "statistics aggregated" << std::endl;
    std::ifstream *ifstream;

    for(int i=0; i< imstats.size(); i++){
        imstats[i].Serialize(cache.base() + "result" + num2str(i) + ".png");
    }

    log << "results serialized" << std::endl;
}
