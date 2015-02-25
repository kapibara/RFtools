#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>

#include "regression/depthdbreg.h"
#include "regression/regtrainingcontext.h"
#include "regression/aggregatedleafs.h"

#include "configuration.h"
#include "subsampler.h"
#include "copyfile.h"
#include "Tree.h"
#include "Random.h"
#include "Forest.h"
#include "ForestTrainer.h"
#include "depthfeature.h"
#include "serialization.h"

#define VoteDim 3
typedef float VoteType;

typedef VotesStatsT<VoteType,VoteDim> Stats;
typedef RegTrainingContext<Stats,FullDepthFeatureFactory> Context;
typedef DepthDBWithVotesSImpl<VoteType,VoteDim> DB;

void collectVotesPerLeafCounts(Tree<DepthFeature,Stats> &tree, std::vector<double> &variance, std::vector<std::vector<int> > &result, int votesCount)
{
    std::vector<int> tmp;
    tmp.resize(votesCount+1,0);
    int tmpindex;

    for(int i=0; i< tree.NodeCount();i++){
        if(tree.GetNode(i).IsLeaf()){
            tmpindex = 0;
            for(Stats::const_iterator itor = tree.GetNode(i).TrainingDataStatistics.begin(); itor != tree.GetNode(i).TrainingDataStatistics.end(); itor++){
                tmp[tmpindex++] =(*itor).ListSize();
            }
            tmp[tmpindex]=tree.GetNode(i).TrainingDataStatistics.Count();
            result.push_back(tmp);
            variance.push_back(tree.GetNode(i).TrainingDataStatistics.VoteVariance());
        }
    }
}

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

    DB  db;
    time_t start,end;
    Random random;

    if (config.useSubsampler()){
        log << "using subsampler - rate: " << config.subsamplerRate() << std::endl;
        db.setSubsampler(new RandomSubsampler(config.subsamplerRate(),random));
    }

    db.setCalibParam(config.calibration());
    log << "calibration paramerers: " << config.calibration() << std::endl;
    db.loadDB(config.databaseFile(),config.databaseHasHeader());

    std::cerr << "db votes count: " << (int)db.voteClassCount() << std::endl;

    ProgressStream progress(log,Verbose);

    std::auto_ptr<Forest<DepthFeature, Stats> > forest;

    log << "load a forest from: " << config.forestFile().c_str() << std::endl;

    std::ifstream in(config.forestFile().c_str(),std::ios_base::binary);

    forest = Forest<DepthFeature, Stats>::Deserialize(in);

    log << "forest deserialized" << std::endl;

    log << "depth: " << forest->GetTree(0).DecisionLevels() << " tree count: " << forest->TreeCount() << std::endl;

    log << "testing" << std::endl;

    std::vector<std::vector<int> > leafIndicesPerTree;

    time(&start);
    forest->Apply(db,leafIndicesPerTree,&progress);
    time(&end);

    log << "forest applied" << std::endl;

    double dif = difftime (end,start);

    log << "time: " << dif << std::endl;

    AggregatedLeafs<DepthFeature,Stats,VoteType,VoteDim> aggLeafs;

    mean_shift::MeanShift mshift;
    mshift.setRadius(config.meanShiftR());
    mshift.setMaxIter(config.meanShiftMaxIter());
    mshift.setMaxNeigboursCount(config.maxNN());

    log << "mean shift radius: " << config.meanShiftR() << std::endl;
    log << "mean shift max iter: " << config.meanShiftMaxIter() << std::endl;
    log << "mean shift max neighbour count: " << config.meanShiftMaxIter() << std::endl;
    log << "small vote weights threashold: " << config.smallWeightThr() << std::endl;

    aggLeafs.SetVarThreashold(config.nodeVarThr());
    aggLeafs.SetNodeSizeThreashold(config.nodeSizeThr());
    aggLeafs.SetSmallWeightsThreashold(config.smallWeightThr());
    aggLeafs.Build(*forest,mshift,db.voteClassCount());

    log << "collecting forest statistics " << std::endl;
    std::vector<std::vector<int> > votesCount;
    std::vector<double> variance;

    for(int t=0; t< forest->TreeCount(); t++){
        collectVotesPerLeafCounts(forest->GetTree(t), variance, votesCount, db.voteClassCount());
    }

    std::ostream &forestStats = cache.openBinStream("forestStats");
    int votesCountSize = votesCount.size();
    forestStats.write((const char *)&votesCountSize,sizeof(votesCountSize));

    for(int i=0; i<votesCountSize; i++){
        serializeVector<int>(forestStats,votesCount[i]);
    }

     std::ostream &vovariance = cache.openBinStream("vovariance");
     serializeVector<double>(vovariance,variance);


    log << "aggregating votes for the forst built " << std::endl;

    std::ostream &leafs = cache.openBinStream("aggLeafs");

    aggLeafs.Serialize(leafs);

    std::vector<VotesAggregator<VoteType,VoteDim> > perImageVotes;
    std::vector<std::vector<cv::Vec<VoteType,VoteDim> > > perImagePoints;

    VotesAggregator<VoteType,VoteDim> tmp(db.voteClassCount());
    perImageVotes.assign(db.imageCount(),tmp);
    perImagePoints.resize(db.imageCount());
    cv::Vec<VoteType,VoteDim> coord3D;
    DB::fileindex_type imindex;

    log << "aggregating votes accross the images" << std::endl;

    for(int i=0; i<db.Count(); i++){
        db.getDataPoint(i,coord3D);

        imindex = db.getImageIdx(i);
        for(int t=0; t<forest->TreeCount(); t++){
            perImageVotes[imindex].AddVotes(aggLeafs.get(t,leafIndicesPerTree[t][i]),coord3D);
        }
        perImagePoints[imindex].push_back(coord3D);
    }

    cache.closeAllStreams();

    std::ofstream *out;
    GroundTruthDecorator<VoteType,VoteDim> deco;

    for(int i=0; i<perImageVotes.size(); i++){

        deco.SetGT(db.getGT(i));
        deco.SetAggregator(perImageVotes[i]);

        out = new std::ofstream((cache.base() + "aggvotes" + num2str(i)).c_str(),std::ios_base::binary);

        deco.Serialize(*out);
        serializeVoteVector<VoteType,VoteDim>(perImagePoints[i],*out);

        out->close();
        delete out;


    }

    log << "done"<< std::endl;
}
