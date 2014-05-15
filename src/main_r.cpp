#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "Tree.h"
#include "Random.h"
#include "Forest.h"
#include "ForestTrainer.h"
#include "depthfeature.h"
#include "parameter.h"
#include "localcache.h"
#include "TrainingParameters.h"
#include "stubtrainingcontext.h"
#include "regression/votesstatst.h"
#include "regression/depthdbreg.h"
#include "regression/regtrainingcontext.h"
#include "regression/votesaggregator.h"
#include "regression/aggregatedleafs.h"
#include "stubstats.h"
#include "string2number.hpp"
#include "rfutils.h"
#include "featurepool.h"
#include "nodedistributionimagestats.h"
#include "configuration.h"
#include "copyfile.h"

#include <time.h>

#define VoteDim 3
typedef float VoteType;

typedef VotesStatsT<VoteType,VoteDim> Stats;

void computeError(const std::vector<cv::Vec<VoteType,VoteDim> > &prediction,
                  const std::vector<cv::Vec<VoteType,VoteDim> > &gt,
                  std::vector<cv::Vec<VoteType,VoteDim> > &error)
{
    for(int i=0; i<gt.size(); i++){
        for(int j=0; j<VoteDim; j++){
            error[i][j] = std::abs(prediction[i][j]-gt[i][j]);
        }
    }
}

void meanShiftVotesAggregation(const Configuration &config,
                               LocalCache &cache,
                               Tree<DepthFeature, Stats>  &tree,
                               std::vector<VotesAggregator<VoteType,VoteDim> > &aggLeafs)
{
    mean_shift::MeanShift mshift;
    mshift.setRadius(config.meanShiftR());
    mshift.setMaxIter(config.meanShiftMaxIter());
    mshift.setMaxNeigboursCount(config.maxNN());

    std::ostream &log = cache.log();

    log << "mean shift radius: " << config.meanShiftR() << std::endl;
    log << "mean shift max iter: " << config.meanShiftMaxIter() << std::endl;
    log << "mean shift max neighbour count: " << config.meanShiftMaxIter() << std::endl;
    log << "small vote weights threashold: " << config.smallWeightThr() << std::endl;

    for(int i=0; i< tree.NodeCount();i++){
        if(tree.GetNode(i).IsLeaf()){
            if(!config.discardHighVar() ||
                tree.GetNode(i).TrainingDataStatistics.VoteVariance() < config.nodeVarThr()){
                aggLeafs[i].AggregateVotes(tree.GetNode(i).TrainingDataStatistics,mshift);
                //remove 'noise' votes
                aggLeafs[i].FilterSmallWeights(config.smallWeightThr());
            }else{
                log << "skipping node: " << i << std::endl;
            }
        }
    }
}

ITrainingContext<DepthFeature,Stats>  *createTrainingContext(const Configuration &config,
                                                                  LocalCache &cache,
                                                                  uchar classCount)
{
    ITrainingContext<DepthFeature,Stats> *context = 0;
    FeatureAccomulator *accomulator = 0;
    TrainingParameters tp = config.forestParameters();

    if(config.serializeInfo()){
        std::ostream &featureOutput = cache.openBinStream("accomulatedFeatures");
        accomulator = new FeatureAccomulator(featureOutput,tp.NumberOfCandidateFeatures*tp.NumberOfCandidateThresholdsPerFeature);
    }

    switch(config.factoryType()){
    case Configuration::FeaturePool:
    {
        cache.log() << "feature factory: FeaturePool" << std::endl;
        std::ifstream in(config.featuresFile().c_str());
        FeaturePool *fp = new FeaturePool(in);
        context =  new RegTrainingContext<Stats,FeaturePool>(classCount,*fp,config.gainType(),config.voteDistThr()*config.voteDistThr());
        break;
    }
    case Configuration::FullFeaturesFactory:
    {
        cache.log() << "feature factory: FullDepthFeatureFactory" << std::endl;
        FullDepthFeatureFactory *fff = new FullDepthFeatureFactory(config.featureParameters());
        context = new RegTrainingContext<Stats,FullDepthFeatureFactory>(classCount,*fff,config.gainType(),config.voteDistThr()*config.voteDistThr());
        if (config.serializeInfo()){
            ((RegTrainingContext<Stats,FullDepthFeatureFactory> *)context)->setFeatureAccomulator(accomulator);
        }
        break;
    }
    case Configuration::PartialFeaturesFactory:
    {
        cache.log() << "feature factory: PartialDepthFeatureFactory" << std::endl;
        PartialDepthFeatureFactory *pff = new PartialDepthFeatureFactory(config.featureParameters());
        context = new RegTrainingContext<Stats,PartialDepthFeatureFactory>(classCount,*pff,config.gainType(),config.voteDistThr()*config.voteDistThr());
        if (config.serializeInfo()){
            ((RegTrainingContext<Stats,PartialDepthFeatureFactory> *)context)->setFeatureAccomulator(accomulator);
        }
        break;
    }
    default:
        cache.log() << "Unknown feature factory type; nothing to return" << std::endl;
    }

    return context;
}

int main(int argc, char **argv)
{
    std::cout << "config: " << argv[1] << std::endl;
    std::cout << "reading config data" << std::endl;

    std::ifstream in(argv[1]);
    Configuration config(in);
    in.close();

    std::cout << "configuration loaded" << std::endl;

    LocalCache cache(config.cacheFolderName(),"/home/kuznetso/tmp");

    if(!cache.init()){
        std::cerr << "failed to initialize temporary directory" << std::endl;
        exit(-1);
    }

    std::ostream &log = cache.log();

    log << "copying config file" << std::endl;
    copyfile(argv[1],cache.base() + "config.xml");

    log << "config name: " << argv[1] << std::endl;

    try{
        DepthDBWithVotesSImpl<VoteType,VoteDim>  db;
        Random random;
        time_t start,end;
        ProgressStream progress(log,Verbose);

        std::auto_ptr<Forest<DepthFeature, Stats> > forest;

        if (config.useSubsampler()){
            log << "using subsampler - rate: " << config.subsamplerRate() << std::endl;
            db.setSubsampler(new RandomSubsampler(config.subsamplerRate(),random));
        }
        db.loadDB(config.databaseFile(),config.databaseHasHeader());

        log << "loading from: " << config.databaseFile() << std::endl;
        log << "number of images: " << db.imageCount() << std::endl;
        log << "number of points: " << db.Count() << std::endl;
        log << "number of vote classes: " << (int)db.voteClassCount() << std::endl;
        log << "discard high variance: " << config.discardHighVar() << " with thr: " << config.nodeVarThr() << std::endl;

        if(!config.testOnly()){

            log << "train a forest" << std::endl;
            //some work here
            DepthFeatureParameters featureParams = config.featureParameters();

            log << featureParams;

            TrainingParameters trainingParameters = config.forestParameters();

            log << "decision levels: " << trainingParameters.MaxDecisionLevels << std::endl
                << "trees: " << trainingParameters.NumberOfTrees << std::endl
                << "features sampled: " << trainingParameters.NumberOfCandidateFeatures << std::endl
                << "threashold per feature: " << trainingParameters.NumberOfCandidateThresholdsPerFeature <<std::endl
                << std::endl;

            ITrainingContext<DepthFeature,Stats> *context = createTrainingContext(config,cache,db.voteClassCount());

            time(&start);

            forest = ForestTrainer<DepthFeature, Stats>::TrainForest (
                random, trainingParameters, *context, db,&progress);

            time(&end);
            double dif = difftime (end,start);

            log << "time: " << dif << std::endl;
            log << "forest trained" << std::endl;

            std::ostream &out = cache.openBinStream("forest");
            forest->Serialize(out);

            log << "forest serialized" << std::endl;
        }

        else{

            log << "load a forest from: " << config.forestFile().c_str() << std::endl;

            std::ifstream in(config.forestFile().c_str(),std::ios_base::binary);

            forest = Forest<DepthFeature, Stats>::Deserialize(in);

            log << "forest deserialized" << std::endl;
        }

        if(config.testOnTrain() || config.testOnly()){

            log << "testing" << std::endl;

            std::vector<std::vector<int> > leafIndicesPerTree;
            forest->Apply(db,leafIndicesPerTree,&progress);

            log << "forest applied" << std::endl;

            AggregatedLeafs<DepthFeature,Stats,VoteType,VoteDim> aggLeafs;

            mean_shift::MeanShift mshift;
            mshift.setRadius(config.meanShiftR());
            mshift.setMaxIter(config.meanShiftMaxIter());
            mshift.setMaxNeigboursCount(config.maxNN());

            log << "mean shift radius: " << config.meanShiftR() << std::endl;
            log << "mean shift max iter: " << config.meanShiftMaxIter() << std::endl;
            log << "mean shift max neighbour count: " << config.meanShiftMaxIter() << std::endl;
            log << "small vote weights threashold: " << config.smallWeightThr() << std::endl;

            log << "aggregating votes for tree " << 0 << std::endl;

            aggLeafs.SetVarThreashold(config.nodeVarThr());
            aggLeafs.SetNodeSizeThreashold(config.nodeSizeThr());
            aggLeafs.SetSmallWeightsThreashold(config.smallWeightThr());
            aggLeafs.Build(*forest,mshift,db.voteClassCount());

            std::ostream &leafs = cache.openBinStream("aggLeafs");

            aggLeafs.Serialize(leafs,*forest);

            std::ostream &leafIds = cache.openBinStream("leafIds");
            serializeVector<int>(leafIds,leafIndicesPerTree[0]);

            cache.closeAllStreams();

            std::vector<int> imgIds(db.Count(),0);
            std::vector<int> x(db.Count(),0);
            std::vector<int> y(db.Count(),0);
            std::string tmpstr;
            cv::Point2i current;

            std::vector<VotesAggregator<VoteType,VoteDim> > perImageVotes;

            VotesAggregator<VoteType,VoteDim> tmp(db.voteClassCount());
            perImageVotes.assign(db.imageCount(),tmp);

            log << "aggregating votes accross the images" << std::endl;

            for(int i=0; i<db.Count(); i++){
                db.getDataPoint(i,tmpstr,current);
                imgIds[i] = db.getImageIdx(i);
                x[i] = current.x;
                y[i] = current.y;
                for(int t=0; t<forest->TreeCount(); t++){
                    perImageVotes[imgIds[i]].AddVotes(aggLeafs.get(t,leafIndicesPerTree[0][i]));
                }
            }

            std::ostream &ids = cache.openBinStream("imgIds");
            serializeVector<int>(ids,imgIds);

            std::ostream &xvals = cache.openBinStream("xVals");
            serializeVector<int>(xvals,x);

            std::ostream &yvals = cache.openBinStream("yVals");
            serializeVector<int>(yvals,y);

            cache.closeAllStreams();

            log << "serializing aggregated votes" << std::endl;

            std::vector<cv::Vec<VoteType,VoteDim> > prediction;
            std::vector<cv::Vec<VoteType,VoteDim> > error;
            std::vector<double> weights;
            prediction.resize(db.voteClassCount());
            error.resize(db.voteClassCount());
            weights.resize(db.voteClassCount());
            mean_shift::MeanShift finalshift;

            finalshift.setRadius(config.meanShiftR());
            finalshift.setMaxIter(config.meanShiftMaxIter());
            finalshift.setMaxNeigboursCount(config.maxNN());

            std::ostream &aggVotesStream = cache.openBinStream("aggVotes");
            std::ostream &preStream = cache.openBinStream("predictions");
            std::ostream &gtStream = cache.openBinStream("gt");

            GroundTruthDecorator<VoteType,VoteDim> deco;
            for(int i=0; i<db.imageCount(); i++){
                //get prediction

                if(perImageVotes[i].Prediction(prediction,weights,finalshift)){
                    log << "prediction for image " << db.imageIdx2Filename(i) << " computed" << std::endl;
                }else{
                    log << "prediction for image " << db.imageIdx2Filename(i) << " failed" << std::endl;
                }

                //serialize
                serializeVoteVector<VoteType,VoteDim>(prediction, preStream);
                serializeVoteVector<VoteType,VoteDim>(db.getGT(i), gtStream);
                //create the decoration
                deco = GroundTruthDecorator<VoteType,VoteDim>(perImageVotes[i]);
                deco.SetGT(db.getGT(i));
                deco.Serialize(aggVotesStream);
                log << "prediction for image " << i << " done"<< std::endl;
            }

            log << "indices saved" << std::endl;
        }

    }catch(std::exception e){
        std::cerr << "exception caught: " << e.what() << std::endl;
        std::cerr.flush();
    }
}

