#include "localcache.h"
#include "configuration.h"

#include "Tree.h"
#include "Random.h"
#include "Forest.h"
#include "ForestTrainer.h"
#include "ProgressStream.h"

#include "regression/depthdbreg.h"
#include "regression/regtrainingcontext.h"
#include "regression/votesaggregator.h"
#include "regression/votesstatst.h"
#include "regression/aggregatedleafs.h"

#include "copyfile.h"

#define VoteDim 3
typedef float VoteType;

typedef VotesStatsT<VoteType,VoteDim> Stats;
typedef AggregatedLeafs<DepthFeature,Stats,VoteType,VoteDim> Leafs;


using namespace MicrosoftResearch::Cambridge::Sherwood;

int main(int argc, char ** argv)
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

    try{

        DepthDBWithVotesSImpl<VoteType,VoteDim>  db;
        Random random;
        ProgressStream progress(log,Verbose);

        if (config.useSubsampler()){
            log << "using subsampler - rate: " << config.subsamplerRate() << std::endl;
            db.setSubsampler(new RandomSubsampler(config.subsamplerRate(),random));
        }

        db.loadDB(config.databaseFile(),config.databaseHasHeader());

        log << "loading from: " << config.databaseFile() << std::endl;
        log << "number of images: " << db.imageCount() << std::endl;
        log << "number of points: " << db.Count() << std::endl;
        log << "number of vote classes: " << (int)db.voteClassCount() << std::endl;


        log << "load a forest from: " << config.forestFile().c_str() << std::endl;
        std::ifstream inforest(config.forestFile().c_str(),std::ios_base::binary);

        std::auto_ptr<Forest<DepthFeature, Stats> > forest = Forest<DepthFeature, Stats>::Deserialize(inforest);

        log << "forest deserialized" << std::endl;

        inforest.close();

        log << "load leafs from: " << config.leafsFile().c_str() << std::endl;

        std::ifstream inleafs(config.leafsFile().c_str(), std::ios_base::binary);
        Leafs aggLeafs;
        aggLeafs.Deserialize(inleafs);

        log << "leafs deserialized" << std::endl;

        std::vector<std::vector<int> > leafIndicesPerTree;
        forest->Apply(db,leafIndicesPerTree,&progress);

        log << "forest applied" << std::endl;

        std::ostream &leafIds = cache.openBinStream("leafIds");
        serializeVector<int>(leafIds,leafIndicesPerTree[0]);

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
            perImageVotes[imgIds[i]].AddVotes(aggLeafs.get(0,leafIndicesPerTree[0][i]));
        }

        std::ostream &ids = cache.openBinStream("imgIds");
        serializeVector<int>(ids,imgIds);

        std::ostream &xvals = cache.openBinStream("xVals");
        serializeVector<int>(xvals,x);

        std::ostream &yvals = cache.openBinStream("yVals");
        serializeVector<int>(yvals,y);

        log << "serializing aggregated votes" << std::endl;

        std::vector<cv::Vec<VoteType,VoteDim> > prediction;
        std::vector<double> weights;
        std::vector<cv::Vec<VoteType,VoteDim> > gt;
        std::ostream &preout = cache.openBinStream("pre");
        std::ostream &gtout = cache.openBinStream("gt");

        prediction.resize(1);
        weights.resize(1);

        mean_shift::MeanShift mshift;
        std::vector<float> mean;
        std::vector<float> std10;

        mean.resize(3);
        std10.resize(3);

        mshift.setRadius(15);
        mshift.setMaxIter(30);
        mshift.setMaxNeigboursCount(30000);

        std::ostream &aggVotesStream = cache.openBinStream("aggVotes");
        GroundTruthDecorator<VoteType,VoteDim> deco;
        for(int i=0; i<db.imageCount(); i++){
            /*create the decoration*/
            std::cerr << "image processing: " << perImageVotes[i].Count(0) << std::endl;

            //perImageVotes[i].Normalize(mean,std10);


            perImageVotes[i].Prediction(prediction,weights,mshift);
            /*for(int l=0; l<prediction.size();l++){
                for(int s=0; s<VoteDim; s++){
                    prediction[l][s] = prediction[l][s]*std10[l*VoteDim+s] + mean[l*VoteDim+s];
                }
            }*/
            serializeVoteVector<VoteType,VoteDim>(prediction, preout);


            //perImageVotes[i].Denormalize(mean,std10);
            deco = GroundTruthDecorator<VoteType,VoteDim>(perImageVotes[i]);

            gt = db.getGT(i);

            for(int l=0; l<prediction.size();l++){
                for(int s=0; s<VoteDim; s++){
                    gt[l][s] = gt[l][s]*config.forest(0).std_[l*VoteDim+s] + config.forest(0).mean_[l*VoteDim+s];
                }
            }

            serializeVoteVector<VoteType,VoteDim>(gt, gtout);

            deco.SetGT(gt);
            deco.Serialize(aggVotesStream);
        }

        log << "indices saved" << std::endl;



    }catch(std::exception e){
        std::cerr << "exception caught: " << e.what() << std::endl;
        std::cerr.flush();
    }
}
