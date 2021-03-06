#include <fstream>
#include <math.h>

#include "configuration.h"
#include "copyfile.h"
#include "localcache.h"
#include "depthfeature.h"
#include "serialization.h"

#include "Random.h"
#include "Forest.h"
#include "ForestTrainer.h"
#include "TrainingParameters.h"

#include "regression/votesstatst.h"
#include "regression/depthdbreg.h"
#include "regression/regtrainingcontext.h"
#include "regression/votesaggregator.h"
#include "regression/aggregatedleafs.h"

#include "seq/forestmerger.h"

#define VoteDim 3
typedef float VoteType;

typedef VotesStatsT<VoteType,VoteDim> Stats;
typedef std::vector<VotesAggregator<VoteType,VoteDim> >  VotesAggregators;

using namespace MicrosoftResearch::Cambridge::Sherwood;

typedef DepthDBWithVotesSImpl<VoteType,VoteDim>::fileindex_type DBfileindex;
typedef DepthDBWithVotesSImpl<VoteType,VoteDim>::index_type DBindex;

typedef std::vector<cv::Vec<VoteType,VoteDim> > Vote;

typedef AggregatedLeafs<DepthFeature,Stats,VoteType,VoteDim> Leafs;

void normalizePrediction(const std::vector<float> &mean, const std::vector<float> &var, int startIdx, cv::Vec<VoteType,VoteDim> &prediction)
{

    for( int i=0; i< VoteDim; i++){
        prediction[i] =  prediction[i]*var[startIdx+i] + mean[startIdx+i];
    }
}

float expProb(float v1, float v2, float div)
{
    return std::exp(-(v1-v2)*(v1-v2)/div);
}

DBfileindex getForestPrediction2Forests(DepthDBWithVotesSImpl<VoteType,VoteDim> &db,
                                Forest<DepthFeature, Stats> &forest1,
                                Forest<DepthFeature, Stats> &forest2,
                                AggregatedLeafs<DepthFeature,Stats,VoteType,VoteDim>  &aggLeafs1,
                                AggregatedLeafs<DepthFeature,Stats,VoteType,VoteDim>  &aggLeafs2,
                                LocalCache &cache,
                                const Configuration &config,
                                VoteType w1, VoteType w2,
                                std::vector<cv::Vec<VoteType,VoteDim> > &prediction, //OUT
                                std::vector<double> &weights, //OUT
                                DBindex &idx) //OUT
{
    //strong assumption: the pixels of one image are following one another

    std::ostream &log = cache.log();

    DBfileindex img = db.getImageIdx(idx);
    std::vector<DBindex> oneimage;


    while(idx < db.Count() && db.getImageIdx(idx) == img){
        oneimage.push_back(idx++);
    }

    //we dont need GT, so simple filebased index will do
    SubindexFileBasedImageDB dboneimage(db,oneimage);
    std::vector<std::vector<int> > leafIndicesPerTree1;
    ProgressStream progress(cache.log(),Verbose);

    forest1.Apply(dboneimage,leafIndicesPerTree1,&progress);

    VotesAggregator<VoteType,VoteDim> imageVotes(db.voteClassCount());

    for(int t=0; t< forest1.TreeCount(); t++){
        for(int i=0; i< dboneimage.Count(); i++){
            if(leafIndicesPerTree1[t][i]>0){
                imageVotes.AddVotes(aggLeafs1.get(t,leafIndicesPerTree1[t][i]),w1);
            }
        }
    }

    std::vector<std::vector<int> > leafIndicesPerTree2;
    forest2.Apply(dboneimage,leafIndicesPerTree2,&progress);

    for(int t=0; t< forest2.TreeCount(); t++){
        for(int i=0; i< dboneimage.Count(); i++){
            if(leafIndicesPerTree2[t][i]>0){
                imageVotes.AddVotes(aggLeafs2.get(t,leafIndicesPerTree2[t][i]),w1);
            }
        }
    }

    mean_shift::MeanShift finalshift;

    finalshift.setRadius(config.meanShiftR());
    finalshift.setMaxIter(config.meanShiftMaxIter());
    finalshift.setMaxNeigboursCount(config.maxNN());

    std::vector<float> imean;
    std::vector<float> istd;

    imean.resize(db.voteClassCount()*VoteDim);
    istd.resize(db.voteClassCount()*VoteDim);

    //imageVotes.Normalize(imean,istd);

    std::cerr << "istd: " << istd[0] << std::endl;

    imageVotes.Prediction(prediction,weights,finalshift);

    imageVotes.Serialize(cache.openBinStream("imageVotes"));

    /*for(int i=0; i<prediction.size();i++){
        for(int j=0; j<VoteDim; j++){
            prediction[i][j] = prediction[i][j]*istd[i*VoteDim+j] + imean[i*VoteDim+j];
        }
    }*/

    log << "prediction computed" << std::endl;

    return img;
}

DBfileindex getForestPrediction(DepthDBWithVotesSImpl<VoteType,VoteDim> &db,
                                Forest<DepthFeature, Stats> &forest,
                                AggregatedLeafs<DepthFeature,Stats,VoteType,VoteDim>  &aggLeafs,
                                LocalCache &cache,
                                const Configuration &config,
                                std::vector<cv::Vec<VoteType,VoteDim> > &prediction, //OUT
                                std::vector<double> &weights, //OUT
                                DBindex &idx) //OUT
{
    //strong assumption: the pixels of one image are following one another
    
    std::ostream &log = cache.log();

    DBfileindex img = db.getImageIdx(idx);
    std::vector<DBindex> oneimage;
    while(idx< db.Count() && db.getImageIdx(idx) == img){
        oneimage.push_back(idx++);
    }

    //we dont need GT, so simple filebased index will do
    SubindexFileBasedImageDB dboneimage(db,oneimage);
    std::vector<std::vector<int> > leafIndicesPerTree;
    ProgressStream progress(cache.log(),Verbose);

    forest.Apply(dboneimage,leafIndicesPerTree,&progress);

    log << "forest applied" << std::endl;

    VotesAggregator<VoteType,VoteDim> imageVotes(db.voteClassCount());

    
    for(int t=0; t< forest.TreeCount(); t++){
        for(int i=0; i< dboneimage.Count(); i++){
            imageVotes.AddVotes(aggLeafs.get(t,leafIndicesPerTree[t][i]));
        }
    }

    log << "votes aggregated" << std::endl;

    
    mean_shift::MeanShift finalshift;
    std::vector<float> imean;
    std::vector<float> istd;

    imean.resize(db.voteClassCount()*VoteDim);
    istd.resize(db.voteClassCount()*VoteDim);

    finalshift.setRadius(config.meanShiftR());
    finalshift.setMaxIter(config.meanShiftMaxIter());
    finalshift.setMaxNeigboursCount(config.maxNN());

    imageVotes.Normalize(imean,istd);

    bool valid = imageVotes.Prediction(prediction,weights,finalshift);

    for(int i=0; i<prediction.size();i++){
        for(int j=0; j<VoteDim; j++){
            prediction[i][j] = prediction[i][j]*istd[i*VoteDim+j] + imean[i*VoteDim+j];
        }
    }

    if(valid)
        log << "prediction for image " << db.imageIdx2Filename(img) << " computed" << std::endl;
    else
        log << "prediction for image " << db.imageIdx2Filename(img) << " failed" << std::endl;

    return img;
}

int main(int argc, char **argv)
{
    std::cout << "config: " << argv[1] << std::endl;
    std::cout << "reading config data" << std::endl;

    std::ifstream in(argv[1]);
    Configuration config(in);
    in.close();

    LocalCache cache(config.cacheFolderName(),"/home/kuznetso/tmp");

    if(!cache.init()){
        std::cerr << "failed to initialize temporary directory" << std::endl;
        exit(-1);
    }

    std::ostream &log = cache.log();

    log << "copying config file" << std::endl;
    copyfile(argv[1],cache.base() + "config.xml");

    DepthDBWithVotesSImpl<VoteType,VoteDim>  db;
    Random random;

    if (config.useSubsampler()){
        log << "using subsampler - rate: " << config.subsamplerRate() << std::endl;
        db.setSubsampler(new RandomSubsampler(config.subsamplerRate(),random));
    }

    log << "loading db" << std::endl;

    db.loadDB(config.databaseFile(),config.databaseHasHeader());

    log << "db loaded; number of images: " << db.imageCount() << std::endl;

    log << "number of forests: " << config.forestsCount() << std::endl;

    std::ifstream inforest;
    std::vector< Forest<DepthFeature, Stats> * > forests;
    std::vector< AggregatedLeafs<DepthFeature,Stats,VoteType,VoteDim> * >aggLeafs;

    std::auto_ptr<Forest<DepthFeature, Stats> >  stmp;

    mean_shift::MeanShift mshift;
    mshift.setRadius(config.meanShiftR());
    mshift.setMaxIter(config.meanShiftMaxIter());
    mshift.setMaxNeigboursCount(config.maxNN());

    for(int i=0; i<config.forestsCount(); i++ ){
        inforest.open(config.forest(i).ff_.c_str());
        stmp = Forest<DepthFeature, Stats>::Deserialize(inforest);
        inforest.close();

        forests.push_back(stmp.release());
        //create and resize aggleafs
        aggLeafs.push_back(new AggregatedLeafs<DepthFeature,Stats,VoteType,VoteDim>());

        aggLeafs.back()->SetVarThreashold(config.nodeVarThr());
        aggLeafs.back()->SetNodeSizeThreashold(config.nodeSizeThr());
        aggLeafs.back()->SetSmallWeightsThreashold(config.smallWeightThr());

        log << "aggregating votes for tree " << 0  << " for forest: " << i << std::endl;

        aggLeafs.back()->Build(*forests.back(),mshift,db.voteClassCount());

        log << "forest " << config.forest(i).ff_ << " loaded" << std::endl;

        aggLeafs.back()->Denormalize(config.forest(i).mean_,config.forest(i).std_);

        log << "leaf votes denormalized" << std::endl;
    }

    log << "forests deserialized" << std::endl;
    

    DBindex idx = 0, idxSave = 0;
    DBfileindex currentImg = -1;
    std::vector<cv::Vec<VoteType,VoteDim> > prediction1;
    std::vector<cv::Vec<VoteType,VoteDim> > prediction2;
    std::vector<cv::Vec<VoteType,VoteDim> > merged;
    std::vector<cv::Vec<VoteType,VoteDim> > fullmerged;
    std::vector<double> weights1;
    std::vector<double> weights2;
    std::vector<double> weightsM;
    std::vector<double> weightsFM;
    std::vector<int> fi;
    std::vector<int> nfi;
    std::vector<int> gtfi;
    std::vector<float> wf;
    std::vector<float> wnf;
    int forestIdx = 3;
    int forestNeighbourIdx = 3;
    int gtforestIdx = 3;
    int rangeIdx = 1;
    float w1,w2,sum;
    float clcent1, clcent2;

    ForestMerger<Stats> forestMerger;
    AggregatedLeafsMerger<VoteType,VoteDim> aggMerger;
    

    prediction1.resize(db.voteClassCount());
    prediction2.resize(db.voteClassCount());
    merged.resize(db.voteClassCount());
    fullmerged.resize(db.voteClassCount());
    weights1.resize(db.voteClassCount());
    weights2.resize(db.voteClassCount());
    weightsM.resize(db.voteClassCount());
    weightsFM.resize(db.voteClassCount());

    fi.resize(db.imageCount());
    nfi.resize(db.imageCount());
    gtfi.resize(db.imageCount());
    wf.resize(db.imageCount());
    wnf.resize(db.imageCount());

    std::ostream &predictionOut1 = cache.openBinStream("pr1");
    std::ostream &predictionOut2 = cache.openBinStream("pr2");
    std::ostream &predictionOutM = cache.openBinStream("prPM");
    std::ostream &predictionOutFM = cache.openBinStream("prFM");
    std::ostream &gtOut = cache.openBinStream("gt");
    


    for(int i=0; i<config.forestsCount(); i++){
	
        if(db.getGT(0)[0][rangeIdx]>config.forest(i).bounds_[0] &
           db.getGT(0)[0][rangeIdx]<=config.forest(i).bounds_[1]){
            gtforestIdx = i;
            forestIdx = i;
            forestNeighbourIdx = i;
            break;
       
	     
	  }
    }

    float div = (config.forest(1).bounds_[1] - config.forest(1).bounds_[0])/16;
    div = 2*div*div;

    //how to intialize?

    w1 = w2 = 0.5;
    float mix  = db.getGT(0)[0][rangeIdx];
 //   int startIdx;

    while(idx<db.Count() & forestIdx>=0 & forestIdx< forests.size() ){

        clcent1 = (config.forest(forestIdx).bounds_[0]+config.forest(forestIdx).bounds_[1])/2;
        clcent2 = (config.forest(forestNeighbourIdx).bounds_[0]+config.forest(forestNeighbourIdx).bounds_[1])/2;

        w1 = expProb(mix,clcent1,div);
        w2 = expProb(mix,clcent2,div);

        sum = w1+w2;

        w1 = w1/sum;
        w2 = w2/sum;

        std::cerr << "w1: " << w1 << "w2: "<< w2 << std::endl;

        forestMerger.SetWeights(w1,w2);
        std::auto_ptr<Forest<DepthFeature,Stats> > forest = forestMerger.mergeForests(*forests[forestIdx],*forests[forestNeighbourIdx]);
        aggMerger.SetWeights(w1,w2);
        std::auto_ptr<Leafs > leafs =aggMerger.mergeAggregatedLeafs(*aggLeafs[forestIdx],*aggLeafs[forestNeighbourIdx]);

        idxSave= idx;

        currentImg = getForestPrediction(db,*forests[forestIdx],*aggLeafs[forestIdx],cache,config,prediction1,weights1,idx);

        idx = idxSave;
        currentImg = getForestPrediction(db,*forests[forestNeighbourIdx],*aggLeafs[forestNeighbourIdx],cache,config,prediction2,weights2,idx);

        idx = idxSave;
        currentImg = getForestPrediction2Forests(db,*forests[forestIdx],*forests[forestNeighbourIdx],*aggLeafs[forestIdx],*aggLeafs[forestNeighbourIdx],cache,config,w1,w2,merged,weightsM,idx);

        idx = idxSave;
        currentImg = getForestPrediction(db,*forest,*leafs,cache,config,fullmerged,weightsFM,idx);

        fi[currentImg] = forestIdx;
        nfi[currentImg] = forestNeighbourIdx;

        //normalize prediction
 /*     if(config.forest(forestIdx).mean_.size()>0){

            startIdx = 0;
            for(int i=0; i<prediction1.size();i++ ){
                normalizePrediction(config.forest(forestIdx).mean_,
                                    config.forest(forestIdx).std_,
                                    startIdx,
                                    prediction1[i]);
                normalizePrediction(config.forest(forestNeighbourIdx).mean_,
                                    config.forest(forestNeighbourIdx).std_,
                                    startIdx,
                                    prediction2[i]);
                startIdx +=VoteDim;
            }
        }*//*
        w1 = (prediction1[0][rangeIdx] - (config.forest(forestIdx).bounds_[0]+config.forest(forestIdx).bounds_[1])/2);
        w2 = (prediction2[0][rangeIdx] - (config.forest(forestNeighbourIdx).bounds_[0]+config.forest(forestNeighbourIdx).bounds_[1])/2);

        w1 = std::abs(w1);
        w2 = std::abs(w2);

        w1 = w1/(w1 + w2);
        w2 = w2/(w1 + w2);*/

        wf[currentImg] = weights1[0];
        wnf[currentImg] = weights2[0];

        //w1 = expProb(prediction1[0][rangeIdx],clcent2,div);
        //w2 = expProb(prediction2[0][rangeIdx],clcent1,div);
/*
       for(int i=0; i< prediction1.size(); i++){
            std::cerr << "merging prediction here: " << std::endl;
            for(int j=0; j<VoteDim; j++){
                merged[i][j] = (w1*prediction1[i][j] +  w2*prediction2[i][j])/(w1+w2);

            }

        }*/

       mix  = fullmerged[0][rangeIdx];

        std::cerr << "mix: " << mix << std::endl;
 //       mix = (prediction1[0][rangeIdx]*w1 + prediction2[0][rangeIdx]*w2)/(w1+w2);
 /*        if ( w2 < 0.7){
            mix = prediction1[0][rangeIdx];
        }else{
            if (w1<0.7){
                mix = prediction2[0][rangeIdx];
            }else{*/

 /*           }
        }*/

        //choose the forest
        forestIdx = -1;
        forestNeighbourIdx = -1;
        for(int i=0; i<config.forestsCount(); i++){
            if (mix>config.forest(i).bounds_[0] &
                mix<=config.forest(i).bounds_[1]){

                if(forestIdx<0){
                    forestIdx = i;
                }else{
                    forestNeighbourIdx = i;
                }
 //               if(prediction1[0][rangeIdx]>=(config.forest(i).bounds_[0]+config.forest(i).bounds_[1])/2)
 //                   forestNeighbourIdx = i+1;
 //               else
 //                   forestNeighbourIdx = i-1;

            }
            if(db.getGT(currentImg)[0][rangeIdx]>config.forest(i).bounds_[0] &
               db.getGT(currentImg)[0][rangeIdx]<=config.forest(i).bounds_[1]){
                gtforestIdx = i;
            }
        }

        if(forestNeighbourIdx<0){
            forestNeighbourIdx = forestIdx;
        }

        //forestIdx = gtforestIdx;
        //forestNeighbourIdx = gtforestIdx;

        gtfi[currentImg] = gtforestIdx;

        if (forestNeighbourIdx<0){
            forestNeighbourIdx = 0;
        }
        if(forestNeighbourIdx>=config.forestsCount()){
            forestNeighbourIdx = forestIdx;
        }

        //forestIdx = gtforestIdx;
        //forestNeighbourIdx = gtforestIdx;

        serializeVoteVector<VoteType,VoteDim>(prediction1, predictionOut1);
        serializeVoteVector<VoteType,VoteDim>(prediction2, predictionOut2);
        serializeVoteVector<VoteType,VoteDim>(merged, predictionOutM);
        serializeVoteVector<VoteType,VoteDim>(fullmerged,predictionOutFM);
        serializeVoteVector<VoteType,VoteDim>(db.getGT(currentImg), gtOut);


        log << "prediction: " << prediction1[0][rangeIdx] << std::endl;
        log << "gt: " << db.getGT(currentImg)[0][rangeIdx] << std::endl;
        log << "next forest: " << forestIdx << std::endl;
        log << "next neighbour forest: " << forestNeighbourIdx << std::endl;
        log << "next gt forest: " << gtforestIdx << std::endl;
    }

    serializeVector<int>(cache.openBinStream("fi"),fi);
    serializeVector<int>(cache.openBinStream("nfi"),nfi);
    serializeVector<int>(cache.openBinStream("gtfi"),gtfi);
    serializeVector<float>(cache.openBinStream("wf"),wf);
    serializeVector<float>(cache.openBinStream("wnf"),wnf);

}
