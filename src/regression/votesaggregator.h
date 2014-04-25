#ifndef VOTESAGGREGATOR_H
#define VOTESAGGREGATOR_H

#include "meanshift.h"
#include "regression/votesstatst.h"

template<class ElemType, int S>
class VotesAggregatorElem;

template<class ElemType, int S>
class VotesAggregator;

template<class ElemType, int S>
class GroundTruthDecorator
{
public:

    GroundTruthDecorator()
    {
        agg_ = 0;
    }

    GroundTruthDecorator(const VotesAggregator<ElemType,S> &agg)
    {
        agg_ = &agg;
    }

    void Serialize(std::ostream &out) const
    {
        agg_->Serialize(out);

        for(int i=0; i< gt_.size(); i++){
            for(int j=0; j<S; j++){
                out.write((const char *)&(gt_[i][j]),sizeof(ElemType));
            }
        }
    }

    void SetGT(const std::vector<cv::Vec<ElemType,S> > &gt)
    {
        gt_ = gt;
    }

private:
    const VotesAggregator<ElemType,S> *agg_;
    std::vector<cv::Vec<ElemType,S> > gt_;
};

template<class ElemType, int S>
class VotesAggregator
{
public:
    VotesAggregator()
    {

    }

    VotesAggregator(int elemCount)
    {
        elems_.resize(elemCount);
    }

    /*run mean shift to create a compressed votes storage*/
    void AggregateVotes(const VotesStatsT<ElemType,S> &stats, mean_shift::MeanShift &mshift)
    {
        elems_.resize(stats.ElemCount());
        int ind = 0;

//        std::cerr << "stats size: " << stats.Count() << std::endl;

        for(typename VotesStatsT<ElemType,S>::const_iterator i = stats.begin(); i != stats.end(); i++){
            if(i->Count()>0){
                elems_[ind].AggregateVotes(*i,mshift);
                ind++;
            }
        }
    }

     void FilterSmallWeights(float weightThr)
     {
         for(int i=0; i< elems_.size(); i++){
             elems_[i].FilterSmallWeights(weightThr);
         }
     }

    void AddVotes(const VotesAggregator<ElemType,S> &agg)
    {
        for(int i=0; i< elems_.size(); i++){
            elems_[i].AddVotes(agg.elems_[i]);
        }
    }

    void AddVotes(const VotesAggregator<ElemType,S> &agg, const cv::Vec<ElemType,S> &coord)
    {
        for(int i=0; i< elems_.size(); i++){
            elems_[i].AddVotes(agg.elems_[i],coord);
        }
    }

    void Serialize(std::ostream &out) const
    {
        int elemCount = elems_.size();
        out.write((const char*)&elemCount,sizeof(elemCount));

        for(int i=0; i< elems_.size(); i++){
            elems_[i].Serialize(out);
        }
    }

private:
    std::vector<VotesAggregatorElem<ElemType,S> > elems_;
};

template<class ElemType, int S>
class VotesAggregatorElem
{
public:
    VotesAggregatorElem()
    {

    }

    void AddVotes(const VotesAggregatorElem<ElemType,S> &agg)
    {
        weights_.insert(weights_.end(),agg.weights_.begin(),agg.weights_.end());
        votes_.insert(votes_.end(),agg.votes_.begin(),agg.votes_.end());
    }

    void AddVotes(const VotesAggregatorElem<ElemType,S> &agg, const cv::Vec<ElemType,S> &coord)
    {
        weights_.insert(weights_.end(),agg.weights_.begin(),agg.weights_.end());
        //pre-allocate chunk of data
        votes_.reserve(agg.votes_.size());
        for(int i=0; i< agg.votes_.size(); i++)
        {
            //add votes
            votes_.push_back(coord + agg.votes_[i]);
        }
    }

    void Count()
    {
        return weights_.size();
    }

    //aggregate votes using meanshift
    void AggregateVotes(const VotesStatsElemT<ElemType,S> &stats, mean_shift::MeanShift &mshift){
        //allocate data
        //std::cerr << "starting votes aggregation" << std::endl;

        typename VotesStatsElemT<ElemType,S>::element_count votesCount = stats.Count();
        mean_shift::ElemType *data = new mean_shift::ElemType[votesCount*S];
        flann::Matrix<mean_shift::ElemType> votes(data,votesCount,S);
        mean_shift::MatrixRow wrapper(votes);
        int row = 0;
        //fill the matrix
        //std::cerr << "copying votes: " << votesCount << std::endl;
        for(typename VotesStatsElemT<ElemType,S>::const_iterator i = stats.begin(); i!= stats.end(); i++){
            convert((*i),wrapper.setRow(row)); //todo
            row++;

        }
        //std::cerr << "row: " <<row <<  " votes count: " << votesCount << std::endl;

        mshift.setPoints(votes);
        //std::cerr << "running mean shift" << std::endl;

        try{
            mshift.run();
//            std::cerr << "clusters detected: " <<mshift.getClusterNumber() << std::endl;
        }catch(flann::FLANNException e){
            std::cerr << "flann exception: " << e.what() << std::endl;
        }
        catch(std::exception e){
            std::cerr << "std exception: " << e.what() << std::endl;
        }

        //std::cerr << "getting centers" << std::endl;
        //get final votes
        flann::Matrix<mean_shift::ElemType> centers  = mshift.getClusterCenters();

        //std::cerr << "before assignment" << std::endl;
        mean_shift::MatrixRow centerswrapper(centers);
        //and their weights
        flann::Matrix<mean_shift::IndexType> sizes =  mshift.getClusterSizes();

        votes_.resize(mshift.getClusterNumber());
        weights_.resize(mshift.getClusterNumber());

        //std::cerr << "converting to votes and weights" << std::endl;

        for(int i=0; i<mshift.getClusterNumber(); i++){
            if(sizes.ptr()[i]<1) {
                std::cerr << "WARNING: sizes.ptr()[i]<1"<< std::endl;
            }
            weights_[i] = ((double)(sizes.ptr()[i]))/((double)votesCount);
            centerswrapper.setRow(i);
            for(int j=0; j< S; j++){
                votes_[i][j] = centerswrapper[j];
            }
        }
        //std::cerr << "done; memory clean" << std::endl;
        //clear memory
        delete [] data;
        delete [] centers.ptr();
        delete [] sizes.ptr();
    }

    void Serialize(std::ostream &out) const
    {
        int elemCount = weights_.size();
        out.write((const char*)&elemCount,sizeof(elemCount));
        int dims = S;
        out.write((const char*)&dims,sizeof(dims));

        for(int i=0; i<votes_.size(); i++){
            for(int j=0; j<S; j++){
                out.write((const char *)(&(votes_[i][j])),sizeof(ElemType));
            }
        }
        for(int i=0; i<weights_.size(); i++){
            out.write((const char *)&(weights_[i]),sizeof(double));
        }
    }

    void FilterSmallWeights(float weightThr)
    {

        std::vector<double>  newweights;
        std::vector<cv::Vec<ElemType,S> > newvotes;

        for(int i=0; i< weights_.size(); i++){
            if(weights_[i]>weightThr) {
                // a little ineffective..
                newweights.push_back(weights_[i]);
                newvotes.push_back(votes_[i]);
            }
        }

        weights_ = newweights;
        votes_ = newvotes;
    }

private:
    void convert(const cv::Vec<ElemType,S> &from, mean_shift::MatrixRow &to)
    {
        //assume the sizes are correct
        for(int i=0; i< S; i++){
            to[i] = from[i];
        }
    }

    std::vector<double> weights_;
    std::vector<cv::Vec<ElemType,S> > votes_;

};

#endif // VOTESAGGREGATOR_H
