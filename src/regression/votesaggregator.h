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

    void SetAggregator(const VotesAggregator<ElemType,S> &agg)
    {
        agg_ = &agg;
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
        isEmpty_ = true;
    }

    VotesAggregator(int elemCount)
    {
        elems_.resize(elemCount);
        isEmpty_ = true;
    }

    void Denormalize(const std::vector<float> &mean, const std::vector<float> &std10)
    {
        int ind = 0;
        for(int i=0; i< elems_.size(); i++){
            elems_[i].Denormalize(mean,std10,ind);
            ind+=S;
        }
    }

    void Normalize(std::vector<float> &mean, std::vector<float> &std10)
    {
        int ind = 0;
        for(int i=0; i< elems_.size(); i++){
            elems_[i].Normalize(mean,std10,ind);
            ind+=S;
        }
    }

    bool Prediction(std::vector<cv::Vec<ElemType,S> > &prediction, std::vector<double> &weight,mean_shift::MeanShift &mshift)     {
        bool hademptysets = false;
        for(int i=0; i< elems_.size(); i++){
            weight[i]=elems_[i].Prediction(prediction[i],mshift);

            hademptysets = hademptysets | (weight[i]<1e-5);
        }
        return ~hademptysets;
    }

    /*run mean shift to create a compressed votes storage*/
    void AggregateVotes(const VotesStatsT<ElemType,S> &stats, mean_shift::MeanShift &mshift)
    {
        isEmpty_ = false;
        elems_.resize(stats.ElemCount());
        int ind = 0;

        //std::cerr << "stats size: " << stats.Count() << std::endl;

        for(typename VotesStatsT<ElemType,S>::const_iterator i = stats.begin(); i != stats.end(); i++){
            if(i->Count()>0){
                elems_[ind].AggregateVotes(*i,mshift);

                mshift.Clean();
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

     void FilterSmallVotes(float voteThr)
     {
         long int ecount = 0;
         for(int i=0; i< elems_.size(); i++){
            ecount += elems_[i].OriCount();
         }
         for(int i=0; i< elems_.size(); i++){
             if((double)elems_[i].OriCount()/ecount<voteThr){
                 elems_[i].Clear();
             }
         }
     }


     void RescaleVoteWeights()
     {
         long int ecount = 0;
         for(int i=0; i< elems_.size(); i++){
            ecount += elems_[i].OriCount();
         }

         for(int i=0; i< elems_.size(); i++){
             elems_[i].RescaleWeights(((double)elems_[i].OriCount()/(double)ecount));
         }
     }


    void AddVotes(const VotesAggregator<ElemType,S> &agg)
    {
        isEmpty_ = false;
        for(int i=0; i< elems_.size(); i++){
            elems_[i].AddVotes(agg.elems_[i]);
        }
    }

    void AddVotes(const VotesAggregator<ElemType,S> &agg, ElemType w)
    {
/*        std::cerr << "agg.elems_.size: " << agg.elems_.size() << std::endl;
        std::cerr << "this.elems_.size: " << agg.elems_.size() << std::endl;
        std::cerr.flush();*/
        isEmpty_ = false;
        for(int i=0; i< elems_.size(); i++){
            elems_[i].AddVotes(agg.elems_[i],w);
        }
    }

    void AddVotes(const VotesAggregator<ElemType,S> &agg, const cv::Vec<ElemType,S> &coord)
    {
        isEmpty_ = false;
        for(int i=0; i< elems_.size(); i++){
            elems_[i].AddVotes(agg.elems_[i],coord);
        }
    }

    int Count(int elem) const
    {
        return elems_[elem].Count();
    }

    int OriCount(int elem) const
    {
        return elems_[elem].OriCount();
    }


    bool IsEmpty() const
    {
        return isEmpty_;
    }

    void Serialize(std::ostream &out) const
    {
        int elemCount = elems_.size();
        out.write((const char*)&elemCount,sizeof(elemCount));

        for(int i=0; i< elems_.size(); i++){
            elems_[i].Serialize(out);
        }
    }

    void Deserialize(std::istream &in)
    {
        int elemCount;
        in.read((char *)&elemCount,sizeof(elemCount));
        elems_.resize(elemCount);
        for(int i=0; i< elems_.size(); i++){
            elems_[i].Deserialize(in);
        }

    }

private:
    std::vector<VotesAggregatorElem<ElemType,S> > elems_;
    bool isEmpty_;
};

template<class ElemType, int S>
class VotesAggregatorElem
{
public:
    VotesAggregatorElem()
    {
        prediction_.resize(S);
        oriVoteCount_ = 0;
    }

    void Denormalize(const std::vector<float> &mean, const std::vector<float> &std10, int idx)
    {
        for(int v= 0; v<votes_.size(); v++){
            for( int i=0; i< S; i++){
                votes_[v][i] =  votes_[v][i]*std10[idx+i] + mean[idx+i];
            }
        }
    }


    void Normalize(std::vector<float> &mean, std::vector<float> &std10, int idx)
    {

        cv::Vec<double,S> m;
        cv::Vec<double,S> m2;
        setTo<double,S>(m,0);
        setTo<double,S>(m2,0);
        for(int v= 0; v<votes_.size(); v++){
            for(int j=0;j<S;j++){
                m[j]+= votes_[v][j];
                m2[j]+= votes_[v][j]*votes_[v][j];
            }
        }

        cv::Vec<double,S> var;
        for(int j=0;j<S;j++){
            m[j] = m[j]/votes_.size();
            var[j] = m2[j]/votes_.size() - m[j]*m[j];
            if(var[j]<1e-5){
	       //just in case it is negative or small
                var[j] = 1;
            }
        }

        for(int v= 0; v<votes_.size(); v++){
            for( int i=0; i< S; i++){
                mean[idx+i] = m[i];
		//make std consisten with what is done in MATLAB, that is not nice
                std10[idx+i] = std::sqrt(var[i])/10;
                votes_[v][i] =  (votes_[v][i] - mean[idx+i])/std10[idx+i];
            }
        }

   }

    double Prediction(cv::Vec<ElemType,S> &prediction, mean_shift::MeanShift &mshift)
    {
        if(votes_.size()<=0){
            std::cerr << "WARNING: empty point set by prediction; returning" << std::endl;
            setTo<ElemType,S>(prediction,0);
            return 0;
        }
      
        //conver points and weights to a matrix
        mean_shift::ElemType *weights_data = new mean_shift::ElemType[weights_.size()];
        mean_shift::ElemType *votes_data = new mean_shift::ElemType[votes_.size()*S];

        flann::Matrix<mean_shift::ElemType> weights(weights_data,weights_.size(),1);
        flann::Matrix<mean_shift::ElemType> votes(votes_data,votes_.size(),S);
        mean_shift::MatrixRow wrapper(votes);

        std::copy(weights_.begin(),weights_.end(),weights_data);

        for(int i=0; i<weights_.size();i++){
            convert(votes_[i],wrapper.setRow(i));
        }
        
        mshift.setPoints(votes,weights);

        try{
            mshift.run();

        }catch(flann::FLANNException e){
            std::cerr << "flann exception: " << e.what() << std::endl;
        }
        catch(std::exception e){
            std::cerr << "std exception: " << e.what() << std::endl;
        }
        
        flann::Matrix<double> clweights = mshift.getClusterWeights();
        //flann::Matrix<mean_shift::IndexType> sizes =  mshift.getClusterSizes();

        //find the biggest cluster
        //mean_shift::IndexType *maxelem = std::max_element(sizes.ptr(),sizes.ptr()+sizes.cols);
        double *maxelem = std::max_element(clweights.ptr(),clweights.ptr() + clweights.cols);
        int idx = maxelem - clweights.ptr();
        //double maxweight = ((double)(*maxelem))/votes_.size();
        double maxweight = *maxelem;

        mshift.getClusterCenter(idx,prediction_);

        std::copy(prediction_.begin(),prediction_.end(),&(prediction[0]));

        delete [] clweights.ptr();
        delete [] weights_data;
        delete [] votes_data;

        return maxweight;

    }

    void AddVotes(const VotesAggregatorElem<ElemType,S> &agg)
    {
        weights_.insert(weights_.end(),agg.weights_.begin(),agg.weights_.end());
        votes_.insert(votes_.end(),agg.votes_.begin(),agg.votes_.end());
    }

    void AddVotes(const VotesAggregatorElem<ElemType,S> &agg, ElemType w)
    {
        weights_.reserve(agg.weights_.size());
        for(int i=0; i<agg.weights_.size();i++) {
            weights_.push_back(agg.weights_[i]*w);
        }
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

    int Count() const
    {
        return weights_.size();
    }

    int OriCount() const
    {
        return oriVoteCount_;
    }

    //aggregate votes using meanshift
    void AggregateVotes(const VotesStatsElemT<ElemType,S> &stats, mean_shift::MeanShift &mshift){
        //allocate data

        oriVoteCount_ = stats.Count();
        //std::cerr << "starting votes aggregation: " << oriVoteCount_ << std::endl;

        typename VotesStatsElemT<ElemType,S>::element_count votesCount = oriVoteCount_;
        mean_shift::ElemType *data = new mean_shift::ElemType[votesCount*S];

        flann::Matrix<mean_shift::ElemType> votes(data,votesCount,S);
        mean_shift::MatrixRow wrapper(votes);
        int row = 0;
        //fill the matrix
        //std::cerr << "copying votes: " << votesCount << std::endl;
        for(typename VotesStatsElemT<ElemType,S>::const_iterator i = stats.begin(); i!= stats.end(); i++){
            convert((*i),wrapper.setRow(row));
            row++;

        }
        //std::cerr << "row: " <<row <<  " votes count: " << votesCount << std::endl;

        mshift.setPoints(votes);
        //std::cerr << "running mean shift" << std::endl;

        try{
            mshift.run();
        //    std::cerr << "clusters detected: " <<mshift.getClusterNumber() << std::endl;
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

        delete [] centers.ptr();
        delete [] sizes.ptr();

        delete [] data;

    }

    void Serialize(std::ostream &out) const
    {
        int elemCount = weights_.size();
        out.write((const char*)&elemCount,sizeof(elemCount));
        int dims = S;
        out.write((const char*)&dims,sizeof(dims));
        out.write((const char *)&oriVoteCount_,sizeof(oriVoteCount_));

        for(int i=0; i<votes_.size(); i++){
            for(int j=0; j<S; j++){
                out.write((const char *)(&(votes_[i][j])),sizeof(ElemType));
            }
        }
        for(int i=0; i<weights_.size(); i++){
            out.write((const char *)&(weights_[i]),sizeof(double));
        }
    }

    void Deserialize(std::istream &in)
    {
        int elemCount;
        in.read((char *)&elemCount,sizeof(elemCount));
        weights_.resize(elemCount);
        votes_.resize(elemCount);
        int dims;
        in.read((char *)&dims,sizeof(dims));
        in.read((char *)&oriVoteCount_,sizeof(oriVoteCount_));
        for(int i=0; i<votes_.size(); i++){
            for(int j=0; j<S; j++){
                in.read((char *)&(votes_[i][j]),sizeof(ElemType));
            }
        }
        for(int i=0; i<weights_.size(); i++){
            in.read((char *)&(weights_[i]),sizeof(double));
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

    void RescaleWeights(float scale)
    {
        for(int i=0; i< weights_.size(); i++){
            weights_[i] *= scale;
        }
    }

    void Clear()
    {
        votes_.clear();
        weights_.clear();
    }

private:

    void convert(const cv::Vec<ElemType,S> &from, mean_shift::MatrixRow &to)
    {
        //assume the sizes are correct
        for(int i=0; i< S; i++){
            to[i] = from[i];
        }
    }

    typename VotesStatsElemT<ElemType,S>::element_count oriVoteCount_;
    std::vector<double> weights_;
    std::vector<cv::Vec<ElemType,S> > votes_;
    std::vector<mean_shift::ElemType> prediction_;



};

#endif // VOTESAGGREGATOR_H
