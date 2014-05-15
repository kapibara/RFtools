#ifndef VOTESSTATST_H
#define VOTESSTATST_H

#include <list>
#include <opencv2/opencv.hpp>
#include "Interfaces.h"
#include "regression/depthdbreg.h"

#include <math.h>

//#define ENABLE_OVERFLOW_CHECKS


template<class ElemType, int S>
void setTo(cv::Vec<ElemType,S> &vec, ElemType val)
{
    for(int i=0; i<S; i++){
        vec[i] = val;
    }
}

template<class ElemType, int S>
cv::Vec<ElemType,S> sqr(const cv::Vec<ElemType,S> &vec)
{
    cv::Vec<ElemType,S> result;
    for(int i=0; i<S; i++){
        result[i] = vec[i]*vec[i];
    }
    return result;
}


template<class ElemType, int S>
void setTo(cv::Matx<ElemType,S,S> &mat, ElemType val)
{
    for(int i=0; i<S; i++){
        for(int j=0; j<S; j++){
            mat(i,j) = val;
        }
    }
}

template <typename ElemType, int S>
class VotesStatsElemT;

//does it make sense to make S dynamic?
template <typename ElemType, int S>
class VotesStatsT
{
    typedef typename std::vector<VotesStatsElemT<ElemType, S> > elem_vector;

public:
    typedef unsigned int element_count;
    typedef typename elem_vector::const_iterator const_iterator;

    VotesStatsT(unsigned char voteElemCount = 0,unsigned int thr2 = 300*300)
    {
        voteElemCount_ = voteElemCount;
        elems_.resize(voteElemCount,VotesStatsElemT<ElemType, S> (thr2));
        container_.resize(voteElemCount);
        pointCount_ = 0;
    }

    void Clear()
    {
        for(int i = 0; i < voteElemCount_; i++){
            elems_[i].Clear();
        }
        pointCount_ = 0;
    }

    void FullStats(bool compute)
    {
        for(int i = 0; i < voteElemCount_; i++){
            elems_[i].FullStats(compute);
        }
    }

    const_iterator begin() const{
        return elems_.begin();
    }

    const_iterator end() const{
        return elems_.end();
    }


    void Aggregate(MicrosoftResearch::Cambridge::Sherwood::IDataPointCollection& data, unsigned int index)
    {

        DepthDBWithVotesS<ElemType, S> &db = dynamic_cast<DepthDBWithVotesS<ElemType, S> &>(data);

        db.getDataPointVote(index,container_);

        for(int i=0; i<voteElemCount_; i++){
            elems_[i].Aggregate(container_[i]);
        }

        pointCount_++;

    }

    void Aggregate(const VotesStatsT<ElemType,S >& stats){

        if (stats.voteElemCount_ != voteElemCount_){
            std::cerr << "different number of elements" << std::endl;
            return;
        }

        for(int i=0; i<voteElemCount_; i++){
            elems_[i].Aggregate(stats.elems_[i]);
        }

        pointCount_ += stats.pointCount_;

    }

    double VoteVariance(){
        double result=0;
        for(int i=0; i<voteElemCount_; i++){
           result += elems_[i].VoteVariance();
        }
        return result;
    }

    double LogEntropy(){
        double result=0;
        for(int i=0; i<voteElemCount_; i++){
            result += elems_[i].NEntropy();
        }
        return std::log(result);
    }

    element_count Count() const
    {
        return pointCount_;
    }

    element_count ElemCount() const
    {
        return voteElemCount_;
    }

    void Compress(){
        for(int i=0; i<voteElemCount_; i++){
            elems_[i].Compress();
        }
    }

    bool Serialize(std::ostream &stream) const
    {
        stream.write((const char *)(&voteElemCount_),sizeof(unsigned char));
        stream.write((const char *)(&pointCount_),sizeof(element_count));
        for(int i=0; i< voteElemCount_; i++){
            elems_[i].Serialize(stream);
        }
    }

    bool Deserialize(std::istream &stream)
    {
        stream.read((char *)(&voteElemCount_),sizeof(unsigned char));
        stream.read((char *)(&pointCount_),sizeof(element_count));
        VotesStatsElemT<ElemType, S> tmp;
        for(int i=0; i< voteElemCount_; i++){
            tmp.Deserialize(stream);
            elems_.push_back(tmp);
        }
    }

    VotesStatsT<ElemType,S> DeepClone() const
    {
        return VotesStatsT<ElemType,S>(*this);
    }



private:

    elem_vector elems_;
    std::vector<cv::Vec<ElemType,S> > container_;
    element_count pointCount_;
    unsigned char voteElemCount_;
};


template <typename ElemType, int S>
class VotesStatsElemT
{
    typedef std::list<cv::Vec<ElemType, S> > voteVector;

public:

    typedef unsigned int element_count;
    typedef typename voteVector::const_iterator const_iterator;

    VotesStatsElemT(unsigned int thr2 = 300*300)
    {
        dthreashold2_ = thr2;
        variance_ = -1;
        enthropy_ = -1;
        aggregationValid_ = true;
        fullStats_ = true;
        votesCount_= 0;

        setTo<double,S>(m_,0);
        setTo<double,S>(m2_,0);
    }

    void Clear()
    {
        setTo<double,S>(m_,0);
        setTo<double,S>(m2_,0);
        votesCount_ = 0;
        votes_.clear();

        variance_ = -1;
        enthropy_ = -1;
        aggregationValid_ = true;
    }

    void FullStats(bool compute)
    {
        fullStats_ = compute;
    }

    //+one more copying.. not so bad though
    void Aggregate(const cv::Vec<ElemType,S> &container){
        container_ = container;
        Aggregate();
    }



    element_count Count() const{
        return votesCount_;
    }

    const_iterator begin() const{
        return votes_.begin();
    }

    const_iterator end() const{
        return votes_.end();
    }

    void Compress(){
        votes_.clear();
        aggregationValid_ = false;
    }

    /*real variance*/
    double VoteVariance(){

        if(variance_ < 0){
            if(votesCount_ > 0){
                variance_ = 0;
                for(int i=0; i<S; i++){
                    variance_ += m2_(i,i)/votesCount_ - (m_[i]/votesCount_)*(m_[i]/votesCount_);
   //                 variance_ += m2_(i,i) - (m_[i]/votesCount_)*m_[i];
                }
                return (double)variance_;
            }else{
                return 0;
            }
        }else {
            return (double)variance_;
        }

    }

    /*determinant of enthropy*/
    double NEntropy(){
        if(enthropy_<0){
            if(votesCount_ > 0){
                cv::Matx<double,S,S> cov;

                //mean product:
                for(int i=0;i<S; i++){
                    for( int j=0;j<S;j++){
                        cov(i,j) = -(m_[i]/votesCount_)*(m_[j]/votesCount_);
                    }
                }

                cov += (m2_*(1/(double)votesCount_));
                enthropy_= cv::determinant(cov);

                return enthropy_;

            }else{
                return 0; //this is very false
            }
        }else{
            return enthropy_;
        }
    }

    bool Serialize(std::ostream &stream) const
    {
        int size = S;
        stream.write((const char *)(&votesCount_),sizeof(votesCount_));
        //write votes size (dimensions)
        stream.write((const char *)(&size),sizeof(size));

        size = votes_.size();
        //write votes_ array size
        stream.write((const char *)(&size),sizeof(size));
        for(const_iterator i=begin(); i!= end(); i++){
            for(int j=0; j<S; j++){

                stream.write((const char *)(&((*i)[j])),sizeof(ElemType));
            }
        }
    }

    bool Deserialize(std::istream &stream)
    {
        int size;
        stream.read((char *)(&votesCount_),sizeof(votesCount_));
        //read votes size (dimensions)
        stream.read((char *)(&size),sizeof(size));
        cv::Vec<ElemType,S> tmp;

        //read votes_ array size
        stream.read((char *)(&size),sizeof(size));
        for(int i=0; i<size; i++){
            for(int j=0; j<S; j++){
                stream.read((char *)(&(tmp[j])),sizeof(ElemType));
            }
            votes_.push_back(tmp);
        }

        RecomputeMM2();
    }

    void Aggregate(const VotesStatsElemT<ElemType,S>& stats)
    {
        variance_ = -1; //invalidate variance
        enthropy_ = -1;
        aggregationValid_ = false; //because some votes will not be added

#ifdef ENABLE_OVERFLOW_CHECKS
            if(votesCount_> (std::numeric_limits<element_count>::max() - stats.votesCount_)){
                std::cerr << "VotesStats::Aggregate(): pointCount_ stats overflow" << std::endl;
                std::cerr.flush();
            }
            for(int j=0; j<S; j++){
                if(m_[j]>std::numeric_limits<ElemType>::max() - stats.m_[j]){
                    std::cerr << "VotesStats::Aggregate(): mx2_ stats overflow" << std::endl;
                    std::cerr.flush();
                }
                for(int i=0; i<S; i++){
                    if(m2_(i,j)>std::numeric_limits<ElemType>::max() - stats.m2_(i,j)){
                        std::cerr << "VotesStats::Aggregate(): my2_ stats overflow" << std::endl;
                        std::cerr.flush();
                    }
                }
            }
#endif

        /*pre-compute variance*/
        votesCount_ += stats.votesCount_;
        m_ += stats.m_;
        m2_ += stats.m2_;

    }

protected:

    void RecomputeMM2()
    {
        setTo<double,S>(m_,0);
        setTo<double,S>(m2_,0);

        for(typename voteVector::iterator i = votes_.begin(); i != votes_.end(); i++){
            for(int k = 0; k< S; k++){
                for(int l = 0; l< S; l++){
                    m2_(k,l) += ((*i)[k])*((*i)[l]);
                }
            }
            m_ += (*i);
        }
    }

    //do not call it without setting a proper contaier!
    void Aggregate()
    {
        cv::Matx<ElemType,S,S> v2;
        unsigned int sum;

        try{
            sum = 0;
            //compute sum(xy)
            for(int i = 0; i< S; i++){
                for(int j = 0; j< S; j++){
#ifdef ENABLE_OVERFLOW_CHECKS
    if(container_[j] > sqrt(std::numeric_limits<ElemType>::max())){
        std::cerr << "VotesStatsT::Aggregate(): v2[] stats overflow" << std::endl;
        std::cerr.flush();
    }
#endif

                    v2(i,j) = (container_[j])*(container_[i]);
                    if(i==j){
                        sum += v2(i,j);
                    }
                }
            }

            if (sum < dthreashold2_){
                if(fullStats_)
                    votes_.push_back(container_);
#ifdef ENABLE_OVERFLOW_CHECKS
        if(votesCount_> (std::numeric_limits<element_count>::max() - 1)){
            std::cerr << "VotesStats::Aggregate(): pointCount_ stats overflow" << std::endl;
            std::cerr.flush();
        }
#endif
                votesCount_++;
                    /*pre-compute variance*/

#ifdef ENABLE_OVERFLOW_CHECKS
    for(int i = 0; i< S; i++){
        for(int j = 0; j< S; j++){
            if(m2_(i,j) > std::numeric_limits<ElemType>::max() - v2(i,j)){
                std::cerr << "VotesStatsT::Aggregate(): m2_ stats overflow" << std::endl;
                std::cerr.flush();
            }
        }
    }
#endif
                variance_ = -1;
                enthropy_ = -1;
                m_ += container_;
                m2_ += v2;

            }
        }
        catch(std::exception e){
            std::cerr << "exception caught during aggregation: "<< e.what() << std::endl;
        }

    }




private:

    voteVector votes_;
    element_count votesCount_;
    cv::Vec<double,S> m_; //average std::vector<cv::Vec< >> m_
    cv::Matx<double,S,S> m2_; //moments
    cv::Vec<ElemType,S> container_;

    unsigned int dthreashold2_;
    double variance_;
    double enthropy_;
    bool aggregationValid_; //votes_ contain all votes needed
    bool fullStats_; //store votes;
};

#endif // VOTESSTATST_H
