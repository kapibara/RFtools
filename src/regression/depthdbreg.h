#ifndef DEPTHDBREG_H
#define DEPTHDBREG_H

#include "depthimagedb.h"
#include "split.h"
#include "string2number.hpp"

#define WITH_BOOST
#define WITH_OPENCV

#include <calibration.h>

template<typename ElemType, unsigned int S>
class PointsSParser: public GeneralStringParser
{
public:
   virtual std::string getFilename(){
        return strsplit_[0];
   }

   void getJoints(std::vector<cv::Vec<ElemType,S> > &out){
        cv::Vec<ElemType,S> tmp;
        for(int i=1; i<strsplit_.size();i+=S){
            for(int j=0; j<S;j++){
                tmp[j] = str2num<ElemType>(strsplit_[i+j]);
            }
            out.push_back(tmp);
        }
   }

   virtual void setString(const std::string &str){
        strsplit_.clear();
        split(str,",",strsplit_);
    }

private:
    std::vector<std::string> strsplit_;
};

template <typename ElemType, unsigned int S>
class DepthDBWithVotesS
{
public:
    typedef unsigned char vote_class_count;
    virtual bool getDataPointVote(DepthFileBasedImageDB::index_type i, std::vector<cv::Vec<ElemType,S> > &vote) = 0;
    virtual vote_class_count voteClassCount() = 0;
};

template <typename ElemType, unsigned int S>
class SubindexDBWithVotesS: public SubindexFileBasedImageDB, public DepthDBWithVotesS<ElemType,S>
{
public:
    SubindexDBWithVotesS(DepthFileBasedImageDB &source, const std::vector<index_type> &subindex){
        SubindexFileBasedImageDB(source,subindex);
    }

    bool getDataPointVote(DepthFileBasedImageDB::index_type i, std::vector<cv::Vec<ElemType,S> > &vote){
        return dynamic_cast<DepthDBWithVotesS<ElemType,S> &>(source_).getDataPointVote(subindex_[i], vote);
    }

    typename DepthDBWithVotesS<ElemType,S>::vote_class_count voteClassCount(){
        return dynamic_cast<DepthDBWithVotesS<ElemType,S> &>(source_).voteClassCount();
    }

};

template <typename ElemType,unsigned int S>
class DepthDBWithVotesSImpl: public DepthFileBasedImageDBImpl, public DepthDBWithVotesS<ElemType,S>
{
public:

    DepthDBWithVotesSImpl(const std::string &basepath="")
    {
        voteClassCount_ = 0;
    }

    bool loadDB(const std::string &filename, bool hasHeader)
    {
        PointsSParser<ElemType,S> parser;

        bool result =  DepthFileBasedImageDBImpl::loadDB(filename, parser,hasHeader);

        if(!hasHeader){
            isRelative_.resize(voteClassCount_,true);
        }

        return result;
    }

    const std::vector<cv::Vec<ElemType,S> > & getGT(fileindex_type i) const{
        return votes_[i];
    }

    using DepthFileBasedImageDBImpl::getDataPoint;

    bool getDataPoint(index_type i, cv::Vec<ElemType,S> &coord3D){
        cv::Point2i coord2D;
        cv::Mat img;

        bool gdpresult = DepthFileBasedImageDBImpl::getDataPoint(i,img,coord2D);
        unsigned short Ival = img.at<unsigned short>(coord2D);
        coord3D = to3D(coord2D,Ival);

        return gdpresult;
    }

    bool getDataPointVote(index_type i, std::vector<cv::Vec<ElemType,S> > &vote){

            if (vote.size() != voteClassCount_){
                std::cerr << "vote container does not have size " << (int)voteClassCount_ << std::endl;
                std::cerr.flush();
            }

            filebased_type pair = getIndex(i);
            cv::Point2i coord2D;
            cv::Mat img;
            cv::Vec<ElemType,S> coord;


            DepthFileBasedImageDBImpl::getDataPoint(i,img,coord2D);

            if (S==2){
                coord[0] = coord2D.x;
                coord[1] = coord2D.y;
            }

            if (S==3){
                unsigned short Ival = img.at<unsigned short>(coord2D);
                coord = to3D(coord2D,Ival);
            }



            for(int i=0; i<votes_[pair.first].size();i++){
                if (isRelative_[i])
                    vote[i] = votes_[pair.first][i]-coord;
                else{
                    vote[i] = votes_[pair.first][i];
                }

            }

            return true;


    }

    cv::Vec<ElemType,S> to3D(const cv::Point2i &p, unsigned short val){
        cv::Point2f np;
        calib_.Proj2Norm(p,np);
        return cv::Vec<ElemType,S>(np.x*val,np.y*val,val);
    }

    void setCalibParam(Calibration calib){
        calib_ = calib;
    }

    typename DepthDBWithVotesS<ElemType,S>::vote_class_count voteClassCount(){
        return voteClassCount_;
    }


protected:
    bool postprocessFile(const cv::Mat &image, GeneralStringParser &parser)
    {

        /*put all pixels into the array*/
        DepthFileBasedImageDBImpl::postprocessFile(image,parser);


        PointsSParser<ElemType,S> &typedparser = dynamic_cast<PointsSParser<ElemType,S> &>(parser);
        std::vector<cv::Vec<ElemType,S> > joints;


        /*add joint locations*/
        typedparser.getJoints(joints);

        votes_.push_back(joints);

        if (voteClassCount_<joints.size())
            voteClassCount_ = joints.size();

        return true;
    }

    void processHeader(const std::string &header)
    {
        std::vector< std::string > splitResult;

        split(header,",",splitResult);

        std::cerr << splitResult.size() << std::endl;

        isRelative_.resize((splitResult.size()-1)/S,true);

        for(int i=1; i<splitResult.size(); i+=S){
            std::cerr << "vote index: " << (i-1)/S << " splitResult[i]: " << splitResult[i] << std::endl;
            if(strcmp(splitResult[i].c_str(),"a") == 0){
                isRelative_[(i-1)/S] = false;
            }
        }
    }

private:
    Calibration calib_;

    typename DepthDBWithVotesS<ElemType,S>::vote_class_count voteClassCount_;

    std::vector<int> isRelative_;
    std::vector<std::vector<cv::Vec<ElemType,S> > > votes_; //stores joint locations for each image

};

#endif // DEPTHDBREG_H
