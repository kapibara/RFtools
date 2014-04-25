#include "houghvotesstats.h"

HoughVotesStats::HoughVotesStats(const cv::Size &s, unsigned char voteClass, const cv::Point2i &center)
{
    voteClass_ = voteClass;
    mat_ = cv::Mat(s.height,s.width,CV_64FC1);
    center_ = center;
    Clear();
}

/*
bool HoughVotesStats::Aggregate(const HoughVotesStats &stats)
{
    //these constrains just make my programming life easier
    if(mat_.rows != stats.mat_.rows | mat_.cols != stats.mat_.cols){
        std::cerr << "internal dimensions do not match" << std::endl;
        return false;
    }

    if(voteClass_ != stats.voteClass_){
        std::cerr << "voteClass does not match" << std::endl;
        return false;
    }

    if(center_ != stats.center_){
        std::cerr << "centers are different" << std::endl;
        return false;
    }

    int rows = mat_.rows, cols = mat_.cols;

    if(mat_.isContinuous() & stats.mat_.isContinuous()){
        cols*=rows;
        rows=1;
    }

    mat_elem_type *thisptr;
    const mat_elem_type *statsptr;

    for(int i=0; i<rows;i++){
        thisptr = mat_.ptr<mat_elem_type>(i);
        statsptr = stats.mat_.ptr<mat_elem_type>(i);
        for(int j=0; j<cols; j++){
            thisptr[j]+=statsptr[j];
        }
    }
}
*/

bool HoughVotesStats::Aggregate(const cv::Point2i &abs,const VotesStats& stats)
{
    if(!mat_.isContinuous()){
        std::cerr << "matrix is not continous" << std::endl;
    }

    if(stats.votesPerVoteClass(voteClass_)==0){
        return true;
    }

    cv::Point2i center;
    cv::Point2i tmp;
    const cv::Mat &m = stats.Distribution(voteClass_,center);

    center =  abs - center + center_;

    //now iterate through the matrix
    for(int i=0; i<m.rows; i++){
        for( int j=0; j<m.cols; j++){
            tmp = cv::Point2i(j,i) + center;

            if(tmp.x < 0 | tmp.y < 0 | tmp.y >= mat_.rows | tmp.x >= mat_.cols)
            {
                outOfBoundaries_+=m.at<VotesStats::mat_elem_type>(cv::Point2i(j,i));
            }
            else
            {

                mat_.at<mat_elem_type>(tmp) += ((double)m.at<VotesStats::mat_elem_type>(cv::Point2i(j,i)))
                        /stats.votesPerVoteClass(voteClass_);
            }
        }
    }

    return false;
}
/*
bool HoughVotesStats::AggregateOld(const cv::Point2i &abs, const VotesStats& stats)
{
    std::cerr << "WARNING: AggregateOld is used; is slow; use Aggregate" << std::endl;


    if(!mat_.isContinuous()){
        std::cerr << "matrix is not continous" << std::endl;
    }

    cv::Point2i tmp;
    unsigned int *ptr = mat_.ptr<unsigned int>(0);
    bool tooBigValues = false;

    for(VotesStats::const_iterator i = stats.begin(voteClass_); i!= stats.end(voteClass_); i++){
        tmp = abs+(*i)+ center_;

        if(tmp.x < 0 | tmp.y < 0 | tmp.x >= mat_.rows | tmp.y >= mat_.cols)
        {
            outOfBoundaries_++;
        }
        else
        {
            if(tmp.x > 320 | tmp.y > 320){
                tooBigValues = true;
            }
            mat_.at<unsigned int>(tmp) +=1;
        }
    }

    return tooBigValues;
}
*/
void HoughVotesStats::Aggregate(const cv::Point2i &abs, const cv::Point2i &vote, double weight)
{
    cv::Point2i tmp = abs + vote + center_;

    if(tmp.x < 0 | tmp.y < 0 | tmp.x >= mat_.cols | tmp.y >= mat_.rows)
    {
        outOfBoundaries_++;
    }
    else
    {
        mat_.at<mat_elem_type>(tmp.x,tmp.y)+=weight;
    }
}

bool HoughVotesStats::Serialize(const std::string &filename)
{
    cv::Mat out(mat_.rows,mat_.cols,CV_8UC3);
    int rows = mat_.rows, cols = mat_.cols;
    out.setTo(cv::Vec3b(0,0,0));

    if(mat_.isContinuous() & out.isContinuous()){
        cols*=rows;
        rows=1;
    }

    mat_elem_type *matptr;
    unsigned char *outptr;

    mat_elem_type maxval = 0;
    for(int i=0; i<rows;i++){
        matptr = mat_.ptr<mat_elem_type>(i);
        for(int j=0; j<cols; j++){
            if(matptr[j]>maxval){
                maxval = matptr[j];
            }
        }
    }

    if(maxval>0){
        for(int i=0; i<rows;i++){
            matptr = mat_.ptr<mat_elem_type>(i);
            outptr = out.ptr(i);
            for(int j=0; j<cols; j++){
                outptr[3*j] = ((double)matptr[j])/maxval*255.0;
                outptr[3*j+1] = ((double)matptr[j])/maxval*255.0;
                outptr[3*j+2] = ((double)matptr[j])/maxval*255.0;
            }
        }
    }

    out.at<cv::Vec3b>(gt_ + center_) = cv::Vec3b(0,0,255);

    cv::imwrite(filename,out);
    return true;
}

bool HoughVotesStats::Serialize(std::ostream &out)
{
    out.write((const char *)(&outOfBoundaries_),sizeof(outOfBoundaries_));
    out.write((const char *)(&(gt_.x)),sizeof(gt_.x));
    out.write((const char *)(&(gt_.y)),sizeof(gt_.y));
    out.write((const char *)(&mat_.cols),sizeof(mat_.cols));
    out.write((const char *)(&mat_.rows),sizeof(mat_.rows));
    out.write((const char *)(&center_.x),sizeof(center_.x));
    out.write((const char *)(&center_.y),sizeof(center_.y));

    mat_elem_type *ptr;

    int rows=mat_.rows,cols=mat_.cols;
    if(mat_.isContinuous()){
        cols *= rows;
        rows = 1;
    }

    for(int i = 0; i< rows; i++)
    {
        ptr = mat_.ptr<mat_elem_type>(i);
        out.write((const char *)ptr,mat_.elemSize()*cols);
    }

    return true;
}
