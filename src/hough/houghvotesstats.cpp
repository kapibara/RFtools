#include "houghvotesstats.h"

HoughVotesStats::HoughVotesStats(const cv::Size &s, unsigned char voteClass)
{
    voteClass_ = voteClass;
    mat_ = cv::Mat(s.height,s.width,CV_8UC4);
    Clear();
}


void HoughVotesStats::Aggregate(const HoughVotesStats &stats)
{
    if(mat_.rows != stats.mat_.rows | mat_.cols != stats.mat_.cols){
        std::cerr << "internal dimensions do not match" << std::endl;
        return;
    }

    if(voteClass_ != stats.voteClass_){
        std::cerr << "voteClass does not match" << std::endl;
    }

    int rows = mat_.rows, cols = mat_.cols;

    if(mat_.isContinuous() & stats.mat_.isContinuous()){
        cols*=rows;
        rows=1;
    }

    unsigned int *thisptr;
    const unsigned int *statsptr;

    for(int i=0; i<rows;i++){
        thisptr = mat_.ptr<unsigned int>(i);
        statsptr = stats.mat_.ptr<unsigned int>(i);
        for(int j=0; j<cols; j++){
            thisptr[j]+=statsptr[j];
        }
    }
}

bool HoughVotesStats::Aggregate(const cv::Point2i &abs, const VotesStats& stats)
{
    if(!mat_.isContinuous()){
        std::cerr << "matrix is not continous" << std::endl;
    }

    cv::Point2i tmp;
    unsigned int *ptr = mat_.ptr<unsigned int>(0);
    bool tooBigValues = false;

    for(VotesStats::const_iterator i = stats.begin(voteClass_); i!= stats.end(voteClass_); i++){
        tmp = abs+(*i);

        if(tmp.x < 0 | tmp.y < 0 | tmp.x >= mat_.rows | tmp.y >= mat_.cols)
        {
            outOfBoundaries_++;
        }
        else
        {
            if(tmp.x > 320 | tmp.y > 320){
                tooBigValues = true;
            }
            mat_.at<unsigned int>(tmp.x ,tmp.y ) +=1;
        }
    }

    return tooBigValues;
}

void HoughVotesStats::Aggregate(const cv::Point2i &abs, const cv::Point2i &vote)
{
    cv::Point2i tmp = abs + vote;

    if(tmp.x < 0 | tmp.y < 0 | tmp.x >= mat_.cols | tmp.y >= mat_.rows)
    {
        outOfBoundaries_++;
    }
    else
    {
        mat_.at<unsigned int>(tmp.x,tmp.y)+=1;
    }
}

bool HoughVotesStats::Serialize(const std::string &filename)
{
    cv::Mat out(mat_.rows,mat_.cols,CV_8UC1);
    int rows = mat_.rows, cols = mat_.cols;

    if(mat_.isContinuous() & out.isContinuous()){
        cols*=rows;
        rows=1;
    }

    unsigned int *matptr;
    unsigned char *outptr;

    unsigned int maxval = 0;
    for(int i=0; i<rows;i++){
        matptr = mat_.ptr<unsigned int>(i);
        for(int j=0; j<cols; j++){
            if(matptr[j]>maxval){
                maxval = matptr[j];
            }
        }
    }

    if(maxval>0){
        for(int i=0; i<rows;i++){
            matptr = mat_.ptr<unsigned int>(i);
            outptr = out.ptr(i);
            for(int j=0; j<cols; j++){
                outptr[j] = ((double)matptr[j])/maxval*255.0;
            }
        }
    }

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

    unsigned int *ptr;

    int rows=mat_.rows,cols=mat_.cols;
    if(mat_.isContinuous()){
        cols *= rows;
        rows = 1;
    }

    for(int i = 0; i< rows; i++)
    {
        ptr = mat_.ptr<unsigned int>(i);
        out.write((const char *)ptr,sizeof(unsigned int)*cols);
    }

    return true;
}
