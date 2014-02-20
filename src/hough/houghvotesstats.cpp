#include "houghvotesstats.h"

HoughVotesStats::HoughVotesStats(const cv::Size &s, unsigned char voteClass)
{
    mat_ = cv::Mat(s,CV_8UC4,0);
    voteClass_ = voteClass;
    outOfBoundaries_ = 0;
}

void HoughVotesStats::Aggregate(const cv::Point2i &abs, const VotesStats& stats)
{
    cv::Point2i tmp;

    for(VotesStats::const_iterator i = stats.begin(voteClass_); i!= stats.end(voteClass_); i++){
        tmp = abs+(*i);

        if(tmp.x < 0 | tmp.y < 0 | tmp.x >= mat_.cols | tmp.y >= mat_.rows)
        {
            outOfBoundaries_++;
        }
        else
        {
            mat_.at<unsigned int>(tmp.x,tmp.y)+=1;
        }
    }
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
        mat_.at<unsigned int>(tmp)++;
    }
}

bool HoughVotesStats::Serialize(std::string &filename)
{
    cv::Mat out;

    mat_.convertTo(out,CV_8UC1);

    cv::imwrite(filename,out);
}

bool HoughVotesStats::Serialize(std::ostream &out)
{
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


}
