#include "votesstats.h"

void VotesStats::Aggregate(MicrosoftResearch::Cambridge::Sherwood::IDataPointCollection& data, unsigned int index)
{
    DepthDBWithVotes &db = dynamic_cast<DepthDBWithVotes &>(data);
    std::vector<cv::Point2i> vote;

    db.getDataPointVote(index,vote);

    /*we suppose: voteClasses_ <= vote.size()*/

    for(int i=0; i<voteClasses_; i++){
        if (norm2(vote[i].x,vote[i].y) < dthreashold2_){
            votes_[i].push_back(vote[i]);
        }
    }

    pointCount_++;

#ifdef ENABLE_OVERFLOW_CHECKS
    if(pointCount_==0){
        std::cerr << "VotesStats::Aggregate(): stats overflow" << std::endl;
        std::cerr.flush();
    }
#endif
}

void VotesStats::Aggregate(const VotesStats& stats)
{
    for(int i=0; i<voteClasses_; i++){
        votes_[i].insert(votes_[i].end(),stats.votes_[i].begin(),stats.votes_[i].end());
    }

#ifdef ENABLE_OVERFLOW_CHECKS
    if ((pointCount_+stats.pointCount_)<pointCount_){
        std::cerr << "VotesStats::Aggregate(): stats overflow" << std::endl;
        std::cerr.flush();
    }
#endif

    pointCount_+=stats.pointCount_;
}

void VotesStats::Compress()
{
    /*radical solution*/
    Clear();
}

bool VotesStats::SerializeChar(std::ostream &stream) const
{
    std::ostringstream ss;
    ss << pointCount_ << std::endl;
    ss << (int)voteClasses_ << std::endl;

    unsigned int vsize;

    for(int i=0; i<voteClasses_;i++){
        vsize = votes_[i].size();
        ss << vsize << std::endl;

        for(int j=0; j<vsize; j++){
            ss << votes_[i][j].x << ";" << votes_[i][j].y << std::endl;
        }

    }

    stream << ss.str();
    return true;
}


bool VotesStats::Serialize(std::ostream &stream) const
{
    stream.write((const char *)(&pointCount_),sizeof(pointCount_));  
    stream.write((const char *)(&voteClasses_),sizeof(voteClasses_));

    unsigned int vsize;

    for(int i=0; i<voteClasses_;i++){
        vsize = votes_[i].size();
        stream.write((const char *)(&vsize),sizeof(unsigned int));
        for(int j=0; j<vsize; j++){
            stream.write((const char *)(&(votes_[i][j].x)),sizeof(votes_[i][j].x));
            stream.write((const char *)(&(votes_[i][j].y)),sizeof(votes_[i][j].y));
        }
    }

    return true;
}

bool VotesStats::Deserialize(std::istream &stream)
{
    stream.read((char *)(&pointCount_),sizeof(pointCount_));
    stream.read((char *)(&voteClasses_),sizeof(voteClasses_));

    unsigned int vsize;
    cv::Point2i p;

    for(int i=0; i<voteClasses_;i++){
        votes_.push_back(voteVector());
        stream.read((char *)(&vsize),sizeof(vsize));
        for(int j=0; j<vsize; j++){

            stream.read((char *)(&(p.x)),sizeof(p.x));
            stream.read((char *)(&(p.y)),sizeof(p.y));

            votes_.back().push_back(p);
        }
    }

    return true;
}

double VotesStats::VoteVariance() const
{
    double mx,my,d2,fulld2;

    fulld2 = 0;

    for(int i=0; i<voteClasses_; i++){
        /*compute mean*/
        mx=0;
        my=0;
        for(int j=0; j<votes_[i].size(); j++){
            mx += votes_[i][j].x;
            my += votes_[i][j].y;
        }
        mx = mx/votes_[i].size();
        my = my/votes_[i].size();

        /*compute average distance*/
        d2=0;
        for(int j=0; j<votes_[i].size(); j++){
            d2+=norm2((double)(votes_[i][j].x-mx),(double)(votes_[i][j].y-my));
        }

        fulld2 += d2;
    }

    return fulld2;
}

