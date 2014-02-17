#include "votesstats.h"

void VotesStats::Aggregate(const MicrosoftResearch::Cambridge::Sherwood::IDataPointCollection& data, unsigned int index)
{
    DepthDBWithVotes &db = (DepthDBWithVotes &)data;
    std::vector<cv::Point2i> vote;

    db.getDataPointVote(index,vote);

    /*we suppose: voteClasses_ = vote.size()*/

    for(int i=0; i<voteClasses_; i++){
        if (norm2(vote[i].x,vote[i].y) < dthreashold2_){
            votes_[i].push_back(vote[i]);
        }
    }

    pointCount_++;
}

void VotesStats::Aggregate(const VotesStats& stats)
{
    for(int i=0; i<voteClasses_; i++){
        votes_[i].insert(votes_[i].end(),stats.votes_[i].begin(),stats.votes_[i].end());
    }

    pointCount_+=stats.pointCount_;
}

void VotesStats::Compress()
{
    /*radical solution*/
    Clear();
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

