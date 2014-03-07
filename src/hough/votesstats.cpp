#include "votesstats.h"

void VotesStats::Aggregate(MicrosoftResearch::Cambridge::Sherwood::IDataPointCollection& data, unsigned int index)
{
    DepthDBWithVotes &db = dynamic_cast<DepthDBWithVotes &>(data);
    std::vector<cv::Point2i> vote;
    variance_ = -1; //invalidate variance

    db.getDataPointVote(index,vote);

    /*we suppose: voteClasses_ <= vote.size()*/

    try{
        for(int i=0; i<voteClasses_; i++){
            if (norm2(vote[i].x,vote[i].y) < dthreashold2_){
                votes_[i].push_back(vote[i]);
                /*pre-compute variance*/
#ifdef ENABLE_OVERFLOW_CHECKS
    if(mx2_[i]>std::numeric_limits<double>::max() - (vote[i].x)*(vote[i].x)){
        std::cerr << "VotesStats::Aggregate(): mx2_ stats overflow" << std::endl;
        std::cerr.flush();
    }
    if(my2_[i]>std::numeric_limits<double>::max() - (vote[i].y)*(vote[i].y)){
        std::cerr << "VotesStats::Aggregate(): my2_ stats overflow" << std::endl;
        std::cerr.flush();
    }
#endif
                mx_[i] += vote[i].x;
                my_[i] += vote[i].y;
                mx2_[i] += (vote[i].x)*(vote[i].x);
                my2_[i] += (vote[i].y)*(vote[i].y);
            }
        }
    }catch(std::exception e){
        std::cerr << "exception caught during aggregation: "<< e.what() << std::endl;
    }
#ifdef ENABLE_OVERFLOW_CHECKS
    if(pointCount_> (std::numeric_limits<element_count>::max() - 1)){
        std::cerr << "VotesStats::Aggregate(): pointCount_ stats overflow" << std::endl;
        std::cerr.flush();
    }
#endif

    pointCount_++;


}

void VotesStats::Aggregate(const VotesStats& stats)
{
    variance_ = -1; //invalidate variance

    for(int i=0; i<voteClasses_; i++){
        votes_[i].insert(votes_[i].end(),stats.votes_[i].begin(),stats.votes_[i].end());
        /*pre-compute variance*/
#ifdef ENABLE_OVERFLOW_CHECKS
    if(mx2_[i]>std::numeric_limits<double>::max() - stats.mx2_[i]){
        std::cerr << "VotesStats::Aggregate(): mx2_ stats overflow" << std::endl;
        std::cerr.flush();
    }
    if(my2_[i]>std::numeric_limits<double>::max() - stats.my2_[i]){
        std::cerr << "VotesStats::Aggregate(): my2_ stats overflow" << std::endl;
        std::cerr.flush();
    }
#endif
        mx_[i] += stats.mx_[i];
        my_[i] += stats.my_[i];
        mx2_[i] += stats.mx2_[i];
        my2_[i] += stats.my2_[i];
    }

#ifdef ENABLE_OVERFLOW_CHECKS
    if (pointCount_> (std::numeric_limits<element_count>::max() - stats.pointCount_)){
        std::cerr << "VotesStats::Aggregate(): pointCount_ stats overflow" << std::endl;
        std::cerr.flush();
    }
#endif

    pointCount_+=stats.pointCount_;
}

void VotesStats::Compress()
{
    /*radical solution*/
    for(int i=0 ; i < voteClasses_; i++){
        votes_[i].clear();
    }
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

        for(voteVector::const_iterator j =  votes_[i].begin(); j!= votes_[i].end(); j++){
            ss << (*j).x << ";" << (*j).y << std::endl;
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
        for(voteVector::const_iterator j =  votes_[i].begin(); j!= votes_[i].end(); j++){
            stream.write((const char *)(&((*j).x)),sizeof((*j).x));
            stream.write((const char *)(&((*j).y)),sizeof((*j).y));
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

double VotesStats::VoteVariance()
{
    if (variance_ < 0){
        double mx,my,d2,fulld2;

        fulld2 = 0;

        for(int i=0; i<voteClasses_; i++){

            mx = mx_[i]/votes_[i].size();
            my = my_[i]/votes_[i].size();

#ifdef ENABLE_OVERFLOW_CHECKS
    if (std::abs(mx_[i]) > (std::numeric_limits<double>::max()/std::abs(mx))){
        std::cerr << "VotesStats::Aggregate(): mx*mx_[i] stats overflow" << std::endl;
        std::cerr.flush();
    }
    if (std::abs(my_[i]) > (std::numeric_limits<double>::max()/std::abs(my))){
        std::cerr << "VotesStats::Aggregate(): my*my_[i] stats overflow" << std::endl;
        std::cerr.flush();
    }
    if ((mx2_[i] - mx*mx_[i]) > (std::numeric_limits<double>::max()- (my2_[i] - my*my_[i]))){
        std::cerr << "VotesStats::Aggregate(): ((mx2_[i] - mx*mx_[i]) + (my2_[i] - my*my_[i])) stats overflow" << std::endl;
        std::cerr.flush();
    }
    if (fulld2 > (std::numeric_limits<double>::max()- ((mx2_[i] - mx*mx_[i]) + (my2_[i] - my*my_[i])))){
        std::cerr << "VotesStats::Aggregate(): fulld2 stats overflow" << std::endl;
        std::cerr.flush();
    }
#endif
            /*optimized variance computation; otherwise n^2, performance killer*/
            fulld2 += ((mx2_[i] - mx*mx_[i]) + (my2_[i] - my*my_[i]));
        }

        variance_ = fulld2;
    }

    return variance_;

}

