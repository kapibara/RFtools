#include "votesstats.h"

void VotesStats::Aggregate(MicrosoftResearch::Cambridge::Sherwood::IDataPointCollection& data, unsigned int index)
{
    DepthDBWithVotes &db = dynamic_cast<DepthDBWithVotes &>(data);
    variance_ = -1; //invalidate variance
    int x2,y2;
    bool acceptedOnce = false;

    db.getDataPointVote(index,container_);

    /*we suppose: voteClasses_ <= vote.size()*/

    try{
        for(int i=0; i<voteClasses_; i++){
/*            if(container_[i].x < minx_[i]){
                minx_[i] = container_[i].x;
            }
            if(container_[i].y < miny_[i]){
                miny_[i] = container_[i].y;
            }
            if(container_[i].x > maxx_[i]){
                maxx_[i] = container_[i].x;
            }
            if(container_[i].y > maxy_[i]){
                maxy_[i] = container_[i].y;
            }*/
            x2 = (container_[i].x)*(container_[i].x);
            y2 = (container_[i].y)*(container_[i].y);
            if (x2+y2 < dthreashold2_){
                if(fullStats_)
                    votes_[i].push_back(container_[i]);
                votesCount_[i]++;
                /*pre-compute variance*/

#ifdef ENABLE_OVERFLOW_CHECKS
    if(mx2_[i]>std::numeric_limits<double>::max() - x2){
        std::cerr << "VotesStats::Aggregate(): mx2_ stats overflow" << std::endl;
        std::cerr.flush();
    }
    if(my2_[i]>std::numeric_limits<double>::max() - y2){
        std::cerr << "VotesStats::Aggregate(): my2_ stats overflow" << std::endl;
        std::cerr.flush();
    }
#endif
                mx_[i] += container_[i].x;
                my_[i] += container_[i].y;
                mx2_[i] += x2;
                my2_[i] += y2;
                acceptedOnce = true;
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

    if (acceptedOnce)
        pointCount_++;


}

void VotesStats::Aggregate(const VotesStats& stats)
{
    variance_ = -1; //invalidate variance
    aggregationValid_ = false; //this aggregation does not copy votes -> therefore invalidate

    for(int i=0; i<voteClasses_; i++){
        //votes_[i].insert(votes_[i].end(),stats.votes_[i].begin(),stats.votes_[i].end());
        votesCount_[i] += stats.votesCount_[i];
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
/*        if(stats.minx_[i] < minx_[i]){
            minx_[i] = stats.minx_[i];
        }
        if(stats.miny_[i] < miny_[i]){
            miny_[i] = stats.miny_[i];
        }
        if(stats.maxx_[i] > maxx_[i]){
            maxx_[i] = stats.maxx_[i];
        }
        if(stats.maxy_[i] > maxy_[i]){
            maxy_[i] = stats.maxy_[i];
        }*/
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

void VotesStats::toMatrices()
{
    std::vector<int> minx(voteClasses_,0),miny(voteClasses_,0), maxx(voteClasses_,0), maxy(voteClasses_,0);
    cv::Mat tmp;

    for(int i=0; i<voteClasses_; i++){

        for(const_iterator j = begin(i); j!=end(i); j++ ){
            if ((*j).x<minx[i]){
                minx[i] = (*j).x;
            }
            if((*j).y<miny[i]){
                miny[i] = (*j).y;
            }
            if((*j).x>maxx[i]){
                maxx[i] = (*j).x;
            }
            if((*j).y>maxy[i]){
                maxy[i] = (*j).y;
            }
        }

        tmp = cv::Mat(maxx[i]-minx[i]+1,maxy[i]-miny[i]+1,CV_8UC4);
    }
}

void VotesStats::Compress()
{
    /*radical solution*/
    for(int i=0 ; i < voteClasses_; i++){
        votes_[i].clear();
    }
    aggregationValid_ = true;
}

bool VotesStats::SerializeChar(std::ostream &stream) const
{
    std::cerr << "SerializeChar is called..." << std::endl;

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

    if(!aggregationValid_){
        std::cerr << "votesstats does not have valid votes" << std::endl;
    }

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

    aggregationValid_ = true;
    variance_ = -1;

    return true;
}

double VotesStats::VoteVariance()
{
    if (variance_ < 0){
        double mx,my,fulld2;

        fulld2 = 0;

        for(int i=0; i<voteClasses_; i++){

            if(votesCount_[i] == 0){

                mx = 0;
                my = 0;

            }else{

                mx = mx_[i]/votesCount_[i];
                my = my_[i]/votesCount_[i];
            }

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
            fulld2 += (mx2_[i] - mx*mx_[i]) + (my2_[i] - my*my_[i]);
        }

        variance_ = fulld2;
    }

    if (isnan(variance_)){
        std::cerr << "variance_: " << variance_ << std::endl;
    }

    return variance_;

}

