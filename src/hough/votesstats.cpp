#include "votesstats.h"

//#define OLD_DESERIALIZATION

void VotesStats::Aggregate(MicrosoftResearch::Cambridge::Sherwood::IDataPointCollection& data, unsigned int index)
{
    DepthDBWithVotes &db = dynamic_cast<DepthDBWithVotes &>(data);
    variance_ = -1; //invalidate variance

    db.getDataPointVote(index,container_);

    Aggregate();
}

void VotesStats::SetContainer(std::vector<cv::Point2i> &input)
{
    for(int i=0; i<container_.size(); i++){
        container_[i] = input[i];
    }

}

void VotesStats::Aggregate()
{
    bool acceptedOnce = false;
    int x2,y2;

    try{
        for(int i=0; i<voteClasses_; i++){

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
    aggregationValid_ = true;
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

    unsigned char matrixStatsSize = matrixStats_.size();
    unsigned short rows,cols;

    stream.write((const char *)&matrixStatsSize,sizeof(matrixStatsSize));
    for(int i=0; i<matrixStatsSize;i++){
        rows = matrixStats_[i].rows;
        cols = matrixStats_[i].cols;
        stream.write((const char *)&rows,sizeof(rows));
        stream.write((const char *)&cols,sizeof(cols));
        serializeMatrix(stream,matrixStats_[i]);
        stream.write((const char *)&(centers_[i].x),sizeof(centers_[i].x));
        stream.write((const char *)&(centers_[i].y),sizeof(centers_[i].y));
    }

    return true;
}

void VotesStats::serializeMatrix(std::ostream &out,const cv::Mat &mat) const
{
    const unsigned char *ptr;
    for(int i=0; i< mat.rows; i++){

        ptr = mat.ptr(i);
        out.write((const char *)ptr,mat.cols*mat.elemSize());
    }
}

bool VotesStats::Deserialize(std::istream &stream)
{
    stream.read((char *)(&pointCount_),sizeof(pointCount_));
    stream.read((char *)(&voteClasses_),sizeof(voteClasses_));

    unsigned int vsize;
    cv::Point2i p;

    mx_.resize(voteClasses_,0);
    my_.resize(voteClasses_,0);
    mx2_.resize(voteClasses_,0);
    my2_.resize(voteClasses_,0);
    votesCount_.resize(voteClasses_,0);

    for(int i=0; i<voteClasses_;i++){
        votes_.push_back(voteVector());
        stream.read((char *)(&vsize),sizeof(vsize));
        for(int j=0; j<vsize; j++){
            stream.read((char *)(&(p.x)),sizeof(p.x));
            stream.read((char *)(&(p.y)),sizeof(p.y));
            mx_[i] += p.x;
            my_[i] += p.y;
            mx2_[i] += p.x*p.x;
            my2_[i] += p.y*p.y;
            votesCount_[i]++;

            votes_.back().push_back(p);
        }
    }

    unsigned char matrixStatsSize;
    unsigned short rows,cols;

    stream.read((char *)&matrixStatsSize,sizeof(matrixStatsSize));

    for(int i=0; i<matrixStatsSize ; i++){
        stream.read((char *)&rows,sizeof(rows));
        stream.read((char *)&cols,sizeof(cols));
        matrixStats_.push_back(cv::Mat(rows,cols,CV_16UC1));
        deserializeMatrix(stream,matrixStats_.back());
        centers_.push_back(cv::Point2i());
        stream.read((char *)&(centers_.back().x),sizeof(centers_.back().x));
        stream.read((char *)&(centers_.back().y),sizeof(centers_.back().y));
    }

    aggregationValid_ = true;
    variance_ = -1;

    return true;
}

void VotesStats::deserializeMatrix(std::istream &out, cv::Mat &mat)
{
    unsigned char *ptr;
    for(int i=0; i< mat.rows; i++){
        ptr = mat.ptr(i);
        out.read((char *)ptr,mat.cols*mat.elemSize());
    }
}

//aggregate votes distribution in a matrix
void VotesStats::FinalizeDistribution(cv::Size maxsize)
{
    cv::Point2i center(maxsize.width/2,maxsize.height/2);

    if(matrixStats_.size()==0){

        int min_x,min_y,max_x,max_y;
        const_iterator itor;
        unsigned short *ptr;



        for(int i = 0; i < voteClasses_; i++){

            cv::Mat m = cv::Mat(maxsize,CV_16UC1);
            m.setTo(0);
            ptr = m.ptr<unsigned short>(0);

            if(!m.isContinuous()){
                std::cerr << "non-continious matrix!" << std::endl;
            }

            itor = begin(i);

            if (itor != end(i)){

                min_x = (*itor).x;
                min_y = (*itor).y;
                max_x = (*itor).x;
                max_y = (*itor).y;

                ptr[point2index((*itor)+center,maxsize)]++;

                itor++;

                for(;itor!=end(i);itor++){
                    if((*itor).x < min_x)
                        min_x = (*itor).x;
                    if((*itor).y < min_y)
                        min_y = (*itor).y;
                    if((*itor).x > max_x)
                        max_x = (*itor).x;
                    if((*itor).y > max_y)
                        max_y = (*itor).y;
                    ptr[point2index((*itor)+center,maxsize)]++;
                }

                min_x += center.x;
                min_y += center.y;
                max_x += center.x;
                max_y += center.y;

                matrixStats_.push_back(m(cv::Range(min_y,max_y+1),cv::Range(min_x,max_x+1)));
                centers_.push_back(center - cv::Point2i(min_x,min_y));
            }else{
                matrixStats_.push_back(cv::Mat());
                centers_.push_back(cv::Point2i(0,0));
            }
        }
    }
}

double VotesStats::VoteVariance()
{
    if (variance_ < 0){
        double mx,my,fulld2;

        fulld2 = 0;

        for(int i=0; i<voteClasses_; i++){


            if(votesCount_[i] > 0){

                mx = mx_[i]/votesCount_[i];
                my = my_[i]/votesCount_[i];


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
        }

        variance_ = fulld2;
    }

    if (isnan(variance_)){
        std::cerr << "variance_: " << variance_ << std::endl;
    }

    return variance_;

}

double VotesStats::normalizedVoteVariance()
{
    double mx,my,fulld2;

    fulld2 = 0;


    for(int i=0; i<voteClasses_; i++){

        if(votesCount_[i] > 1) {

            mx = mx_[i]/votesCount_[i];
            my = my_[i]/votesCount_[i];
            fulld2 += (mx2_[i] - mx*mx_[i])/(votesCount_[i]-1) + (my2_[i] - my*my_[i])/(votesCount_[i]-1);

        }
    }

    return fulld2;
}


