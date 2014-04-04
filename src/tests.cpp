#include "classification/depthdb.h"
#include "Random.h"
#include "hough/votesstats.h"
#include "hough/houghvotesstats.h"

#include <fstream>

using namespace MicrosoftResearch::Cambridge::Sherwood;
using namespace std;

bool testDepthDBSubindex(DepthDBClassImage &db)
{

    try{
    std::vector<DepthFileBasedImageDB::index_type> index;

    Random random;
    int N = 5000;
    int ind;
    bool passed = true;

    for(int i=0; i< N; i++){
        ind = random.Next(0,db.Count()-1);
        index.push_back(ind);

    }

    std::cerr << "index vector is created" << std::endl;

    DepthDBSubindex test(db,index);

    std::cerr << "test db is created" << std::endl;

    cv::Point2i p_test,p;
    cv::Mat m_test,m;
    std::string filename_test,filename;
    ClassificationDB::label_type label_test,label;
    DepthFileBasedImageDB::fileindex_type fileindex_test,fileindex;

    if(test.Count() != index.size()){
        std::cerr << "test.Count() != index.size()" << std::endl;
        passed = false;
    }

    for(int i=0; i<test.Count(); i++){

        test.getDataPoint(i,m_test,p_test);
        db.getDataPoint(index[i],m,p);
        if(p_test!=p){
            std::cerr << "getDataPoint(..cv::Mat..) failed" << std::endl;
            passed = false;
        }

        test.getDataPoint(i,filename_test,p_test);
        db.getDataPoint(index[i],filename,p);
        if(p_test!=p | filename_test!=filename){
            std::cerr << "getDataPoint(..std::string..) failed" << std::endl;
            passed = false;
        }

        label_test=test.getNumericalLabel(i);
        label = db.getNumericalLabel(index[i]);
        if(label_test!=label){
            std::cerr << "getNumericalLabel() failed" << std::endl;
            passed = false;
        }

        fileindex_test= test.getOriginalImageIdx(i);
        fileindex = db.getImageIdx(index[i]);
        if(fileindex_test!=fileindex){
            std::cerr << "getOriginalImageIdx() failed" << std::endl;
            passed = false;
        }
    }

    return passed;

    }catch(std::exception &e){
        std::cerr << "exception caught: " << e.what() << std::endl;
    }

}

void testVotesStatsAndHoughStats()
{
    std::cerr << "starting tests" << std::endl;
    std::cerr << "test: vs1" << std::endl;

    VotesStats vs1(3);
    std::vector<cv::Point2i> votes;

    votes.push_back(cv::Point(1,1));
    votes.push_back(cv::Point(-1,0));
    votes.push_back(cv::Point(1,1));

    vs1.SetContainer(votes);
    vs1.Aggregate();

    votes[0] = cv::Point2i(2,2);
    votes[1] = cv::Point2i(-1,-1);
    votes[2] = cv::Point2i(1,1);

    vs1.SetContainer(votes);
    vs1.Aggregate();

    votes[0] = cv::Point2i(3,3);
    votes[1] = cv::Point2i(0,-1);
    votes[2] = cv::Point2i(1,1);

    vs1.SetContainer(votes);
    vs1.Aggregate();

    std::cerr << "vs1.vv(): " << vs1.VoteVariance() << std::endl;

    std::ofstream outvs1("vs1");
    vs1.Serialize(outvs1);
    outvs1.close();


    std::cerr << "stats size: " << vs1.Count() << std::endl;

    std::cerr << "finalizing distribution" << std::endl;

    vs1.FinalizeDistribution(cv::Size(100,200));

    std::cerr << "test: vs2" << std::endl;

    VotesStats vs2(3);

    votes[0] = cv::Point2i(-4,-4);
    votes[1] = cv::Point2i(0,0);
    votes[2] = cv::Point2i(-3,-2);

    vs2.SetContainer(votes);
    vs2.Aggregate();

    votes[0] = cv::Point2i(-4,-4);
    votes[1] = cv::Point2i(0,0);
    votes[2] = cv::Point2i(-3,-3);

    vs2.SetContainer(votes);
    vs2.Aggregate();

    votes[0] = cv::Point2i(-4,-4);
    votes[1] = cv::Point2i(0,1);
    votes[2] = cv::Point2i(-2,-3);

    vs2.SetContainer(votes);
    vs2.Aggregate();

    std::cerr << "vs2.vv(): " << vs2.VoteVariance() << std::endl;

    std::ofstream outvs2("vs2");
    vs2.Serialize(outvs2);
    outvs2.close();

    std::cerr << "finalizing distribution" << std::endl;

    vs2.FinalizeDistribution(cv::Size(100,200));

    std::cerr << "creating hvstats1" << std::endl;

    HoughVotesStats hvstats1(cv::Size(20,20),0);

    hvstats1.Aggregate(cv::Point2i(19,19),vs1);
    hvstats1.Aggregate(cv::Point2i(19,19),vs2);

    std::ofstream out1("hvstats1");
    hvstats1.Serialize(out1);

    std::cerr << "creating hvstats2" << std::endl;

    HoughVotesStats hvstats2(cv::Size(20,20),0);

    hvstats2.Aggregate(cv::Point2i(10,10),vs1);
    hvstats2.Aggregate(cv::Point2i(5,5),vs2);

    std::ofstream out2("hvstats2");
    hvstats2.Serialize(out2);
}

class A
{
public:
    A(int i){
        std::cerr << "A()" << std::endl;
        i_ = i;
    }

    A(const A &){
        i_ = 0;
        std::cerr << "A(A)" << std::endl;
    }
    ~A(){
        std::cerr << "~A():" << i_ << std::endl;
    }

private:
    int i_;
};

A foo(){
    A a(10);

    return a;
}

int main(int argc, char **argv)
{

    A a(11);
    A b = foo();

    return 0;
/*    DepthDBClassImage db;
    db.loadDB(argv[1]);
    testDepthDBSubindex(db);*/

/*    VotesStats stats(1);
    DepthDBWithVotesImpl db;
    db.loadDB(argv[1]);

    //same image
    stats.Aggregate(db,0);
    stats.Aggregate(db,1);
    stats.Aggregate(db,2);

    std::cout << "VoteVariance: " << stats.VoteVariance() << std::endl;*/

    testVotesStatsAndHoughStats();
}
