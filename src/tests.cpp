#include "classification/depthdb.h"
#include "Random.h"
#include "hough/votesstats.h"
#include "hough/houghvotesstats.h"
#include "hough/depthdbwithvotes.h"

#include "regression/votesstatst.h"
#include "regression/depthdbreg.h"
#include "regression/votesaggregator.h"
#include "serialization.h"
#include "rfutils.h"
#include "featureaccomulator.h"

#include <fstream>
#include <memory>
#include <queue>
#include <cstdlib>

#include <time.h>

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

void testFileDBIndexing(SubindexFileBasedImageDB &db)
{
    std::vector<int> imgIds(db.Count(),0);
    std::vector<int> x(db.Count(),0);
    std::vector<int> y(db.Count(),0);

    std::string tmpstr;
    cv::Point2i current;

    for(int i=0; i<db.Count(); i++){
        db.getDataPoint(i,tmpstr,current);
        imgIds[i] = db.getImageIdx(i);
        x[i] = current.x;
        y[i] = current.y;
    }

    std::ofstream ids("imgIds",std::ios_base::binary);
    serializeVector<int>(ids,imgIds);
    ids.close();

    std::ofstream xvals("xVals",std::ios_base::binary);
    serializeVector<int>(xvals,x);
    xvals.close();

    std::ofstream yvals("yVals",std::ios_base::binary);
    serializeVector<int>(yvals,y);
    yvals.close();
}

void testCameraTransform()
{
    DepthDBWithVotesSImpl<float,3> db;
    db.loadDB("/home/kuznetso/tmp/HoughTests/Test3D/files.txt",true);
    cv::Mat m;
    cv::Point2i p2D;
    cv::Vec<float,3> p3D;
    std::vector<cv::Vec<float,3> > v;

    std::cout << "db image size: " << db.getSize() << std::endl;

    v.resize(1);

    for(int i=0; i< db.Count(); i++){
        db.getDataPoint(i,m,p2D);
        p3D = db.to3D(p2D,m.at<unsigned short>(p2D));
        db.getDataPointVote(i,v);

            std::cerr << "p2D: " << p2D << std::endl;
            std::cerr << "v[0]: " << v[0][0] << ";" << v[0][1] << ";" << v[0][2] << std::endl;
            std::cerr << "p3D: " << p3D << std::endl;
            std::cerr << "probably wrong calibration parameters" << std::endl;

    }
}

void test3DPoints()
{
    DepthDBWithVotesSImpl<float,3> db;
    //assume only one image in the files.txt"
    db.loadDB("/home/kuznetso/tmp/HoughTests/Head_NT3DTest/files.txt",true);
    cv::Vec<float,3> p3D;
    cv::Point2i p2D;
    cv::Mat m;
    std::vector<cv::Vec<float,3> > v;
    std::ofstream out("pc.csv");

    v.resize(2);

    std::cout << "size: " << db.Count() << std::endl;

    for(int i=0; i< db.Count(); i++){
        db.getDataPoint(i,m,p2D);
        p3D = db.to3D(p2D,m.at<unsigned short>(p2D));

        out << p3D[0] << "," << p3D[1] << "," << p3D[2] << std::endl;
    }

    //one image only..
    db.getDataPointVote(0,v);

    out << v[0][0] << "," << v[0][1] << "," << v[0][2] << std::endl;
    out << v[1][0] << "," << v[1][1] << "," << v[1][2] << std::endl;

    out.close();
}

void priorityQueuetest()
{
    std::priority_queue<FeatureAccomulator::FeatureGainType> pq;
    int num;

    std::srand (time(NULL));

    for(int i=10; i>-30; i--){
        num = rand() % 80 - 50;
        std::cout << "num: " << num << std::endl;
        if (pq.size()<5){
            pq.push(FeatureAccomulator::FeatureGainType(DepthFeature(),0,-num));
        }else{
            if(-num<pq.top().gain_){
                pq.pop();
                pq.push(FeatureAccomulator::FeatureGainType(DepthFeature(),0,-num));
            }
        }
    }

    int size = pq.size();
    while(!pq.empty()){
        std::cout << -pq.top().gain_ << std::endl;
        pq.pop();
    }
    std::cout << pq.empty() << std::endl;
}

void testVoteVarianceT()
{
    VotesStatsElemT<float,3> vv(10000*10000);
    cv::Vec<float,3> container;



    for(int i=0; i< 100000000; i++){
        container[0] = i/((double)325600);
        container[1] = i/((double)100000);
        container[2] = i/((double)111100);
        vv.Aggregate(container);
    }

    std::cerr << "variance: " << vv.VoteVariance() << std::endl;

}

void testAggregatorsDeserialization()
{
    std::ifstream in("/home/kuznetso/Projects/CPP/DepthRF/test/aggLeafs",std::ios_base::binary);
    std::ofstream out("/home/kuznetso/Projects/CPP/DepthRF/test/aggLeafsDes",std::ios_base::binary);
    VotesAggregator<float,3> tmp;
    std::vector<VotesAggregator<float,3> > result;

    while(!in.eof()){
        tmp.Deserialize(in);
        if(!in.eof()){
            tmp.Serialize(out);
            result.push_back(tmp);
        }
    }

    std::cerr << "result: " << result.size() << std::endl;
}

void testForestDeserialization()
{
    std::ifstream in("/home/kuznetso/Projects/CPP/DepthRF/test/forest",std::ios_base::binary);

    std::auto_ptr<Forest<DepthFeature,  VotesStatsT<float,3> > > forest = Forest<DepthFeature, VotesStatsT<float,3> >::Deserialize(in);

    std::ofstream out("/home/kuznetso/Projects/CPP/DepthRF/test/forestDes",std::ios_base::binary);

    forest->Serialize(out);
}

int main(int argc, char **argv)
{

    testForestDeserialization();
//    priorityQueuetest();

/*    cv::Vec<float,3> tocopy;
    tocopy[0] = 1;
    tocopy[1] = 2;
    tocopy[2] = 3;
    cv::Vec<float,3> copyto;

    copyto = tocopy;

    std::cerr << tocopy << std::endl;
    std::cerr << copyto << std::endl;*/

//    test3DPoints();

/*    DepthDBClassImage db;
    db.loadDB(argv[1]);
    testDepthDBSubindex(db);*/
/*
    Random random;
    DepthFileBasedImageDBImpl db;
    SimpleParser parser;
    db.loadDB("/home/kuznetso/tmp/HoughTests/Head_Full/files.txt",parser);
    std::auto_ptr<SubindexFileBasedImageDB> test;
    std::auto_ptr<SubindexFileBasedImageDB> train;
    RFUtils::splitRandom<SubindexFileBasedImageDB>(random,db,train,test,0.5);


    testFileDBIndexing(*test);*/
/*    //same image
    stats.Aggregate(db,0);
    stats.Aggregate(db,1);
    stats.Aggregate(db,2);

    std::cout << "VoteVariance: " << stats.VoteVariance() << std::endl;*/

 //   testVotesStatsAndHoughStats();
}
