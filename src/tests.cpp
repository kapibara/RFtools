#include "classification/depthdb.h"
#include "Random.h"
#include "hough/votesstats.h"

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

int main(int argc, char **argv)
{
/*    DepthDBClassImage db;
    db.loadDB(argv[1]);
    testDepthDBSubindex(db);*/

    VotesStats stats(1);
    DepthDBWithVotesImpl db;
    db.loadDB(argv[1]);

    //same image
    stats.Aggregate(db,0);
    stats.Aggregate(db,1);
    stats.Aggregate(db,2);

    std::cout << "VoteVariance: " << stats.VoteVariance() << std::endl;
}
