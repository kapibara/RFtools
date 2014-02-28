#include "testclassificationforest.h"
#include "rfutils.h"
#include "time.h"

#include "cltrainingcontext.h"

#include "ForestTrainer.h"



TestClassificationForest::TestClassificationForest()
{
}

void TestClassificationForest::addParameterSet(const ExtendedTrainingParameters &param)
{
    params_.push_back(param);
}

void TestClassificationForest::test(DepthDBClassImage *db)
{
    std::auto_ptr<Forest<DepthFeature, ClassStats> > forest;
    std::auto_ptr<DepthDBSubindex> test;
    std::auto_ptr<DepthDBSubindex> train;
    Random  random;
    double result;
    time_t start,end;

    for(int i=0; i< params_.size(); i++){
        LocalCache cache("TestClassificationForest","/home/kuznetso/tmp");
        cache.init();

        std::ostream &log = cache.log();

        log << params_[i];

        RFUtils::splitRandom<DepthDBSubindex ,DepthFileBasedImageDB>(random,*db,train,test);

        log << "Train samples: " << train->Count() << std::endl;
        log << "Test samples: " << test->Count() << std::endl;

        DepthFeatureFactory factory(params_[i].paramFeatures_);
        ClTrainingContext context(db->classCount(),factory);

        log << "start forest training ... " << std::endl;

        time_t start,end;
        ProgressStream ps(log,Verbose);

        time(&start);

        forest = ForestTrainer<DepthFeature, ClassStats>::TrainForest (
                random, params_[i].paramForest_, context, *train, &ps );

        time(&end);
        double dif = difftime (end,start);

        log << "Training time: " << dif << " s" << std::endl;

        std::ostream &out = cache.openBinStream("forest");
        forest->Serialize(out);

        log << "Forest saved" << std::endl;

        ClassStats clStatsPixel(db->classCount());

        result = RFUtils::testClassificationForest<DepthFeature,ClassStats>
                (*forest,clStatsPixel,*train,cache,false);

        log << "Train error: " << result << std::endl;

        result = RFUtils::testClassificationForest<DepthFeature,ClassStats>
                (*forest,clStatsPixel,*test,cache,false);

        log << "Test error: " << result << std::endl;

        for(int i=0;i<db->classCount();i++){
            log << db->labelIndex2Name(i) << "-" << (int)i << std::endl;
        }

    }
}
