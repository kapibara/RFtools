#include "configuration.h"

#include "split.h"

#include <boost/property_tree/xml_parser.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>

/*copying too muc
 * but it is not performace - crilical*/
std::string trim_with_return(std::string in)
{
    boost::algorithm::trim(in);
    return in;
}



Configuration::Configuration()
{

}

Configuration::Configuration(std::istream &input)
{
    bpt::ptree tree;
    bpt::read_xml(input,tree);

    bpt::ptree subtree = tree.get_child("forest");
    forestType_ = trim_with_return(subtree.get<std::string>("type"));
    cacheFolderName_ = trim_with_return(subtree.get<std::string>("folder"));

    bpt::ptree properties =  subtree.get_child("properties");
    dbFile_ = trim_with_return(properties.get<std::string>("dbfile",""));
    testOnly_ = properties.get<int>("testonly",0)!=0;

    testOnTrain_ = (properties.get<int>("testontrain",0)!=0) & ~testOnly_;
    testOnTest_ = (properties.get<int>("testontest",1)!=0);

    dbHasHeader_ = properties.get<int>("dbhasheader",0)!=0;
    testTrainSplit_ = properties.get<float>("testtrainsplit",1.0);
    subsamplerRate_ = properties.get<float>("subsamplerrate",-1);
    varianceThr_ = properties.get<float>("nodevarthr",-1);
    sizeThr_ = properties.get<int>("nodesizethr",-1);
    voteDistThr_ = properties.get<float>("votethr",100);

    boost::optional<bpt::ptree &> meanshiftProps = properties.get_child_optional("meanshift");
    if (meanshiftProps){
        r_ = meanshiftProps.get().get<float>("bandwidth",100);
        maxIter_ = meanshiftProps.get().get<float>("maxiter",10);
        maxNN_ = meanshiftProps.get().get<float>("maxnn",5000);
        weightThr_ = meanshiftProps.get().get<float>("smallweight",0);
    }else{
        r_ = 100;
        maxIter_ = 10;
        maxNN_ = 5000;
        weightThr_ = 0;
    }

    if (testOnly_){
        readForestsList(properties);
    }else{
        serializeInfo_ = properties.get<int>("serializeinfo")>0;

        bpt::ptree forestProperties = properties.get_child("forestproperties");
        forestParam_.NumberOfTrees = forestProperties.get<int>("treecount");
        forestParam_.MaxDecisionLevels = forestProperties.get<int>("depth")-1;
        forestParam_.NumberOfCandidateFeatures = forestProperties.get<int>("featurespernode");
        forestParam_.NumberOfCandidateThresholdsPerFeature = forestProperties.get<int>("thrperfeature");
        gainType_ = forestProperties.get<std::string>("gaintype","variance");


        bpt::ptree featureproperties = properties.get_child("featureproperties");
        dfParams_.uvlimit_ = featureproperties.get<int>("uvlimit");
        dfParams_.zeroplane_ = featureproperties.get<int>("zeroplane");
        factory_ = parseFactoryName(trim_with_return(featureproperties.get<std::string>("factory")));

        if (factory_ == Configuration::FeaturePool){
            featuresLocation_ = trim_with_return(featureproperties.get<std::string>("featuresfile"));
        }
    }
}

Configuration::Factory Configuration::parseFactoryName(const std::string &factory)
{

    if (factory.compare("FullDepthFeatureFactory")==0)
        return Configuration::FullFeaturesFactory;
    if (factory.compare("PartialDepthFeatureFactory")==0)
        return Configuration::PartialFeaturesFactory;
    if (factory.compare("FeaturePool")==0)
        return Configuration::FeaturePool;

    return Configuration::Unknown;
}

void Configuration::readForestsList(bpt::ptree &props)
{
    BOOST_FOREACH(bpt::ptree::value_type const&val, props.get_child("forests"))
    {
        bpt::ptree forest = val.second;
        std::string ff = forest.get<std::string>("forestfile");
        std::string lf = forest.get<std::string>("leafsfile","");
        ForestParam fp = ForestParam(ff,lf);



        readFloatVector(forest.get<std::string>("bounds",""),fp.bounds_);

        readFloatVector(forest.get<std::string>("mean",""),fp.mean_);
        readFloatVector(forest.get<std::string>("std",""),fp.std_);

        forests_.push_back(fp);
    }
}

void Configuration::readFloatVector(const std::string &in, std::vector<float> &out)
{
    std::vector<std::string> nums;
    split(in,",",nums);

    for(int i=0; i< nums.size(); i++){
        out.push_back(boost::lexical_cast<float>(nums[i]));
    }
}
