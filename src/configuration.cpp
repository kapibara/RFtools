#include "configuration.h"

#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/algorithm/string/trim.hpp>

/*copying too muc
 * but it is not performace - crilical*/
std::string trim_with_return(std::string in)
{
    boost::algorithm::trim(in);
    return in;
}


namespace bpt = boost::property_tree;

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
    dbFile_ = trim_with_return(properties.get<std::string>("dbfile"));
    testOnly_ = properties.get<int>("testonly")!=0;

    testOnTrain_ = (properties.get<int>("testontrain",0)!=0) & ~testOnly_;
    testOnTest_ = (properties.get<int>("testontest",1)!=0);

    dbHasHeader_ = properties.get<int>("dbhasheader",0)!=0;
    testTrainSplit_ = properties.get<float>("testtrainsplit");
    subsamplerRate_ = properties.get<float>("subsamplerrate",-1);
    varianceThr_ = properties.get<float>("nodevarthr",-1);
    voteDistThr_ = properties.get<float>("votethr",100);

    if (testOnly_){
        forestLocation_ = properties.get<std::string>("forestfile");
    }else{
        serializeInfo_ = properties.get<int>("serializeinfo")>0;

        bpt::ptree forestProperties = properties.get_child("forestproperties");
        forestParam_.NumberOfTrees = forestProperties.get<int>("treecount");
        forestParam_.MaxDecisionLevels = forestProperties.get<int>("depth")-1;
        forestParam_.NumberOfCandidateFeatures = forestProperties.get<int>("featurespernode");
        forestParam_.NumberOfCandidateThresholdsPerFeature = forestProperties.get<int>("thrperfeature");

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
