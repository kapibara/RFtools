#include "depthdb.h"

#include <fstream>
#include <iostream>
#include <map>

#include "split.h"

using namespace std;

void LabelPerImageStringParser::setString(const std::string &str){
    strsplit_.clear();
    split(str,",",strsplit_);
}


DepthDBClassImage::DepthDBClassImage(const std::string &basepath):
    DepthFileBasedImageDBImpl(basepath,true)

{

}

bool DepthDBClassImage::loadDB(const std::string &filename)
{
    return DepthFileBasedImageDBImpl::loadDB(filename,LabelPerImageStringParser());
}

bool DepthDBClassImage::postprocessFile(const cv::Mat &image, GeneralStringParser &parser){
    DepthFileBasedImageDBImpl::postprocessFile(image,parser);

    LabelPerImageStringParser &typedparser = dynamic_cast<LabelPerImageStringParser &>(parser);

    /*add labels*/
    std::pair<std::map<std::string,label_type>::iterator,bool> itor = labels_.insert(labelmap_type(typedparser.getLabel(),labels_.size()));
    datalabels_.push_back(itor.first->second);

    return true;
}




