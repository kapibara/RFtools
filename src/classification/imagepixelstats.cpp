#include "imagepixelstats.h"

bool ImagePixelStats::Serialize(std::ostream &stream) const{

    unsigned int st = pixels_.size();
    if(st==0){
        return false;
    }

    stream.write((const char *)(&st),sizeof(st));
    int x, y;

    for(pToStatsMap::const_iterator itor = pixels_.begin(); itor!=pixels_.end(); itor++)
    {
        x = (itor->first).x;
        y = (itor->first).y;
        stream.write((const char *)(&x),sizeof(x));
        stream.write((const char *)(&y),sizeof(y));
        (itor->second).Serialize(stream);
    }

    return true;
}

bool ImagePixelStats::Serialize(const std::string &filename) const{

    if(pixels_.size()==0){
        return false;
    }

    int minx=2000,maxx=0,miny=2000,maxy=0;

    for(pToStatsMap::const_iterator itor = pixels_.begin(); itor!=pixels_.end(); itor++){
        if ((itor->first).x < minx)
            minx = (itor->first).x;
        if ((itor->first).x > maxx)
            maxx = (itor->first).x;
        if((itor->first).y < miny)
            miny = (itor->first).y;
        if((itor->first).y > maxy)
            maxy = (itor->first).y;
    }

    cv::Mat out(maxx-minx+3,maxy-miny+3,CV_8UC3,cv::Scalar(0,0,0));

    for(pToStatsMap::const_iterator itor = pixels_.begin(); itor!=pixels_.end(); itor++){
        out.at<cv::Vec3b>((itor->first).x-minx+1,(itor->first).y-miny+1)=toColor((itor->second).ClassDecision());
    }
    bool result;

    if(!(result = cv::imwrite(filename,out))){
        std::cerr << "could not save png..." << std::endl;
    }

    std::cerr << "result: " << result << std::endl;

    return result;
}

cv::Vec3b ImagePixelStats::toColor(unsigned short val) const {

    return palette_[val];
}
