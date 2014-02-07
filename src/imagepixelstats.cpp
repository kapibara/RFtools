#include "imagepixelstats.h"

bool ImagePixelStats::Serialize(std::ostream &stream) const{
    std::vector<unsigned short> result(clCount_);
    std::ostringstream ss;
    ClassStats stats(2);

    for(int i=0; i< result.size();i++){
        result[i]=0;
    }


    for(std::map<cv::Point2i,ClassStats,cvPointicompare>::const_iterator itor = pixels_.begin(); itor!=pixels_.end(); itor++){
        result[(itor->second).ClassDecision()]++;
        stats.Aggregate((itor->second));
    }

    for(int i=0; i<result.size(); i++){
        ss << result[i] << ";";
    }

    stream << ss.str() << std::endl;

    return true;
}

bool ImagePixelStats::Serialize(const std::string &filename) const{

    std::cerr << "pixels_.size()" << pixels_.size() << std::endl;
    std::cerr.flush();

    if(pixels_.size()==0){
        return false;
    }

    int minx=2000,maxx=0,miny=2000,maxy=0;

    for(std::map<cv::Point2i,ClassStats,cvPointicompare>::const_iterator itor = pixels_.begin(); itor!=pixels_.end(); itor++){
        if ((itor->first).x < minx)
            minx = (itor->first).x;
        if ((itor->first).x > maxx)
            maxx = (itor->first).x;
        if((itor->first).y < miny)
            miny = (itor->first).y;
        if((itor->first).y > maxy)
            maxy = (itor->first).y;
    }

    std::cerr  << "(" << minx << ";" << maxx << ")(" << miny << ";" << maxy << ")" << std::endl;

    cv::Mat out(maxx-minx+3,maxy-miny+3,CV_8UC3,cv::Scalar(0,0,0));

    for(std::map<cv::Point2i,ClassStats,cvPointicompare>::const_iterator itor = pixels_.begin(); itor!=pixels_.end(); itor++){
        out.at<cv::Vec3b>((itor->first).x-minx+1,(itor->first).y-miny+1)=toColor((itor->second).ClassDecision());
    }

    cv::imwrite(filename,out);

    return true;
}

cv::Vec3b ImagePixelStats::toColor(unsigned short val) const {

    return palette_[val];
}
