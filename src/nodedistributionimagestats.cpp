#include "nodedistributionimagestats.h"

#include "colorgenerator.h"

NodeDistributionImagestats::NodeDistributionImagestats(cv::Size imageSize, const std::vector<int> &indices)
{
    indices_ = indices;
    buffer_ = new uchar[imageSize.height*imageSize.width];
    memset(buffer_,0,imageSize.height*imageSize.width);
    size_ = imageSize;
}

NodeDistributionImagestats::NodeDistributionImagestats(cv::Size imageSize, int index)
{
    indices_.push_back(index);
    buffer_ = new uchar[imageSize.height*imageSize.width];
    memset(buffer_,0,imageSize.height*imageSize.width);
    size_ = imageSize;
}

NodeDistributionImagestats::NodeDistributionImagestats(const NodeDistributionImagestats &current)
{
    indices_ = current.indices_;
    size_ = current.size_;
    buffer_ = new uchar[size_.height*size_.width];
    memset(buffer_,0,size_.height*size_.width);
}

void NodeDistributionImagestats::addPoint(cv::Point2i p, int value)
{
    int i = 0;
    uchar result = 1;
    while(i< indices_.size() & indices_[i]!=value){
        i++;
    }

    if(i == indices_.size()){
        //label is not interesting
        result = indices_.size()+1;
    }else {
        //we want indices from 1 to indices_.size()
        result = i+1;
    }

    // 0 is zero

    buffer_[sub2ind(p.x,p.y)] = result;

}

void NodeDistributionImagestats::Serialize(const std::string &filename) const
{
    std::vector<cv::Vec3b> palette;

    std::cerr << "before generating palette" << std::endl;

    generatePalette(palette,indices_.size());

    std::cerr << "" << std::endl;

    palette.push_back(cv::Vec3b(255,255,255)); //uninteresting pixels
    cv::Mat m(size_,CV_8UC3);
    m.setTo((0,0,0));

    std::cerr << "before cycle" << std::endl;

    for(int i=0; i<size_.width; i++){
        for(int j=0; j< size_.height; j++){
            if (buffer_[sub2ind(i,j)] > 0){
                m.at<cv::Vec3b>(j,i) = palette[buffer_[sub2ind(i,j)]-1];
            }

        }
    }

    cv::imwrite(filename,m);
}
void NodeDistributionImagestats::Serialize(std::ostream &out) const
{

}
