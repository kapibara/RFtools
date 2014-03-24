#ifndef NODEDISTRIBUTIONIMAGESTATS_H
#define NODEDISTRIBUTIONIMAGESTATS_H

#include <vector>
#include <opencv2/opencv.hpp>

class NodeDistributionImagestats
{
public:
    NodeDistributionImagestats(cv::Size imageSize, const std::vector<int> &indices);
    NodeDistributionImagestats(cv::Size imageSize, int index);

    NodeDistributionImagestats(const NodeDistributionImagestats &current);

    ~NodeDistributionImagestats()
    {
        delete [] buffer_;
    }

    void addPoint(cv::Point2i p, int value);

    void Serialize(const std::string &filename) const;
    void Serialize(std::ostream &out) const;

private:

    int sub2ind(int i, int j) const
    {
      return i + j*size_.width;
    }

    //assume that the number of indices is less then 255
    uchar *buffer_;
    cv::Size size_;

    //this will not be too big..
    std::vector<int> indices_;

};

#endif // NODEDISTRIBUTIONIMAGESTATS_H
