#ifndef CALIBRATION_H
#define CALIBRATION_H

#include <ostream>
#include <istream>

#define WITH_BOOST
#define WITH_OPENCV

#ifdef WITH_BOOST
#include <boost/property_tree/ptree.hpp>
#endif

#ifdef WITH_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace bpt = boost::property_tree;

class Calibration
{


public:
    friend std::ostream & operator<<(std::ostream &os, const Calibration& c){
        os << "(fx;fy):" << c.fx_ << ";" << c.fy_ << " (cx;cy):" << c.cx_ << ";" << c.cy_ << std::endl;
    }

    Calibration(){
        fy_ = fx_ =1;
        cx_ = cy_ = 0;
    }

    Calibration(double fx, double fy, double cx, double cy);

    void Serialize(std::ostream &out)
    {
        out.write((const char *)&fx_,sizeof(fx_));
        out.write((const char *)&fy_,sizeof(fy_));
        out.write((const char *)&cx_,sizeof(cx_));
        out.write((const char *)&cy_,sizeof(cy_));
    }

    static Calibration Deserialize(std::istream &in)
    {
        Calibration result;

        in.read((char *)&(result.fx_), sizeof(result.fx_));
        in.read((char *)&(result.fy_), sizeof(result.fy_));
        in.read((char *)&(result.cx_), sizeof(result.cx_));
        in.read((char *)&(result.cy_), sizeof(result.cy_));

        return result;
    }

    void Proj2Norm(float x, float y, float &nx, float &ny)
    {
        nx = (x - cx_)/fx_;
        ny = (y - cy_)/fy_;
    }

    void Norm2Proj(float nx, float ny, float &x, float &y)
    {
        x = nx*fx_ + cx_;
        y = ny*fy_ + cy_;
    }

#ifdef WITH_BOOST
    static Calibration Deserialize(bpt::ptree &calibtree)
    {
        Calibration result;

        result.fx_ = calibtree.get<float>("fx");
        result.fy_ = calibtree.get<float>("fy");
        result.cx_ = calibtree.get<float>("cx");
        result.cy_ = calibtree.get<float>("cy");

        return result;
    }
#endif

#ifdef WITH_OPENCV
    void Proj2Norm(const cv::Point2f &p, cv::Point2f &np)
    {
       Proj2Norm(p.x, p.y, np.x, np.y);
    }

    void Norm2Proj(const cv::Point2f &np, cv::Point2f &p)
    {
       Norm2Proj(np.x, np.y, p.x, p.y);
    }
#endif

private:
    float fx_, fy_ , cx_, cy_;
};

#endif // CALIBRATION_H
