#ifndef COLORGENERATOR_H
#define COLORGENERATOR_H

#include <vector>
#include <opencv2/opencv.hpp>

void generatePalette(std::vector<cv::Vec3b> &palette, int count);

#endif // COLORGENERATOR_H
