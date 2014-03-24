#include "colorgenerator.h"

void generatePalette(std::vector<cv::Vec3b> &palette, int count)
{

    int color_divs = floor(pow((double)count,1/3.0))+1;
    int r,g,b;
    for (int i=0; i< count; i++){
        r = ((i)%(color_divs*color_divs))%color_divs+1;
        g = ((i)%(color_divs*color_divs))/color_divs+1;
        b = (i)/(color_divs*color_divs)+1;
        palette.push_back(cv::Vec3b(r*255/(color_divs),g*255/(color_divs),b*255/(color_divs)));
    }
}
