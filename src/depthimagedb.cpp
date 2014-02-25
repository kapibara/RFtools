#include "depthimagedb.h"

#include <fstream>

DepthFileBasedImageDBImpl::DepthFileBasedImageDBImpl(const std::string &base,
                                             bool constImgSize)
{
    path_ = base;
    elementCount_ = 0;
    constImgSize_ = constImgSize;
    cachCallCount_ = 0;
}

bool DepthFileBasedImageDBImpl::loadDB(const std::string &filename, GeneralStringParser &stringParser){
    try{
        readFiles(filename,stringParser);
        return true;
    }catch(std::exception &e){
        std::cerr << "expcetion caught: " << e.what() << std::endl;
        return false;
    }
}

bool DepthFileBasedImageDBImpl::getDataPoint(unsigned int i, cv::Mat &img, cv::Point2i &coordinate)
{
    if (i >= pointsIndex_.size()){
        return false;
    }

    if (pointsIndex_[i].first != previous_)
    {
        //std::cerr << "loading from disk: " << i << std::endl;
        //the images is not in cache
        cachCallCount_++;
        cache_ = cv::imread(files_[pointsIndex_[i].first],-1);
        previous_ = pointsIndex_[i].first;
    }

    img = cache_;

    coordinate = index2point(pointsIndex_[i].second,cv::Size(cache_.cols,cache_.rows));

    return true;
}

bool DepthFileBasedImageDBImpl::getDataPoint(index_type i, std::string &file, cv::Point2i &coordinate)
{

    if (i >= pointsIndex_.size()){
        return false;
    }

    file = files_[pointsIndex_[i].first];

    if(constImgSize_){
        coordinate = index2point(pointsIndex_[i].second,imgSize_);
        return true;
    }

    if (pointsIndex_[i].first != previous_)
    {
        //the images is not in cache
        //load the image
        cache_ = cv::imread(file,-1);
        previous_ = pointsIndex_[i].first;
    }

    coordinate = index2point(pointsIndex_[i].second,cv::Size(cache_.cols,cache_.rows));

    return true;
}

void DepthFileBasedImageDBImpl::readFiles(const std::string &file, GeneralStringParser &parser)
{
    std::ifstream input(file.c_str());
    cv::Mat image;
    std::string tmp,filename;

    if (input.is_open()){

        while(!input.eof()){

            tmp = "";

            input >> tmp;

            if (!tmp.empty()){

                parser.setString(tmp);
                filename = parser.getFilename();

                if (path_.length()>0){
                    filename = path_+filename;
                }

                std::cout << "reading file: " << filename << std::endl;

                files_.push_back(filename);
                image = cv::imread(filename,-1);

                if (elementCount_ == 0){
                    if(constImgSize_)
                        imgSize_ = cv::Size(image.cols,image.rows);
                    cache_.push_back(image);
                    previous_ = elementCount_;
                }

                if(constImgSize_){
                    if((imgSize_.width != image.cols) || (imgSize_.height != image.rows)){
                        std::cerr << "image size is set to constant, while image size differs for the file: "
                              << filename << std::endl;
                    }
                }

                postprocessFile(image,parser); /*user-defined function*/

                elementCount_++;
            }
        }
    }
    input.close();
}

bool DepthFileBasedImageDBImpl::postprocessFile(const cv::Mat &mat,GeneralStringParser &parser){
    int rows = mat.rows,cols = mat.cols;
    const unsigned short *dataptr;
    cv::Size imgSize(mat.cols,mat.rows);

    if (mat.isContinuous()){
        cols *= rows;
        rows = 1;
    }

    for(int i=0; i<rows; i++){

        dataptr = mat.ptr<unsigned short>(i);

        for(int j=0; j<cols; j++){
            if (dataptr[j]>0){
                push_pixel(point2index(cv::Point2i(j,i),imgSize));
            }
        }
    }

    return true;
}

void DepthFileBasedImageDBImpl::push_pixel(unsigned short index){
    try{
        pointsIndex_.push_back(filebased_type(files_.size()-1 ,index));
    }catch(std::bad_alloc &e){
        std::cerr << "bad allocation pointsIndex_:" << pointsIndex_.vector_size() << std::endl;
    }
}

