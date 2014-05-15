#include "depthimagedb.h"

#include <fstream>

DepthFileBasedImageDBImpl::DepthFileBasedImageDBImpl(const std::string &base,
                                             bool constImgSize)
{
    path_ = base;
    elementCount_ = 0;
    constImgSize_ = constImgSize;
    cachCallCount_ = 0;
    cache_ = new InMemoryCache();
    sub_ = new StubSubsampler();
}

bool DepthFileBasedImageDBImpl::loadDB(const std::string &filename, GeneralStringParser &stringParser, bool hasHeader){
    try{
        readFiles(filename,stringParser, hasHeader);
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

    cache_->getImage(pointsIndex_[i].first,img);


    coordinate = index2point(pointsIndex_[i].second,img.size());

    return true;
}

bool DepthFileBasedImageDBImpl::getDataPoint(index_type i, std::string &file, cv::Point2i &coordinate)
{

    if (i >= pointsIndex_.size()){
        return false;
    }

    file = cache_->imageIdx2Filename(pointsIndex_[i].first);

    if(constImgSize_){
        coordinate = index2point(pointsIndex_[i].second,imgSize_);
        return true;
    }

    cv::Mat img;
    cache_->getImage(pointsIndex_[i].first,img);

    coordinate = index2point(pointsIndex_[i].second,img.size());

    return true;
}

void DepthFileBasedImageDBImpl::readFiles(const std::string &file,  GeneralStringParser &parser, bool hasHeader)
{
    std::ifstream input(file.c_str());
    cv::Mat image;
    std::string tmp,filename;

    std::cerr << "reading files from: " << file << std::endl;

    if (input.is_open()){

        if (hasHeader){
            tmp = "";

            input >> tmp;
            processHeader(tmp);

        }

        while(!input.eof()){

            tmp = "";

            input >> tmp;

            if (!tmp.empty()){

                parser.setString(tmp);
                filename = parser.getFilename();

                if (path_.length()>0){
                    filename = path_+filename;
                }

                cache_->addToCache(filename);
                image = cv::imread(filename,-1);

                if (elementCount_ == 0){
                    if(constImgSize_){
                        imgSize_ = image.size();
                    }
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

    for(int i=0; i<rows; i++){

        dataptr = mat.ptr<unsigned short>(i);

        for(int j=0; j< cols; j++){
            if (dataptr[j]>0){
                //cv::Point2i (x,y)
                if(sub_->add(cv::Point2i(j,i),dataptr[j])){
                    push_pixel(point2index(cv::Point2i(j,i),mat.size()));
                }
            }
        }
    }

    return true;
}

void DepthFileBasedImageDBImpl::processHeader(const std::string &header)
{
    // do nothing
}

void DepthFileBasedImageDBImpl::push_pixel(in_image_index index){

    try{
        pointsIndex_.push_back(filebased_type(cache_->imageCount()-1 ,index));
    }catch(std::bad_alloc &e){
        std::cerr << "bad allocation pointsIndex_:" << pointsIndex_.vector_size() << std::endl;
    }
}

