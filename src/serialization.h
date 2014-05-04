#ifndef SERIALIZATION_H
#define SERIALIZATION_H

#include <vector>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>

template<class T>
void serializeVector(std::ostream &stream, const std::vector<T> &in, int from = -1, int to = -1)
{
    if (from < 0){
        from = 0;
        to = in.size();
    }

    unsigned int size = in.size();
    stream.write((const char *)(&size),sizeof(unsigned int));
    for(typename std::vector<T>::const_iterator i = in.begin()+from; i!= in.begin()+to; i++){
        stream.write((const char *)(&(*i)),sizeof(T));
    }
}

template<class T>
void serializeVector(const std::string &filename, const std::vector<T> &in, int from = -1, int to =-1)
{
    std::ofstream out(filename.c_str(),std::ios_base::binary);
    serializeVector<T>(out,in, from , to);
    out.close();
}

template<class ElemType, int S>
void serializeVoteVector(const std::vector<cv::Vec<ElemType,S> > &vec, std::ostream &out)
{
    int size = vec.size();
    out.write((const char *)&size,sizeof(size));
    size = S;
    out.write((const char *)&size,sizeof(size));
    for(int i=0; i<vec.size(); i++){
        for(int j=0; j<S; j++){
            out.write((const char *)&(vec[i][j]),sizeof(ElemType));
        }
    }
}



#endif // SERIALIZATION_H
