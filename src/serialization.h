#ifndef SERIALIZATION_H
#define SERIALIZATION_H

#include <vector>
#include <fstream>
#include <string>

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



#endif // SERIALIZATION_H
