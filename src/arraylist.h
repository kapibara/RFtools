#ifndef ARRAYLIST_H
#define ARRAYLIST_H

#include <vector>
#include <array>

#include <iostream>

template<class T>
class ArrayList
{
    typedef size_t size_type;
public:
    ArrayList(size_type elements_per_block = 5000){
        epb_ = elements_per_block;
        size_ = 0;
    }

    ArrayList(const ArrayList &a): arrays_(a.arrays_),epb_(a.epb_),size_(a.size_){
        for(size_type i=0; i<arrays_.size(); i++){
            arrays_[i] =  new T[epb_];
            memcpy(arrays_[i],a.arrays_[i],sizeof(T)*epb_);
        }
    }

    ~ArrayList(){
        for(size_type i=0; i<arrays_.size(); i++){
            delete [] arrays_[i];
        }
    }

    size_type vector_size() const {
        return arrays_.size();
    }

    size_type size() const{
        return size_;
    }

    T& operator[](size_type n){
        if (n >= size_ | n < 0){
            throw std::exception("index out of bounds");
        }

        size_type i1 = n / epb_;
        size_type i2 = n % epb_;

        return arrays_[i1][i2];
    }

    const T& operator[](size_type n) const{
        if (n >= size_ | n < 0){
            throw std::exception("index out of bounds");
        }

        size_type i1 = n / epb_;
        size_type i2 = n % epb_;

        return const_cast<const T&>(arrays_[i1][i2]);
    }

    void push_back(const T &val){
        T *memory;

        size_type i1 = size_ / epb_;
        size_type i2 = size_ % epb_;

        if (i1 == arrays_.size()){
            memory = new T[epb_];

            arrays_.push_back(memory);
        }

        if (i1 > arrays_.size()){
            std::cerr << "unexpected i1 value" << std::endl;
            exit(-4);
        }

        arrays_[i1][i2] = val;

        size_++;
    }


private:
    size_type epb_;
    size_type size_;

    std::vector< T *> arrays_;
};

#endif // ARRAYLIST_H
