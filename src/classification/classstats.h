#ifndef CLASSSTATS_H
#define CLASSSTATS_H

#include "Interfaces.h"
#include <ostream>

#define ENABLE_OVERFLOW_CHECKS
#define ENABLE_BOUNDARY_CHECKS

class ClassStats
{
    typedef unsigned long bintype;
public:
    ClassStats(unsigned short clCount = 0);
    ClassStats(const ClassStats &obj); /*this deep copying might be not needed according to the framework*/
    ~ClassStats();

    void Clear();

    void Aggregate(const MicrosoftResearch::Cambridge::Sherwood::IDataPointCollection& data, unsigned int index);

    void Aggregate(const ClassStats& i);

    void Aggregate(bintype i);

    double Entropy() const;

    unsigned char ClassDecision() const;

    void Compress(){
        //do nothing
    }

    ClassStats & operator=(const ClassStats & obj); /*this deep copying might be not needed according to the framework*/

    bintype SampleCount() const {
        return sampleCount_;
    }

    unsigned char ClassCount() const{
        return binCount_;
    }

    virtual ClassStats DeepClone() const;

    bool Serialize(std::ostream &stream) const;
    bool Deserialize(std::istream &stream);

    bool SerializeChar(std::ostream &stream) const;

private:

    bintype *bins_;
    unsigned char binCount_;
    bintype sampleCount_;
};

#endif // CLASSSTATS_H
