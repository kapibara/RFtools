#ifndef STUBVOTESSTATS_H
#define STUBVOTESSTATS_H

#include "Interfaces.h"

#include <ostream>

class StubStats
{
public:
    StubStats()
    {}

    void Clear(){}

    void Aggregate(MicrosoftResearch::Cambridge::Sherwood::IDataPointCollection& data, unsigned int index)
    {}

    void Aggregate(const StubStats& i)
    {}

    void FullStats(bool compute)
    {}

    StubStats DeepClone() const
    {
        return StubStats();

    }

    void Compress()
    {}

    bool Serialize(std::ostream &stream) const
    {
        return true;

    }

    bool Deserialize(std::istream &stream)
    {

        return true;
    }

};

#endif // STUBVOTESSTATS_H
