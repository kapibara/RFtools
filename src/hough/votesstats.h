#ifndef VOTESSTATS_H
#define VOTESSTATS_H

class VotesStats
{
public:
    void Clear();

    void Aggregate(const IDataPointCollection& data, unsigned int index);

    void Aggregate(const VotesStats& i);

    virtual VotesStats DeepClone() const;
};

#endif // VOTESSTATS_H
