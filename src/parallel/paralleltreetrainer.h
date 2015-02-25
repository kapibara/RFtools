#ifndef PARALLELTREETRAINER_H
#define PARALLELTREETRAINER_H

#include <omp.h>

#include "Forest.h"
#include "ForestTrainer.h"
#include "Tree.h"


namespace MSRF =MicrosoftResearch::Cambridge::Sherwood;

/* DO NOT USE FOR FEATURE AGGREGATION ETC*/
template<class F, class S>
class TreeBasedParallelTrainer
{
public:
  /// <summary>
  /// Train a new decision forest given some training data and a training
  /// problem described by an instance of the ITrainingContext interface.
  /// </summary>
  /// <param name="random">Random number generator.</param>
  /// <param name="parameters">Training parameters.</param>
  /// <param name="context">An ITrainingContext instance describing
  /// the training problem, e.g. classification, density estimation, etc. </param>
  /// <param name="data">The training data.</param>
  /// <returns>A new decision forest.</returns>
  static std::auto_ptr<MSRF::Forest<F,S> > TrainForest(
    MSRF::Random& random,
    const MSRF::TrainingParameters& parameters,
    MSRF::ITrainingContext<F,S>& context,
    MSRF::IDataPointCollection& data,
    MSRF::ProgressStream* progress=0)
  {
    MSRF::ProgressStream defaultProgress(std::cout, parameters.Verbose? MSRF::Verbose : MSRF::Interest);
    if(progress==0)
      progress=&defaultProgress;

    std::vector<MSRF::Tree<F, S> *> trainedTrees;
    trainedTrees.resize(parameters.NumberOfTrees,0);

    #pragma omp parallel for num_threads(parameters.NumberOfTrees)
    for (int t = 0; t < parameters.NumberOfTrees; t++)
    {

        (*progress)[MSRF::Interest] << "\rTraining tree "<< t << "...";

        trainedTrees[t] = (MSRF::TreeTrainer<F, S>::TrainTree(random, context, parameters, data, progress)).release();

    }

    (*progress)[MSRF::Interest] << "\rTrained " << parameters.NumberOfTrees << " trees.         " << std::endl;

    std::auto_ptr<MSRF::Forest<F,S> > forest = std::auto_ptr<MSRF::Forest<F,S> >(new MSRF::Forest<F,S>());

    for(int t=0; t< parameters.NumberOfTrees; t++){
        forest->AddTree(std::auto_ptr<MSRF::Tree<F, S> >(trainedTrees[t]));
    }

    return forest;
  }
};

#endif // PARALLELTREETRAINER_H
