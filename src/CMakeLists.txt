set(EXECCL DepthRFCL)
set(EXECH DepthRFHOUGH)
set(FEATUREGEN FeatureGen)
set(TESTS tests)

set(CL_FOLDER classification)
set(H_FOLDER hough)

set(CL_HEADERS ${CL_FOLDER}/depthdb.h /
            ${CL_FOLDER}/classstats.h /
            ${CL_FOLDER}/cltrainingcontext.h /
            ${CL_FOLDER}/inmemdb.h /
            ${CL_FOLDER}/imagepixelstats.h
            ${CL_FOLDER}/testclassificationforest.h)

set(HOUGH_HEADERS ${H_FOLDER}/houghtrainingcontext.h /
                  ${H_FOLDER}/votesstats.h /
                  ${H_FOLDER}/depthdbwithvotes.h /
                  ${H_FOLDER}/houghvotesstats.h /
                  ${H_FOLDER}/pooledhoughtrainingcontext.h )

set(GENERAL_HEADERS /
            localcache.h /
            depthfeature.h /
            parameter.h /
            arraylist.h /
            depthimagedb.h/
            split.h /
            rfutils.h /
            serialization.h /
            subsampler.h /
            imagecache.h /
            stubstats.h /
            stubtrainingcontext.h /
            featurepool.h /
            featureaccomulator.h /
            nodedistributionimagestats.h /
            colorgenerator.h /
)

set(CL_SOURCE ${CL_FOLDER}/depthdb.cpp /
            ${CL_FOLDER}/classstats.cpp /
            ${CL_FOLDER}/inmemdb.cpp /
            ${CL_FOLDER}/cltrainingcontext.cpp /
            ${CL_FOLDER}/imagepixelstats.cpp
            ${CL_FOLDER}/testclassificationforest.cpp)

set(HOUGH_SOURCE ${H_FOLDER}/houghtrainingcontext.cpp /
                 ${H_FOLDER}/votesstats.cpp /
                 ${H_FOLDER}/depthdbwithvotes.cpp /
                 ${H_FOLDER}/houghvotesstats.cpp /
                 ${H_FOLDER}/pooledhoughtrainingcontext.cpp)

set(GENERAL_SOURCE  /
                    localcache.cpp /
                    depthfeature.cpp /
                    parameter.cpp /
                    arraylist.cpp/
                    depthimagedb.cpp /
                    split.cpp /
                    rfutils.cpp /
                    serialization.cpp /
                    subsampler.cpp /
                    imagecache.cpp /
                    stubstats.cpp /
                    stubtrainingcontext.cpp /
                    featurepool.cpp /
                    featureaccomulator.cpp /
                    nodedistributionimagestats.cpp /
                    colorgenerator.cpp /
)

#required to see the code in QT creator only...
set(FOREST ${Forest_DIR}/Forest.h /
           ${Forest_DIR}/ForestTrainer.h /
           ${Forest_DIR}/Interfaces.h /
           ${Forest_DIR}/Node.h /
           ${Forest_DIR}/ParallelForestTrainer.h /
           ${Forest_DIR}/ProgressStream.h /
           ${Forest_DIR}/Random.h /
           ${Forest_DIR}/Sherwood.h /
           ${Forest_DIR}/TrainingParameters.h /
           ${Forest_DIR}/Tree.h)



add_executable(${EXECCL} main_cl.cpp ${CL_SOURCE} ${CL_HEADERS} ${GENERAL_HEADERS} ${GENERAL_SOURCE} ${FOREST})
add_executable(${EXECH} main_h.cpp ${HOUGH_SOURCE} ${HOUGH_HEADERS} ${GENERAL_HEADERS} ${GENERAL_SOURCE} ${FOREST})
add_executable(${TESTS} tests.cpp ${CL_SOURCE} ${CL_HEADERS} ${HOUGH_SOURCE} ${HOUGH_HEADERS} ${GENERAL_HEADERS} ${GENERAL_SOURCE} ${FOREST})
add_executable(${FEATUREGEN} featureGenerator.cpp ${GENERAL_SOURCE})



target_link_libraries(${EXECCL} ${OpenCV_LIBS} ${Boost_LIBS})
target_link_libraries(${EXECH} ${OpenCV_LIBS} ${Boost_LIBS})
target_link_libraries(${TESTS} ${OpenCV_LIBS} ${Boost_LIBS})
target_link_libraries(${FEATUREGEN} ${OpenCV_LIBS} ${Boost_LIBS})

#install(TARGETS ${EXECCL} DESTINATION .)
install(TARGETS ${EXECH} DESTINATION .)
install(TARGETS ${TESTS} DESTINATION .)
install(TARGETS ${FEATUREGEN} DESTINATION .)
