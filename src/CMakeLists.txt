set(EXECCL DepthRFCL)
set(EXECH DepthRFHOUGH)

set(CL_FOLDER classification)
set(H_FOLDER hough)

set(CL_HEADERS ${CL_FOLDER}/depthdb.h /
            ${CL_FOLDER}/classstats.h /
            ${CL_FOLDER}/cltrainingcontext.h /
            ${CL_FOLDER}/inmemdb.h /
            ${CL_FOLDER}/imagepixelstats.h)

set(HOUGH_HEADERS ${H_FOLDER}/houghtrainingcontext.h /
                  ${H_FOLDER}/votesstats.h /
                  ${H_FOLDER}/depthdbwithvotes.h /
                  ${H_FOLDER}/houghvotesstats.h  )

set(GENERAL_HEADERS /
            depthfeature.h /
            parameter.h /
            arraylist.h /
            depthimagedb.h/
            split.h /
            rfutils.h)

set(CL_SOURCE ${CL_FOLDER}/depthdb.cpp /
            ${CL_FOLDER}/classstats.cpp /
            ${CL_FOLDER}/inmemdb.cpp /
            ${CL_FOLDER}/cltrainingcontext.cpp /
            ${CL_FOLDER}/imagepixelstats.cpp)

set(HOUGH_SOURCE ${H_FOLDER}/houghtrainingcontext.cpp /
                 ${H_FOLDER}/votesstats.cpp /
                 ${H_FOLDER}/depthdbwithvotes.cpp /
                 ${H_FOLDER}/houghvotesstats.cpp)

set(GENERAL_SOURCE  /
                    depthfeature.cpp /
                    parameter.cpp /
                    arraylist.cpp/
                    depthimagedb.cpp /
                    split.cpp /
                    rfutils.cpp)

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


target_link_libraries(${EXECCL} ${OpenCV_LIBS})
target_link_libraries(${EXECH} ${OpenCV_LIBS})

install(TARGETS ${EXECCL} DESTINATION .)
install(TARGETS ${EXECH} DESTINATION .)
