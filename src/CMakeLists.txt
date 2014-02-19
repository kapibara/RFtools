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
                  ${H_FOLDER}/depthdbwithvotes.h)

set(GENERAL_HEADERS /
            depthfeature.h /
            parameter.h /
            arraylist.h /
            depthimagedb.h/
            split.h)

set(CL_SOURCE ${CL_FOLDER}/depthdb.cpp /
            ${CL_FOLDER}/classstats.cpp /
            ${CL_FOLDER}/inmemdb.cpp /
            ${CL_FOLDER}/cltrainingcontext.cpp /
            ${CL_FOLDER}/imagepixelstats.cpp)

set(HOUGH_SOURCE ${H_FOLDER}/houghtrainingcontext.cpp /
                 ${H_FOLDER}/votesstats.cpp /
                 ${H_FOLDER}/depthdbwithvotes.cpp)

set(GENERAL_SOURCE  /
                    depthfeature.cpp /
                    parameter.cpp /
                    arraylist.cpp/
                    depthimagedb.cpp /
                    split.cpp)

set(FOREST ../../Sherwood/cpp/lib/Forest.h /
           ../../Sherwood/cpp/lib/ForestTrainer.h /
           ../../Sherwood/cpp/lib/Interfaces.h /
           ../../Sherwood/cpp/lib/Node.h /
           ../../Sherwood/cpp/lib/ParallelForestTrainer.h /
           ../../Sherwood/cpp/lib/ProgressStream.h /
           ../../Sherwood/cpp/lib/Random.h /
           ../../Sherwood/cpp/lib/Sherwood.h /
           ../../Sherwood/cpp/lib/TrainingParameters.h /
           ../../Sherwood/cpp/lib/Tree.h)



add_executable(${EXECCL} main_cl.cpp ${CL_SOURCE} ${CL_HEADERS} ${GENERAL_HEADERS} ${GENERAL_SOURCE} ${FOREST})
#add_executable(${EXECCL} main_cl.cpp ${GENERAL_HEADERS} ${GENERAL_SOURCE})
add_executable(${EXECH} main_h.cpp ${HOUGH_SOURCE} ${HOUGH_HEADERS} ${GENERAL_HEADERS} ${GENERAL_SOURCE} ${FOREST})
#add_executable(test ${S2})
#add_executable(${EXECH} main_h.cpp)

target_link_libraries(${EXECCL} ${OpenCV_LIBS})
target_link_libraries(${EXECH} ${OpenCV_LIBS})

install(TARGETS ${EXECCL} DESTINATION .)
install(TARGETS ${EXECH} DESTINATION .)
#install(TARGETS test DESTINATION .)
