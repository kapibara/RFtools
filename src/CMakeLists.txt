set(EXEC DepthRF)

set(HEADERS depthdb.h /
            depthfeature.h /
            classstats.h /
            trainingcontext.h /
            parameter.h /
            arraylist.h /
            inmemdb.h /
            imagepixelstats.h)

set(SOURCES main.cpp /
            split.hpp)

set(S2 test.cpp)

set(MODULES depthdb.cpp /
            depthfeature.cpp /
            classstats.cpp /
            trainingcontext.cpp /
            parameter.cpp /
            arraylist.cpp /
            inmemdb.cpp /
            imagepixelstats.cpp)

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



add_executable(${EXEC} ${SOURCES} ${MODULES} ${HEADERS} ${FOREST})
#add_executable(test ${S2})

target_link_libraries(${EXEC} ${OpenCV_LIBS})

install(TARGETS ${EXEC} DESTINATION .)
#install(TARGETS test DESTINATION .)
