
function(pto_vec_st NAME)
    add_library(${NAME}_kernel SHARED ${NAME}_kernel.cpp)
    target_compile_options(${NAME}_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220-vec -D__DAV_V220 -std=c++17)
    target_include_directories(${NAME}_kernel PRIVATE
        ${PROJECT_SOURCE_DIR}/../../../../../include
        ${PROJECT_SOURCE_DIR}/../../../../../include/common
    )
    target_link_options(${NAME}_kernel PRIVATE --cce-fatobj-link)

    add_executable(${NAME} main.cpp)
    target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/../../../../../include/common
        ${PROJECT_SOURCE_DIR}/../../../common
    )

    target_link_directories(${NAME} PUBLIC
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
        ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
    )

    target_link_libraries(${NAME} PRIVATE
        ${NAME}_kernel
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
        stdc++ ascendcl m tiling_api platform c_sec dl nnopbase GTest::gtest GTest::gtest_main pthread
    )
endfunction()

function(pto_cube_st NAME)
    add_library(${NAME}_kernel SHARED ${NAME}_kernel.cpp)
    target_compile_options(${NAME}_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220-cube -D__DAV_V220 -std=c++17)
    target_include_directories(${NAME}_kernel PRIVATE
        ${PROJECT_SOURCE_DIR}/../../../../../include/
        ${PROJECT_SOURCE_DIR}/../../../../../include/common
    )
    target_link_options(${NAME}_kernel PRIVATE --cce-fatobj-link)

    add_executable(${NAME} main.cpp)
    target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/../../../../../include/common
        ${PROJECT_SOURCE_DIR}/../../../common
    )

    target_link_directories(${NAME} PUBLIC
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
        ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
    )

    target_link_libraries(${NAME} PRIVATE
        ${NAME}_kernel
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
        stdc++ ascendcl m tiling_api platform c_sec dl nnopbase GTest::gtest GTest::gtest_main pthread
    )
endfunction()

# 定义测试用例列表 testcase下如有新增用例目录需要在此加
set(ALL_TESTCASES
tmatmul
trowsum
textract
tmov
tload
ttrans
tmrgsort
tgather
tsort32
tstore
tcopy
trowexpand
tcvt
tadd
tsub
tload_gm2mat
)

if(TEST_CASE IN_LIST ALL_TESTCASES)
message(STATUS "run : ${TEST_CASE}")
else()
message(FATAL_ERROR "not found TEST_CASE: ${TEST_CASE}, support case is : ${ALL_TESTCASES}")
endif()


foreach(TESTCASE ${ALL_TESTCASES})
if((DEFINED TEST_CASE AND TEST_CASE STREQUAL TESTCASE) OR (NOT DEFINED TEST_CASE))
    add_subdirectory(${TESTCASE})
endif()
endforeach()
