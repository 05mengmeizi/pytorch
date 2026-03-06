cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(profiler_tests CXX)

set(PROFILER_TEST_ROOT ${TORCH_ROOT}/test/cpp/profiler)

find_package(Torch REQUIRED)
find_package(GTest REQUIRED)

# PrivateUse1 profiler tests require Kineto
if(USE_KINETO)
  add_executable(test_privateuse1_profiler
    ${TORCH_ROOT}/test/cpp/common/main.cpp
    ${PROFILER_TEST_ROOT}/test_privateuse1_profiler.cpp
  )

  target_link_libraries(test_privateuse1_profiler PRIVATE torch GTest::gtest)
  target_include_directories(test_privateuse1_profiler PRIVATE ${ATen_CPU_INCLUDE})
  target_compile_definitions(test_privateuse1_profiler PRIVATE USE_KINETO)
  target_include_directories(test_privateuse1_profiler PRIVATE
    ${TORCH_ROOT}/third_party/kineto/libkineto/include
  )
endif()
