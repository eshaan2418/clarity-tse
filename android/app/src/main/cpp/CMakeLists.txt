cmake_minimum_required(VERSION 3.22.1)
project(clarity_jni CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(oboe REQUIRED CONFIG)
add_library(onnxruntime SHARED IMPORTED)  # unpacked from the AAR by the extractOrt gradle task
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION ${ORT_DIR}/jni/${ANDROID_ABI}/libonnxruntime.so
  INTERFACE_INCLUDE_DIRECTORIES ${ORT_DIR}/headers)
set(RT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../rt)
add_library(clarity_jni SHARED clarity_jni.cpp ${RT}/src/engine.cpp)
target_include_directories(clarity_jni PRIVATE ${RT}/include)
target_link_libraries(clarity_jni oboe::oboe onnxruntime log android)
