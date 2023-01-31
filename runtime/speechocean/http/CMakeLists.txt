# add_executable(model model.cc)
add_library(model STATIC model.cc)
target_link_libraries(model PUBLIC decoder)