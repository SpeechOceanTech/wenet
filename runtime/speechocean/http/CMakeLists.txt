# 编译成而可执行二进制文件
add_executable(model model.cc)

# 编译成动态库，便于go集成
# add_library(model STATIC model.cc)
target_link_libraries(model PUBLIC decoder)