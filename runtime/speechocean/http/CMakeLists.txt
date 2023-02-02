# 编译成而可执行二进制文件
if (WENET_BIN)
  add_executable(wenet-runtime model.cc)
  target_link_libraries(wenet-runtime PUBLIC decoder)
elseif (WENET_LIB)
  # 编译成动态库，便于go集成
  add_library(wenet SHARED model.cc)
  target_link_libraries(wenet PUBLIC decoder)
endif()