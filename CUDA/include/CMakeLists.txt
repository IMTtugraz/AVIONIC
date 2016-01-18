SET(CONFIG_DIR "${CMAKE_SOURCE_DIR}/config/")
SET(GEN_WARNING "/* WARNING: Automatically generated file. Do not modify! */")
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/config_dir.h.cmake ${CMAKE_CURRENT_SOURCE_DIR}/config_dir.h)
