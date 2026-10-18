cmake_minimum_required(VERSION 3.16)
project(astro_align LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Put executables in <repo>/output (works for single- and multi-config generators).
set(_astro_out ${CMAKE_CURRENT_SOURCE_DIR}/../output)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${_astro_out})
foreach(_cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_cfg} ${_astro_out})
endforeach()

option(ASTRO_USE_SYSTEM_LIBRAW "Link a system-installed LibRaw instead of the vendored copy" OFF)

if(ASTRO_USE_SYSTEM_LIBRAW)
    find_path(LIBRAW_INCLUDE_DIR libraw/libraw.h REQUIRED)
    find_library(LIBRAW_LIBRARY NAMES raw_r raw REQUIRED)
    set(ASTRO_LIBRAW_INCLUDE ${LIBRAW_INCLUDE_DIR})
    set(ASTRO_LIBRAW_LIB ${LIBRAW_LIBRARY})
else()
    set(LIBRAW_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/LibRaw CACHE PATH "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(ENABLE_OPENMP  ON  CACHE BOOL "" FORCE)
    set(ENABLE_LCMS    OFF CACHE BOOL "" FORCE)
    set(ENABLE_JASPER  OFF CACHE BOOL "" FORCE)
    set(ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(ENABLE_RAWSPEED OFF CACHE BOOL "" FORCE)
    set(ENABLE_X3FTOOLS ON CACHE BOOL "" FORCE)
    set(ENABLE_6BY9RPI  ON CACHE BOOL "" FORCE)
    set(LIBRAW_UNINSTALL_TARGET OFF CACHE BOOL "" FORCE)
    set(LIBRAW_INSTALL OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/LibRaw-cmake ${CMAKE_BINARY_DIR}/libraw EXCLUDE_FROM_ALL)
    set(ASTRO_LIBRAW_INCLUDE ${LIBRAW_PATH})
    set(ASTRO_LIBRAW_LIB raw_r)
endif()

add_executable(astro_align
    src/main.cpp
    src/util.cpp
    src/register.cpp
    src/rawdecode.cpp
    src/stack.cpp
)
target_include_directories(astro_align PRIVATE src ${ASTRO_LIBRAW_INCLUDE})
target_link_libraries(astro_align PRIVATE ${ASTRO_LIBRAW_LIB})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(astro_align PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(astro_align PRIVATE /O2 /fp:fast /openmp)
    target_compile_definitions(astro_align PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(astro_align PRIVATE -O3 -ffast-math)
endif()
