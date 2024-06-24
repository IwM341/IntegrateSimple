cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ version selection")


message("CMAKE_BUILD_TYPE = " ${CMAKE_BUILD_TYPE})
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message("Debug flags = " ${CMAKE_CXX_FLAGS})
endif()
if(CMAKE_BUILD_TYPE MATCHES Release)
    message("Release flags = " ${CMAKE_CXX_FLAGS})
    message("Release flags = " ${CMAKE_CXX_FLAGS_RELEASE})
endif()
if(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    message("RelWithDebInfo flags = " ${CMAKE_CXX_FLAGS})
endif()


project(simple_intagration LANGUAGES CXX VERSION 0.0.1)



set(P_SOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(SOURCES 
    ${P_SOURCES_DIR}/main.cpp
)

include(cmake/CPM.cmake)
find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
    message("pybind11 not found, download from git")
    CPMAddPackage("gh:pybind/pybind11@2.11.1")
endif()
find_package(Grob)
if(NOT Grob_FOUND)
    message("Grob not found, download from git")
    CPMAddPackage("gh:IwM341/GridObject@1.0.2")
endif()


add_library(Integrate MODULE ${SOURCES})
set_property(TARGET Integrate PROPERTY CXX_STANDARD 20)

target_link_libraries(Integrate pybind11::module)
target_link_libraries(Integrate Grob::grob)

pybind11_extension(Integrate)



##copy python module to output file
set(PYMODULE_DIR ${CMAKE_BINARY_DIR}/pymodule)
set(PYMODULE_SRC_DIR ${CMAKE_SOURCE_DIR})
file(COPY ${PYMODULE_SRC_DIR}/Integrate.pyi DESTINATION ${PYMODULE_DIR})


set_target_properties(Integrate PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PYMODULE_DIR}
)