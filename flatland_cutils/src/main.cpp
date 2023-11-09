// #include <pybind11/pybind11.h>
#include "treeobs.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(flatland_cutils, m) {
    // py::class_<AgentsLoader>(m, "AgentsLoader")
    //     // .def(py::init<>())
    //     .def("update", &AgentsLoader::update)
    //     // .def("set_env", &AgentsLoader::set_env)
    //     .def("reset", &AgentsLoader::reset)
    //     .def("clear", &AgentsLoader::clear);
         
    py::class_<TreeObsForRailEnv>(m, "TreeObsForRailEnv")
        .def(py::init<const int, const int>())
        .def("set_env", &TreeObsForRailEnv::set_env)
        .def("reset", &TreeObsForRailEnv::reset)
        .def("get_many", &TreeObsForRailEnv::get_many)
        .def("get_properties", &TreeObsForRailEnv::get_properties);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}