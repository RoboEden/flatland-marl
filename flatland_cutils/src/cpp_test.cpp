// #include <bitset>
// #include <bits/stdc++.h>
// #include <iostream>
// #include <type_traits>
// #include <typeinfo>
// #include <stdexcept>
// #include <sstream>
// #include <exception>
// #include <string>
// #include <fstream>
// // #include "loader.h"
// #include <boost/array.hpp>
// #include <iostream>
// #include <cassert>
// #include <charconv>
// #include <iomanip>
// #include <iostream>
// #include <optional>
// #include <string_view>
// #define OUTPUT(a) std::cout << #a <<": "<<a<< '\n'

// using namespace std;
// #define getName(VariableName) # VariableName

// template<typename T>
// std::ostream& print(std::ostream &out, T const &val) { 
//   return (out << val << " ");
// }

// template<typename T1, typename T2>
// std::ostream& print(std::ostream &out, std::pair<T1, T2> const &val) { 
//   return (out << "{" << val.first << " " << val.second << "} ");
// }

// template<template<typename, typename...> class TT, typename... Args>
// std::ostream& operator<<(std::ostream &out, TT<Args...> const &cont) {
//   for(auto&& elem : cont) print(out, elem);
//   return out;
// }

// #include<ctime>

// #define OUTPUT(a) std::cout << #a << ": " << a << '\n'
// #define OUTPUT_VEC(vec)        \
//     std::cout << #vec << ": "; \
//     for (int v : vec) {        \
//         std::cout << v << " "; \
//     }                          \
//     std::cout << std::endl

// template <typename T>
// std::vector<T> operator+(std::vector<T> const& x, std::vector<T> const& y) {
//     std::vector<T> vec;
//     vec.reserve(x.size() + y.size());
//     vec.insert(vec.end(), x.begin(), x.end());
//     vec.insert(vec.end(), y.begin(), y.end());
//     return vec;
// }
// template <typename T>
// std::vector<T>& operator+=(std::vector<T>& x, const std::vector<T>& y) {
//     x.reserve(x.size() + y.size());
//     x.insert(x.end(), y.begin(), y.end());
//     return x;
// }

// int main()
// {   
//     std::vector<float> agent_attr;
//     float moving = 1;
//     agent_attr += {moving};
//     OUTPUT_VEC(agent_attr);
//     return 0;
// }