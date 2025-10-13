#ifndef COMPLEX_DATATYPES_HPP
#define COMPLEX_DATATYPES_HPP

#include <armadillo>
#include <variant>
#include <map>

using HashMapParameters = std::map<std::string, std::variant<int, float, std::string, bool>>;

#endif
