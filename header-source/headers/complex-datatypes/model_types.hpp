#pragma once
#include <map>
#include <string>
#include <variant>

using HashMapParameters = std::map<
    std::string, 
    std::variant<int, float, std::string, bool>
>;

using IntegerCategoricalList = std::variant<
    std::vector<int>,
    std::vector<float>
>;

using SIDual = std::variant<std::string, int>;
