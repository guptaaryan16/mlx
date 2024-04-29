#include <any>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "mlx/mlx.h"

namespace mlx::core {
namespace nn {

// Helper function to flatten a nested structure
void flatten(
    const std::string& prefix,
    const std::any& value,
    std::map<std::string, std::vector<std::any>>& flattened) {
  if (value.type() == typeid(std::map<std::string, std::any>)) {
    const auto& nested =
        std::any_cast<const std::map<std::string, std::any>&>(value);
    for (const auto& [key, val] : nested) {
      std::string path = prefix.empty() ? key : prefix + "." + key;
      flatten(path, val, flattened);
    }
  } else {
    flattened[prefix].push_back(value);
  }
}

// tree.flatten method
std::map<std::string, std::vector<std::any>> tree_flatten(
    const std::map<std::string, std::any>& tree) {
  std::map<std::string, std::vector<std::any>> flattened;
  for (const auto& [key, val] : tree) {
    flatten(key, val, flattened);
  }
  return flattened;
}

// Helper function to unflatten a flattened structure
std::any unflatten(
    const std::map<std::string, std::vector<std::any>>& flattened,
    const std::string& prefix) {
  std::map<std::string, std::any> nested;
  for (const auto& [key, values] : flattened) {
    if (key.find(prefix) == 0) {
      size_t pos = prefix.length();
      std::string subkey = (pos < key.length()) ? key.substr(pos + 1) : "";
      if (subkey.empty()) {
        nested[key.substr(prefix.length())] =
            values.size() == 1 ? values[0] : std::vector<std::any>(values);
      } else {
        pos = subkey.find('.');
        std::string next_prefix = prefix +
            (pos == std::string::npos ? "." + subkey
                                      : "." + subkey.substr(0, pos));
        nested[subkey.substr(0, pos)] = unflatten(flattened, next_prefix);
      }
    }
  }
  return nested;
}

void print_tree(const std::map<std::string, std::any>& tree, int indent = 0) {
  for (const auto& [key, val] : tree) {
    std::cout << std::string(indent * 2, ' ') << key << ": ";
    if (val.type() == typeid(std::map<std::string, std::any>)) {
      std::cout << std::endl;
      print_tree(
          std::any_cast<const std::map<std::string, std::any>&>(val),
          indent + 1);
    } else {
      std::cout << std::any_cast<std::string>(val) << std::endl;
    }
  }
}

// Helper function to print a flattened structure
void print_flattened(
    const std::map<std::string, std::vector<std::any>>& flattened) {
  for (const auto& [key, values] : flattened) {
    std::cout << key << ": [";
    bool first = true;
    for (const auto& val : values) {
      if (!first)
        std::cout << ", ";
      first = false;
      std::cout << std::any_cast<std::string>(val);
    }
    std::cout << "]" << std::endl;
  }
}

// tree.unflatten method
std::map<std::string, std::any> tree_unflatten(
    const std::map<std::string, std::vector<std::any>>& flattened) {
  std::map<std::string, std::any> result;
  for (const auto& [key, values] : flattened) {
    if (key.find('.') == std::string::npos) {
      result[key] =
          values.size() == 1 ? values[0] : std::vector<std::any>(values);
    } else {
      result = std::any_cast<std::map<std::string, std::any>>(
          unflatten(flattened, ""));
      break;
    }
  }
  return result;
}

int main() {
  // Example nested structure
  std::map<std::string, std::any> tree = {
      {"a", 1},
      {"b",
       std::map<std::string, std::any>{
           {"c", 2},
           {"d", std::map<std::string, std::any>{{"e", 3}, {"f", 4}}}}}};

  // Flatten the nested structure
  std::map<std::string, std::vector<std::any>> flattened = tree_flatten(tree);
  // Flattened structure:
  // {
  //     "a": [1],
  //     "b.c": [2],
  //     "b.d.e": [3],
  //     "b.d.f": [4]
  // }
  // std::cout << flattened << std::endl;

  // Unflatten the flattened structure
  std::map<std::string, std::any> unflattened = tree_unflatten(flattened);
  // Unflattened structure is the same as the original nested structure
  // std::cout << unflattened <<std::endl;
  return 0;
}

} // namespace nn
} // namespace mlx::core
