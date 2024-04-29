#include <any>
#include <iostream>
#include <map>
#include <string>
#include "mlx/mlx.h"

namespace mlx::core {

namespace nn {
class Module {
 public:
  // TODO: Enable Training Mode
  //   bool _training = false;
  //   bool _no_grad = false;
  std::string module_name;
  std::map<std::string, array>
      parameters_; // all module props like weights and biases
  std::map<std::string, Module> children_;
  std::vector<Module>
      layers; // layers will be numbered as layer.0, layer.1 etc.
  bool has_child = false; // for checking if Module is a leaf node

  Module() = default;
  ~Module(){};

  /// register_module is called to register a submodule
  /// this->register_module("linear1", nn::Linear(<args>));
  void register_module(const std::string& name, Module& m) {
    if (!has_child) {
      has_child = true;
    }
    m->name = name;
    children_[name] = m;
  }

  void register_module(const std::string& name, Module&& m) {
    if (!has_child) {
      has_child = true;
    }
    m->name = name;
    children_[name] = m;
  }

  void register_layer(const std::string& module_name, Module& m) {
    this->layers.push_back(m);
  }

  std::map<std::string, any> parameters() {
    std::map<std::string, any>;
    for (auto m : children_) {
      params[m.module_name] = weights;
      if (m.has_child) {
        m.parameters(params);
      }
    };

    // static valid_parameter()

    void load_weights(SafetensorsLoad & dict) {
      std::unordered_map<std::string, array> weights = dict.first;
      std::unordered_map<std::string, std::string> metadata = dict.second;
      this->update(weights);
    }

    void load_weights(std::string & filename) {
      SafetensorsLoad dict = load_safetensors(filename, s);
      this->load_weights(dict);
    }

    void update(std::map<std::string, array> & parameters) {
      void apply(Module & m, std::map<std::string, array> parameters){
          // for(auto k: parameters){
          //     if(k.)
          // }
      };

      void apply(Module & m, std::map) {}
      apply(*this, this->parameters_);
      apply(*this, this->children_);
      apply(*this, this->layers);
    }

    bool valid_parameter_filter(Module & m, std::string key, array value) {}

    std::map<std::string, Module> submodules() {
      return this->children_;
    }
  };

} // namespace nn;

} // namespace mlx::core;

// using namespace mlx::core;

// int main(){

// }