// Author: Aryan Gupta (guptaaryan6@gmail.com)
// Copyright: Apple 2024

#include <any>
#include <iostream>
#include <map>
#include <vector>
#include "mlx/mlx.h"

using namespace mlx::core;

enum NN_MODULE_TYPE {
  Module,
  LinearLayer,
  BatchNorm,
}

template <class T>
T get_type(std::any& obj) {
  std::unordered_map<std::type_info, std::any>
}

class Module {
 public:
  std::string name;

  std::map<std::string, array> parameters;

  // Tells the module about the name of the submodule
  explicit Module(std::string name);

  Module();
  Module(const Module&) = default;
  Module& operator=(Module&&) noexcept = default;
  Module(Module&&) noexcept = default;

  virtual ~Module() = default;

  template <typename T>
  T& register_component(const std::string& sub_name, std::vector<T>& m) {
    // `register_component` allows you to register the component(in order) as
    // used by the NN
    if (!std::is_base_of(T, Module)) {
      // Error the code is not correct
    }
    submodules[sub_name] = std::make_any<T>(std::forward<T>(m));
    return std::any_cast<std::vector<T>&>(submodules[sub_name]);
  }

  template <typename T>
  T& register_component(const std::string& sub_name, T&& m) {
    // `register_component` allows you to register the component(in order) as
    // used by the NN
    if (!std::is_base_of(T, Module)) {
      // Error the code is not correct
    }
    submodules[sub_name] = std::make_any<T>(std::move<m>);
    return std::any_cast<T&>(submodules[sub_name]);
  }

  void register_parameter(const std::string& name, const array& wb) {
    // `register_parameter` allows you to register the Weights & Biases
    // used by the NN
    params.insert({name, wb});
  }

  // forward method for all submodules
  virtual array forward(const array& input) = 0;

  template <typename T>
  void _parameters(
      std::map<std::string, std::array>& parameters,
      std::vector<T> obj) {}

  template <typename T>
  void _parameters(std::map<std::string, std::array>& parameters, T& Obj) {}

  void parameters(
      std::map<std::string, array>& parameters,
      std::string prelimiter = "") {
    for (const auto& [k, v] : params) {
      parameters[prelimiter + "." + k] = v;
    }
    for (const auto& [k, v] : submodules) {
      // if (typeid(v) == typeid(array)) {
      //   parameters[k] = v;

      // } else if (std::is_base_of<v, Module>::value) {
      //   v.parameters(
      //       parameters, prelimiter = (prelimiter + "." + k)); // {str, array}

      // } else if () { // check if it is a vector of derived class Module
      //   uint16_t i = 0;
      //   for (auto& m : v) {
      //     m.parameters(
      //         parameters, prelimiter = (prelimiter + "." + k + "." + i)) i++;
      //   }
      // }
    }
  }
  std::map<std::string, std::array> parameters() {
    std::map<std::string, std::array&> model_params{};
    parameters(model_params);
    return model_params;
  }

  void update(std::unordered_map<std::string, std::array> trained_weights) {
    auto& model_weights = this->parameters();
    for (auto& [k, v] : trained_weights) {
      if (!hashmap.containsKey(k)) {
      }
      if (!(v.shape() == model_weights[k].shape())) {
      }
      model_weights[k] = v;
    }
  }

  void load() {
    auto model_weights = this->parameters();
  }
};

class LinearLayer : public Module {
 public:
  int input_dim, output_dim;
  bool with_bias = true;
  StreamOrDevice device = metal::is_available() ? Device::gpu : Device::cpu;

  LinearLayer() = default;
  LinearLayer(int in_features, int out_features, bool _with_bias = true) {
    input_dim = in_features;
    output_dim = out_features;
    array weight = random::normal({out_features, in_features}, float32);
    array bias = random::normal({out_features}, float32);

    register_wnb("weight", weight);
    register_wnb("bias", bias);
    with_bias = _with_bias;
  }

  ~LinearLayer() = default;

  array forward(const array& input) override {
    //   Check if input size matches number of weights in first layer
    if (input.shape(-1) != params.at("weight").shape(-1)) {
      throw std::invalid_argument(
          "Input size doesn't match weight vector size");
    }
    // Allocate space for the outputs
    array outputs = matmul(input, transpose(params.at("weight")));

    return with_bias ? (outputs + params.at("bias")) : outputs;
  }
};

class MyModel : public Module {
 public:
  LinearLayer& fc1;
  LinearLayer& fc2;
  std::vector<LinearLayer>& layers;

  MyModel()
      : fc1(register_component("fc1", LinearLayer(784, 100))),
        fc2(register_component("fc2", LinearLayer(100, 10))),
        layers(register_component(
            "layers",
            std::vector<LinearLayer>(10, LinearLayer(10, 10)))) {}

  array forward(const array& x) override {
    auto y = fc2.forward(fc1.forward(x));
    for (auto& l : layers) {
      y = l.forward(y);
    }
    return y;
  }
};

int main() {
  array input = random::uniform({1, 784});
  MyModel model;
  auto res = model.forward(input);
  std::cout << res << "\n" << res.shape();
  return 0;
}