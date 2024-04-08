#include <any>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>
#include "mlx/mlx.h"

using namespace mlx::core;

// get_name function for data resolution
template <typename T>
std::string get_name(std::string& prelimiter, const T& value) {
  std::ostringstream oss;
  if (prelimiter.empty()) {
    oss << value;
  } else {
    oss << prelimiter << "." << value;
  }
  return oss.str();
}

template <typename T, typename... Args>
std::string
get_name(std::string& prelimiter, const T& value, const Args&... args) {
  return get_name(
      prelimiter,
      std::vector<std::string>{
          get_name(prelimiter, value), get_name(prelimiter, args...)});
}

template <typename T>
std::string get_name(std::string& prelimiter, const std::vector<T>& values) {
  std::ostringstream oss;
  bool is_first = true;
  for (const auto& value : values) {
    if (is_first) {
      is_first = false;
      oss << get_name(prelimiter, value);
    } else {
      oss << "." << value;
    }
  }
  return oss.str();
}

class Module {
 public:
  std::string name = "";
  std::map<std::string, array> parameters{};
  std::map<std::string, array> buffers{};
  std::map<std::string, Module&> submodules{};

  StreamOrDevice device = metal::is_available() ? Device::gpu : Device::cpu;

  // Tells the module about the name of the submodule
  // explicit Module(std::string name);

  Module(){};
  Module(const Module&) = default;
  // Module& operator=(Module&&) noexcept = default;
  //  Module(Module&&) noexcept = default;

  virtual ~Module() = default;

  array& register_parameter(std::string_view name, array& wb) {
    // `register_parameter` allows you to register the Weights & Biases
    // used by the NN
    std::string temp = std::string(name);
    parameters.insert({temp, wb});
    return parameters.at(temp);
  }

  array& register_parameter(std::string_view name, array&& wb) {
    // `register_parameter` allows you to register the Weights & Biases
    // used by the NN
    std::string temp = std::string(name);
    parameters.insert({temp, wb});
    return parameters.at(temp);
  }

  array& register_buffer(std::string_view name, array&& wb) {
    // `register_parameter` allows you to register the Weights & Biases
    // used by the NN
    std::string temp = std::string(name);
    buffers.insert({temp, wb});
    return buffers.at(temp);
  }

  template <typename T>
  Module register_module(const std::string_view sub_name, T&& m) {
    // `register_component` allows you to register the component(in order) as
    // used by the NN
    if (!std::is_base_of<T, Module>::value) {
      // Error the code is not correct
    }
    std::string temp = std::string(name);
    submodules.insert({temp, m});
    return submodules.at(temp);
  }

  template <typename T>
  std::vector<Module> register_layer(
      std::string_view sub_name,
      std::vector<T>& layers) {
    // `register_component` allows you to register the layers(in order) as
    // used by the NN
    if (!std::is_base_of<T, Module>::value) {
      // Error the code is not correct
    }
    std::vector<Module> layers_ptr;
    int i = 0;
    std::string temp = std::string(name);
    for (auto l : layers) {
      std::string sub_name_l = get_name(temp, i);
      this->submodules[sub_name_l] = l;
      layers_ptr.push_back(submodules[sub_name_l]);
      i++;
    }
    return layers_ptr;
  }

  // Forward method for all submodules
  // TODO:: Make A general method for all forward implementations
  virtual array forward(const array& input) {
    return input;
  };

  void named_parameters(
      std::map<std::string, array>& params,
      std::string prelimiter = "") {
    for (const auto& [k, v] : parameters) {
      params.insert({get_name(prelimiter, k), v});
    }
    for (const auto& [k, v] : buffers) {
      params.insert({get_name(prelimiter, k), v});
    }
    for (const auto& [k, v] : submodules) {
      v.named_parameters(params, /*prelimiter=*/k);
    }
  }

  std::map<std::string, array> named_parameters() {
    std::map<std::string, array> model_params{};
    named_parameters(model_params);
    return model_params;
  }

  void update(std::unordered_map<std::string, array&> trained_weights) {
    auto model_weights = this->named_parameters();
    for (auto& [k, v] : trained_weights) {
      //   if (!model_weights.containsKey(k)) {
      //     // Raise Error
      //   }
      //   if (!(v->shape() == model_weights[k]->shape())) {
      //     // Raise Error
      //   }
      model_weights.at(k) = v;
    }
  }

  void print_parameters() {
    auto model_weights = this->named_parameters();
    std::cout << "\n[\nparameters:\n";
    for (auto& [k, v] : model_weights) {
      std::cout << k << ":\n" << v << "\n";
    }
    std::cout << "]\n";
  }
};

class LinearLayer : public Module {
 public:
  int input_dim, output_dim;
  bool with_bias = true;

  LinearLayer() = default;
  LinearLayer(const LinearLayer&) = default;
  LinearLayer(int in_features, int out_features, bool _with_bias = true) {
    input_dim = in_features;
    output_dim = out_features;
    array weight = random::normal({out_features, in_features}, float32);
    array bias = random::normal({out_features}, float32);

    register_parameter("weight", weight);
    register_parameter("bias", bias);
    with_bias = _with_bias;
  }

  ~LinearLayer() = default;

  array forward(const array& input) override {
    // Check if input size matches number of weights in first layer
    if (input.shape(-1) != parameters.at("weight").shape(-1)) {
      throw std::invalid_argument(
          "Input size doesn't match weight vector size");
    }
    // Allocate space for the outputs
    array outputs = matmul(input, transpose(parameters.at("weight")));

    return with_bias ? (outputs + parameters.at("bias")) : outputs;
  }
};

class MyModel : public Module {
 public:
  LinearLayer fc1 = LinearLayer(784, 100, false);
  LinearLayer fc2 = LinearLayer(100, 10, false);

  MyModel() {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    // for (int i = 0; i < 10; i++) {
    //   layers.push_back(LinearLayer(10, 10));
    // }
    // register_layer("layers", layers);
  }
  array forward(const array& x) override {
    auto y = fc2.forward(fc1.forward(x));
    // for (auto& l : layers) {
    //   y = l.forward(y);
    // }
    return y;
  }
};

int main() {
  array input = random::uniform({1, 784});
  MyModel model;
  auto res = model.forward(input);
  std::cout << res << "\n" << res.shape();
  model.print_parameters();
  return 0;
}