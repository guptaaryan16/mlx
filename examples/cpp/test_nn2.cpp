#include <any>
#include <iostream>
#include <map>
#include <memory>
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
  Module& register_module(std::string&& sub_name, T&& m) {
    // `register_component` allows you to register the component(in order) as
    // used by the NN
    if (!std::is_base_of<T, Module>::value) {
      // Error the code is not correct
    }
    std::string temp = std::string(sub_name);
    submodules.insert({temp, m});
    return submodules.at(temp);
  }

  template <typename T>
  void _register_module(std::string_view sub_name, T& m) {
    // `register_component` allows you to register the component(in order) as
    // used by the NN
    if (!std::is_base_of<T, Module>::value) {
      // Error the code is not correct
    }
    std::string temp = std::string(sub_name);
    submodules.insert({temp, m});
    return submodules.at(temp);
  }

  template <typename T>
  void register_layer(std::string&& sub_name, std::vector<T>& layers) {
    // `register_layer` allows you to register the layers(in order) as
    // used by the NN
    // if (!std::is_base_of<T, Module>::value) {
    //   // Error the code is not correct
    // }
    for (size_t i = 0; i < layers.size(); ++i) {
      std::string l_name = sub_name + std::string(i);
      _register_module(l_name, layers[i]);
    }
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

array relu(array& x) {
  return max(x, 0);
}

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

class LayerNorm : public Module {
 public:
  int dims;
  float eps = 1e-5;
  bool affine = true;

  LayerNorm() = default;
  LayerNorm(LayerNorm&) = default;

  LayerNorm(int _dims, float _eps = 1e-5, bool _affine = true) {
    dims = _dims;
    eps = _eps;
    affine = _affine;
    if (affine) {
      array bias = random::normal({dims}, float32);
      array weight = random::normal({dims}, float32);
      register_parameter("bias", bias);
      register_parameter("weight", weight);
    }
  }
  array forward(array& x) {
    array means = mean(x, -1, true);
    array _var = var(x, -1, true);
    array _x = (x - means) * rsqrt(_var + eps);
    return affine ? (parameters.at("bias") * _x + parameters.at("weight")) : _x;
  }
};

class Dropout : public Module {
 public:
  float p = 0.5;
  float _p_1 = 1 - p;

  Dropout(const Dropout&) = default;
  Dropout(float _p = 0.5) {
    p = _p;
    _p_1 = 1 - _p_1;
  };

  array forward(array& x) {
    if (_p_1 == 1.0)
      return x;
    array mask = random::bernoulli(_p_1, x.shape());
    return (1 / _p_1) * mask * x;
  }
};

class MultiHeadAttention : public Module {
 public:
  LinearLayer query_proj, key_proj, value_proj, out_project;

  int dims, num_heads;
  int query_input_dims = NULL;
  int key_input_dims = NULL;
  int value_input_dims = NULL;
  int value_dims = NULL;
  int value_output_dims = NULL;
  bool bias = false;

 public:
  MultiHeadAttention() = default;
  MultiHeadAttention(
      int dims,
      int num_heads,
      int query_input_dims = 0,
      int key_input_dims = 0,
      int value_input_dims = NULL,
      int value_dims = NULL,
      int value_output_dims = NULL,
      bool bias = false) {
    query_input_dims = query_input_dims || dims;
    key_input_dims = key_input_dims || dims;
    value_input_dims = value_input_dims || key_input_dims;
    value_dims = value_dims || dims;
    value_output_dims = value_output_dims || dims;
    key_input_dims = key_input_dims || dims;
    value_input_dims = value_input_dims || key_input_dims;
    value_dims = value_dims || dims;
    value_output_dims = value_output_dims || dims;

    this->num_heads = num_heads;

    query_proj = LinearLayer(query_input_dims, dims, bias = bias);
    key_proj = LinearLayer(key_input_dims, dims, bias = bias);
    value_proj = LinearLayer(value_input_dims, value_dims, bias = bias);
    out_project = LinearLayer(value_dims, value_output_dims, bias = bias);
    register_module("q_proj", query_proj);
    register_module("k_proj", key_proj);
    register_module("v_proj", value_proj);
    register_module("o_project", out_project);
  }
  array forward(array& queries, array& keys, array& values, array& mask) {
    array _queries = query_proj.forward(queries);
    array _keys = key_proj.forward(keys);
    array _values = value_proj.forward(values);
    int B = _queries.shape(0), L = _queries.shape(1), D = _queries.shape(2),
        S = _keys.shape(1);
    _queries =
        transpose(reshape(_queries, {B, L, num_heads, -1}), {0, 2, 1, 3});
    _keys = transpose(reshape(_keys, {B, S, num_heads}), {2, 3, 1});
    _values = transpose(reshape(_values, {B, S, num_heads, -1}), {0, 2, 1, 3});
    eval(_queries);
    eval(_keys);
    eval(_values);

    float scale = std::sqrt(1 / _queries.shape(-1));
    array scores = matmul((_queries * scale), _keys);
    scores = softmax(scores, -1);
    array values_hat =
        reshape(transpose(matmul(scores, _values), {0, 2, 1, 3}), {B, L, -1});
    return out_project.forward(values_hat);
  }
};

class TransformerEncoderLayer : public Module {
 public:
  int dims, num_heads;
  int mlp_dims = NULL;
  float dropout = 0.0f;
  array (*activation)(array&);
  MultiHeadAttention attention;
  LayerNorm ln1, ln2;
  LinearLayer linear1, linear2;
  Dropout dropout1, dropout2;
  bool norm_first;

  TransformerEncoderLayer() = default;
  TransformerEncoderLayer(
      int dims,
      int num_heads,
      int mlp_dims = NULL,
      float dropout = 0.0,
      array activation(array&) = relu,
      bool norm_first = true) {
    mlp_dims = mlp_dims || dims * 4;
    attention = MultiHeadAttention(dims, num_heads);
    ln1 = LayerNorm(dims);
    ln2 = LayerNorm(dims);
    linear1 = LinearLayer(dims, mlp_dims);
    linear2 = LinearLayer(mlp_dims, dims);
    dropout1 = Dropout(dropout);
    dropout2 = Dropout(dropout);

    register_module("linear1", linear1);
    register_module("linear2", linear2);
    register_module("dropout1", dropout1);
    register_module("dropout2", dropout2);

    this->activation = activation;
    this->norm_first = norm_first;
  }

  array forward(array x, array mask) {
    if (norm_first) {
      array y = ln1.forward(x);
      y = attention.forward(y, y, y, mask);
      // y = dropout1(y);
      x = x + y;
      eval(x);
      y = ln2.forward(x);
      y = linear1.forward(y);
      y = activation(y);
      // y = dropout2.forward(y);
      y = linear2.forward(y);
      y = x + y;
      return y;
    } else {
      array y = attention.forward(x, x, x, mask);
      y = dropout1.forward(y);
      array z = x + y;
      eval(z);
      y = ln1.forward(z);
      y = linear1.forward(y);
      y = activation(y);
      // y = dropout2(y);
      y = linear2.forward(y);
      y = ln2.forward(y);
      return y;
    }
  }
};

class MyModel : public Module {
 public:
  LinearLayer fc1 = LinearLayer(784, 100, false);
  LinearLayer fc2 = LinearLayer(100, 10, false);
  std::vector<LinearLayer&> layers{};

  LayerNorm ln = LayerNorm(10);
  Dropout dt = Dropout(0.2);
  MyModel() {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("ln", ln);
    register_module("dt", dt);
    for (int i = 0; i < 10; i++) {
      layers.push_back(LinearLayer(10, 10));
    }
    register_layer("layers", layers);
  }
  array forward(const array& x) override {
    auto y = fc2.forward(fc1.forward(x));
    for (auto& l : layers) {
      y = l.forward(y);
    }
    y = ln.forward(y);
    y = dt.forward(y);
    eval(y);
    return y;
  }
};

int main() {
  array input = random::uniform({1, 784});
  MyModel model;
  auto res = model.forward(input);
  std::cout << res << "\n" << res.shape();
  std::cout << "\n" << sum(res) << std::endl;
  model.print_parameters();
  return 0;
}