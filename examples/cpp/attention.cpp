#include <iostream>
#include "mlx/mlx.h"

using namespace mlx::core;

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
    return affine ? (parameters["bias"] * _x + parameters["weights"]) : _x;
  }
};

class Dropout : public Module {
 public:
  float p = 0.5;
  float _p_1 = 1 - p;

  Dropout(float _p = 0.5) {
    p = _p;
    _p_1 = 1 - _p_1;
  }

  array forward(array& x) {
    if (_p_1 == 1.0)
      return x;
    array mask = random::bernoulli(_p_1, x.shape());
    return (1 / _p_1) * mask * x;
  }
};

// struct SingleHeadAttention {
//  private:
//   int hidden_size;
//   bool bias;
//   struct LinearLayer Wqkv, Wo;

//  public:
//   SingleHeadAttention(int hidden_size, bool bias = true) {
//     Wqkv = LinearLayer(hidden_size, (hidden_size / 4) * 3, bias = bias);
//     Wo = LinearLayer(hidden_size / 4, hidden_size, bias = bias);
//     this->bias = bias;
//   }
//   array forward(array& x) {
//     int B = x.shape(0), S = x.shape(1), C = x.shape(2);
//     std::vector<int> dim = {B, S, 3, C / 4};
//     std::cout << dim << std::endl;
//     std::vector<array> qkv = split(reshape(Wqkv.forward(x), dim), 3, 2);

//     std::cout << qkv[0] << std::endl;
//     std::cout << qkv[1] << std::endl;
//     std::cout << qkv[2] << std::endl;
//     // array attn = matmul(squeeze(qkv[0]), transpose(squeeze(qkv[1])));
//     // attn = attn / sqrt(qkv[1].size());
//     // attn = softmax(attn);
//     // eval(attn);
//     // auto temp = matmul(attn, squeeze(qkv[2]));
//     // eval(temp);
//     // return Wo.forward(temp);
//     return x;
//   }
//   // void load_weights(std::vector<std::vector<array>> weights){
//   // }
// };

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
      int query_input_dims = NULL,
      int key_input_dims = NULL,
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

array relu(array& x) {
  return x;
}

// struct EmbeddingTable {};

class TransformerEncoderLayer : public Module {
 public:
  int dims, num_heads;
  int mlp_dims = NULL;
  float dropout = 0.0f;
  array (*activation)(array&);
  struct MultiHeadAttention attention;
  struct LayerNorm ln1, ln2;
  struct LinearLayer linear1, linear2;
  struct Dropout dropout1, dropout2;
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

      y = ln2.forward(x);
      y = linear1.forward(y);
      y = activation(y);
      // y = dropout2.forward(y);
      y = linear2.forward(y);
      y = x + y;
      eval(y);
      return y;
    } else {
      array y = attention.forward(x, x, x, mask);
      y = dropout1.forward(y);
      array z = x + y;
      y = ln1.forward(z);

      y = linear1.forward(y);
      y = activation(y);
      // y = dropout2(y);
      y = linear2.forward(y);
      y = ln2.forward(y);
      eval(y);
      return y;
    }
  }
};

struct TransformerEncoder {
 private:
  int num_layers;
  int dims;
  int num_heads;
  int mlp_heads;
  int mlp_dims;
  float dropout = 0.0f;
  array (*activation)(array&);
  bool norm_first = false;
  std::vector<TransformerEncoderLayer> layers;
  LayerNorm ln;
  // checkpoint
 public:
  TransformerEncoder() = default;
  TransformerEncoder(
      int _num_layers,
      int _dims,
      int _num_heads,
      int _mlp_dims,
      float _droupout = 0.0,
      array (*_activation)(array&) = relu,
      bool _norm_first = true) {
    num_layers = _num_layers;
    dims = _dims;
    num_heads = _num_heads;
    mlp_dims = _mlp_dims;
    dropout = _droupout;
    activation = _activation;
    norm_first = _norm_first;

    for (int i = 0; i < num_layers; i++) {
      layers.push_back(TransformerEncoderLayer(
          dims, num_heads, mlp_dims, dropout, activation, norm_first));
    }
    ln = LayerNorm(dims);
  }

  array forward(array x, array mask) {
    array _x = x;
    for (int i = 0; i < num_layers; i++) {
      _x = layers[i].forward(_x, mask);
    }
    return ln.forward(_x);
  }
};

struct TransformerDecoderLayer {
 private:
  int dims;
  int num_heads;
  int mlp_dims;
  float dropout;
  bool norm_first;
  MultiHeadAttention self_attention, cross_attention;
  LayerNorm ln1, ln2, ln3;
  LinearLayer linear1, linear2;
  Dropout dropout1, dropout2, dropout3;
  array (*activation)(array&);

 public:
  TransformerDecoderLayer() = default;
  TransformerDecoderLayer(
      int _dims,
      int _num_heads,
      int _mlp_dims,
      float _dropout,
      array (*_activation)(array&) = relu,
      bool _norm_first = true) {
    dims = _dims;
    mlp_dims = _mlp_dims || _dims * 4;
    activation = _activation;
    norm_first = _norm_first;
    dropout = _dropout;

    ln1 = LayerNorm(dims);
    ln2 = LayerNorm(dims);
    ln3 = LayerNorm(dims);
    linear1 = LinearLayer(dims, mlp_dims);
    linear2 = LinearLayer(mlp_dims, dims);
    dropout1 = Dropout(dropout);
    dropout2 = Dropout(dropout);
    dropout3 = Dropout(dropout);
    activation = _activation;
    norm_first = _norm_first;
  }
  array forward(array& x, array& memory, array& x_mask, array& memory_mask) {
    if (norm_first) {
      array y = ln1.forward(x);
      y = self_attention.forward(y, y, y, x_mask);
      // y = dropout1.forward(y);
      x = x + y;

      y = ln2.forward(x);
      y = cross_attention.forward(y, memory, memory, memory_mask);
      // y = dropout2.forward(y);
      x = x + y;

      y = ln3.forward(x);
      y = linear1.forward(y);
      y = activation(y);
      y = dropout3.forward(y);
      y = linear2.forward(y);
      y = x + y;

      eval(y);
      return y;
    } else {
      array y = self_attention.forward(x, x, x, x_mask);
      y = dropout1.forward(y);
      array z = x + y;
      x = ln1.forward(z);

      y = cross_attention.forward(y, memory, memory, memory_mask);
      // y = dropout2,forward(y);
      z = x + y;
      x = ln1.forward(z);

      y = linear1.forward(x);
      y = activation(y);
      y = dropout3.forward(y);
      y = linear2.forward(y);
      z = x + y;
      y = ln3.forward(z);
      eval(y);

      return y;
    }
  }
};

struct TransformerDecoder {
 private:
  int num_layers;
  int dims;
  int num_heads;
  int mlp_dims;
  float dropout;
  array (*activation)(array&);
  bool norm_first;
  std::vector<TransformerDecoderLayer> layers;
  LayerNorm ln;

 public:
  TransformerDecoder() = default;
  TransformerDecoder(
      int _num_layers,
      int _dims,
      int _num_heads,
      int _mlp_dims,
      float _dropout = 0.0,
      array (*_activation)(array&) = relu,
      bool _norm_first = true) {
    num_layers = _num_layers;
    dims = _dims;
    num_heads = _num_heads;
    mlp_dims = _mlp_dims;
    dropout = _dropout;
    activation = _activation;
    norm_first = _norm_first;
    for (int i = 0; i < num_layers; i++) {
      layers.push_back(TransformerDecoderLayer(
          dims, num_heads, mlp_dims, dropout, activation, norm_first));
    }
    ln = LayerNorm(dims);
  }

  array forward(array& x, array& memory, array& x_mask, array& memory_mask) {
    for (int i = 0; i < num_layers; i++) {
      x = layers[i].forward(x, memory, x_mask, memory_mask);
    }
  }
};

struct Transformer {
 private:
  int dims = 512;
  int num_heads = 8;
  int num_encoder_layers = 6;
  int num_decoder_layers = 6;
  int mlp_dims = NULL;
  float dropout = 0.0;
  bool norm_first = true;
  bool checkpoint = false;
  array (*activation)(array&);
  TransformerEncoder encoder;
  TransformerDecoder decoder;

 public:
  Transformer(
      int _dims = 512,
      int _num_heads = 8,
      int _num_encoder_layers = 6,
      int _num_decoder_layers = 6,
      int _mlp_dims = NULL,
      float _dropout = 0.0,
      array (*_activation)(array&) = relu,
      bool _norm_first = true) {
    dims = _dims;
    num_heads = _num_heads;
    num_encoder_layers = _num_encoder_layers;
    num_decoder_layers = _num_decoder_layers;
    mlp_dims = _mlp_dims;
    dropout = _dropout;
    activation = _activation;
    norm_first = _norm_first;

    encoder = TransformerEncoder(
        num_decoder_layers,
        dims,
        num_heads,
        mlp_dims,
        dropout,
        activation,
        norm_first);
    decoder = TransformerDecoder(
        num_decoder_layers,
        dims,
        num_heads,
        mlp_dims,
        dropout,
        activation,
        norm_first);
  }

  array forward(
      array src,
      array tgt,
      array src_mask,
      array tgt_mask,
      array memory_mask) {
    array memory = encoder.forward(src, src_mask);
    return decoder.forward(tgt, memory, tgt_mask, memory_mask);
  }
};

int main(int argc, char* argv[]) {
  array x = random::normal({10, 32, 3});
  // auto SL = SingleHeadAttention(5);
  // auto y = SL.forward(x);
  // std::cout << y << std::endl;
  // std::cout << y.shape() << std::endl;

  LayerNorm ln = LayerNorm(x.ndim());
  std::cout << ln.forward(x) << std::endl;
  return 0;
}