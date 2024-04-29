#include <fstream>
#include <iostream>
#include <vector>
#include "mlx/mlx.h" // Include MLX array API

using namespace mlx::core;

array EPS = array({1e-5});

// typedef array array;
// typedef std::vector<array> tensor2d;
// typedef std::vector<tensor2d> array;

struct Config {
  int dim; // transformer dimension
  int hidden_dim; // for ffn layers
  int n_layers; // number of layers
  int n_heads; // number of query heads
  int n_kv_heads; // number of key/value heads (can be < query heads because of
                  // multiquery)
  int vocab_size; // vocabulary size, usually 256 (byte-level)
  int seq_len; // max sequence length
};

struct TransformerWeights {
  array token_embedding_table; // [vocab_size, dim]
  // weights for rmsnorms
  array rms_att_weight; // [layer, dim]
  array rms_ffn_weight; // [layer, dim]
  // weights for attention matmuls
  array wq; // [layer, dim, dim]
  array wk; // [layer, dim, dim]
  array wv; // [layer, dim, dim]
  array wo; // [layer, dim, dim]
  // weights for ffn
  array w1; // [layer, hidden_dim, dim]
  array w2; // [layer, dim, hidden_dim]
  array w3; // [layer, hidden_dim, dim]
  // final x
  array rms_final_weight; // [dim]
  // freq_cis for RoPE relatively positional embeddings
  array freq_cis_real; // [seq_len, (dim/n_heads)/2]
  array freq_cis_imag; // [seq_len, (dim/n_heads)/2]
};

struct RunState {
  // current wave of activations
  array x; // activation at current time stamp [dim]
  array xb; // same, but inside a residual branch [dim]
  array xb2; // an additional buffer just for convenience [dim]
  array hb; // buffer for hidden dimension in the ffn [hidden_dim]
  array hb2; // another buffer for hidden dimension in the ffn [hidden_dim]
  array q; // query [dim]
  array k; // key [dim]
  array v; // value [dim]
  array attention; // buffer for scores/attention values [seq_len]
  array logits; // buffer for logits [vocab_size]
  // kv cache
  array key_cache; // [layer, seq_len, dim]
  array value_cache; // [layer, seq_len, dim]
};

// --------------------------------------------------------------------------------------
// Tensor allocation and deallocation

void rmsnorm(array& output, array& input, array& weight, StreamOrDevice s) {
  array ss = sum(square(input, s), s);
  ss = ss / input.size() + EPS;
  array inv_ss = 1 / sqrt(ss);
  output = multiply(multiply(input, weight, s), inv_ss, s);
}

// void accum(array& lhs, std::vector<array>& rhs) {
//   lhs = lhs + rhs[0];
// }

// void transformer(
//     int token_index,
//     int token_position,
//     Config& config,
//     RunState& state,
//     TransformerWeights& transformer_weights) {
//   // a few convenience variables
//   int dim = config.dim;
//   int hidden_dim = config.hidden_dim;
//   int head_size = dim / config.n_heads;

//   // copy the token embedding into x
//   copy(state.x, transformer_weights.token_embedding_table.item(token_index));

//   for (int layer = 0; layer < config.n_layers; ++layer) {
//     // attention rmsnorm
//     rmsnorm(state.xb, state.x, transformer_weights.rms_att_weight[layer]);

//     // attention
//     matmul(state.q, state.xb, transformer_weights.wq.item(layer));
//     matmul(state.k, state.xb, transformer_weights.wk.item(layer));
//     matmul(state.v, state.xb, transformer_weights.wv.item(layer));
//     // apply RoPE positional embeddings
//     for (int head = 0; head < config.n_heads; ++head) {
//       int start = head * head_size;
//       for (int i = 0; i < head_size; i += 2) {
//         float q0 = state.q[start + i];
//         float q1 = state.q[start + i + 1];
//         float k0 = state.k[start + i];
//         float k1 = state.k[start + i + 1];
//         float fcr = transformer_weights.freq_cis_real[token_position][i / 2];
//         float fci = transformer_weights.freq_cis_imag[token_position][i / 2];
//         state.q[start + i] = q0 * fcr - q1 * fci;
//         state.q[start + i + 1] = q0 * fci + q1 * fcr;
//         state.k[start + i] = k0 * fcr - k1 * fci;
//         state.k[start + i + 1] = k0 * fci + k1 * fcr;
//       }
//     }

//     // save key/value in cache
//     copy(state.key_cache[layer][token_position], state.k);
//     copy(state.value_cache[layer][token_position], state.v);
//     // multiquery attention
//     for (int head = 0; head < config.n_heads; ++head) {
//       for (int timestep = 0; timestep < token_position; ++timestep) {
//         float score = 0;
//         for (int i = 0; i < head_size; ++i)
//           score += state.q.item(head * head_size + i) *
//               state.key_cache[layer][timestep][head * head_size + i];
//         score /= std::sqrt(head_size * 1.0);
//         state.attention[timestep] = score;
//       }

//       // softmax
//       softmax(state.attention, state.attention, token_position + 1);

//       // weighted sum
//       for (int i = 0; i < head_size; ++i) {
//         state.xb[head * head_size + i] = 0;
//         for (int timestep = 0; timestep <= token_position; ++timestep)
//           state.xb[head * head_size + i] += state.attention[timestep] *
//               state.value_cache[layer][timestep][head * head_size + i];
//       }
//     }

//     // final matmul to get the output of the attention
//     matmul(state.xb2, state.xb, transformer_weights.wo[layer]);

//     // residual connection back into x
//     accum(state.x, state.xb2);

//     // ffn rmsnorm
//     rmsnorm(state.xb, state.x, transformer_weights.rms_ffn_weight[layer]);

//     // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x))) *
//     self.w3(x)
//     // first calculate self.w1(x) and self.w3(x)
//     matmul(state.hb, state.xb, transformer_weights.w1[layer]);
//     matmul(state.hb2, state.xb, transformer_weights.w3[layer]);

//     // F.silu; silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
//     for (int i = 0; i < hidden_dim; ++i)
//       state.hb[i] = state.hb[i] * (1.0 / (1.0 + std::exp(-state.hb[i])));

//     // elementwise multiple with w3(x)
//     for (int i = 0; i < hidden_dim; ++i)
//       state.hb[i] = state.hb[i] * state.hb2[i];

//     // final matmul to get the output of the ffn
//     matmul(state.xb, state.hb, transformer_weights.w2[layer]);
//     // residual connection
//     accum(state.x, state.xb);
//   }

//   // final rmsnorm
//   rmsnorm(state.x, state.x, transformer_weights.rms_final_weight);

//   // classifier into logits
//   matmul(state.logits, state.x, transformer_weights.token_embedding_table);
// }

// Define a simple neural network model
struct NeuralNetwork {
  std::vector<array> weights;
  StreamOrDevice device = metal::is_available() ? Device::gpu : Device::cpu;
  // Dummy forward method for inference
  array forward(const array& input) {
    array output = input;
    for (const auto& w : weights) {
      output = matmul(output, w, device);
    }
    return output;
  }
  void print_weights() {
    for (auto i : weights) {
      std::cout << i << std::endl;
    }
  }
  void to(StreamOrDevice& s) {
    device = s;
  }
  void saveSaveTensorsFile(std::string& s){};
  void loadSafetensorsFile(std::string& s){};
};

// Function to save model parameters to file
// void save_model(const NeuralNetwork& model, const std::string& filename)
// {
//   std::ofstream file(filename, std::ios::binary);
//   if (!file.is_open()) {
//     std::cerr << "Error: Unable to open file for writing\n";
//     return;
//   }

//   for (const auto& weight : model.weights) {
//     auto weight_vec = weight;
//     file.write(
//         reinterpret_cast<const char*>(weight_vec.data()),
//         weight_vec.size() * sizeof(double));
//   }

//   file.close();
// }

// // Function to load model parameters from file
// NeuralNetwork load_model(const std::string& filename) {
//   std::ifstream file(filename, std::ios::binary);
//   if (!file.is_open()) {
//     std::cerr << "Error: Unable to open file for reading\n";
//     return NeuralNetwork{};
//   }

//   NeuralNetwork model;

//   // Load weights from file
//   while (!file.eof()) {
//     size_t num_elems;
//     file.read(reinterpret_cast<char*>(&num_elems), sizeof(size_t));

//     std::vector<double> weight_data(num_elems);
//     file.read(
//         reinterpret_cast<char*>(weight_data.data()),
//         num_elems * sizeof(double));

//     model.weights.emplace_back(array(weight_data.data(), num_elems));
//   }

//   file.close();

//   return model;
// }

int main() {
  // Create a sample neural network
  NeuralNetwork model;
  // Assume a network with two layers
  model.weights.emplace_back(random::normal({10, 4})); // Layer 1 weights
  model.weights.emplace_back(random::normal({4, 3})); // Layer 2 weights

  model.print_weights();

  // // Save the model
  // save_model(model, "model.bin");

  // // Load the model
  // NeuralNetwork loaded_model = load_model("model.bin");

  // Test the loaded model
  array input = random::normal(
      {1, 10}); // Input size matches the input dimension of the first layer
  array output = model.forward(input);
  array weights = random::normal({1, 10});
  array o = random::normal({1, 10});
  rmsnorm(o, input, weights, Device::gpu);
  std::cout << "Output: " << output << std::endl;
  std::cout << "Output: " << o << std::endl;
  return 0;
}
