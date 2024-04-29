#include <cmath>
#include <iostream>
#include <map>
#include <vector>
#include "mlx/mlx.h"

using namespace mlx::core;

// class NN{
//     std::map<string, array>params;

// };

int main(int argc, char* argv[]) {
  //   array& a;
  std::map<std::string, array&> params;
  array weights = ones({3, 2});
  array x = random::normal({3, 2});
  array z = array(x);
  //   array& ref = x;
  params.insert({"x", x});
  auto y = random::normal({2, 3, 4});
  std::cout << "x = " << x << std::endl;
  std::cout << "address of x = " << &x << std::endl;
  std::cout << "z = " << z << std::endl;
  std::cout << "address of z = " << &z << std::endl;
  params.at("x") = y;
  std::cout << "x = " << x << std::endl;
  std::cout << "address of x = " << &x << std::endl;
  std::cout << "z = " << z << std::endl;
  std::cout << "address of z = " << &z << std::endl;

  // std::cout << weights.shape() << std::endl; // (3,2)
  // std::cout << weights.shape(0) << std::endl; // 3
  // std::cout << weights.shape(1) << std::endl; // 2
  // std::cout << weights.size() << std::endl; // complete size (3*2) = 6
  // std::cout << weights.ndim() << std::endl;
  // std::cout << multiply(weights, x) << std::endl; // scalar dot product
  // std::cout << matmul(weights, y) << std::endl;
  // std::cout << y << std::endl;
  // std::cout << y / sqrt(10) << std::endl;
  // // std::cout << y.item<float>()<<std::endl;
  // std::cout << y << std::endl;
  // std::cout << std::endl;
  // std::cout << weights + x << std::endl;

  //   std::cout << y << std::endl;
  //   auto z = y->shape(-1);
  //   std::cout << z << std::endl;
  //   std::string filename =
  //       "/Users/guptaaryan16/Desktop/OSS/mlx/examples/cpp/model.safetensors";
  //   SafetensorsLoad wei = load_safetensors(filename);
  //   int i = 0;
  //   for (auto layer : wei.first) {
  //     std::cout << layer.first << std::endl;
  //     std::cout << layer.second << std::endl;
  //     ++i;
  //     if (i == 10)
  //       break;
  //   }
  return 0;
}