#include <iostream>
#include "mlx/mlx.h"

namespace mx = mlx::core;

int main() {
  // Set the device to GPU
  mx::set_default_device(mx::Device::gpu);

  // Create a 2x2 array 'a'
  mx::array a = mx::array({1, 2, 3, 4}, {2, 2});
  std::cout << "Array a:" << std::endl;
  std::cout << a << std::endl;

  // Create another 2x2 array 'b'
  mx::array b = mx::array({5, 6, 7, 8}, {2, 2});
  std::cout << "Array b:" << std::endl;
  std::cout << b << std::endl;

  // Add them together
  mx::array c = a + b;

  // Evaluate the computation
  mx::eval(c);

  // Print the result
  std::cout << "Result of a + b:" << std::endl;
  std::cout << c << std::endl;

  return 0;
}