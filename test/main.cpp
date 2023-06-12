#include <iostream>
#include <random>
#include <typeinfo>
#include <histo/histo.hpp>

template <class T>
void histo_test(const std::size_t len) {
  std::printf("Test for %s\n", typeid(T).name());
  std::vector<T> vec(len);

  for (std::size_t i = 0; i < len; i++) {
    vec[i] = (i * i) % 11;
  }

  mtk::histo::print_abs_histogram(vec, 10);
}

int main() {
  histo_test<std::uint64_t>(100000);
  histo_test<float        >(100000);
}
