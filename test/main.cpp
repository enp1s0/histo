#include <iostream>
#include <random>
#include <typeinfo>
#include <type_traits>
#include <histo/histo.hpp>

template <class T>
void histo_test(const std::size_t len) {
  std::printf("## %s, Test for dtype = %s\n", __func__, typeid(T).name());
  std::vector<T> vec(len);

  for (std::size_t i = 0; i < len; i++) {
    vec[i] = (i * i) % 13;
    if (std::is_signed<T>::value && i % 2 == 0) {
      vec[i] *= -1;
    }
  }

  std::printf("Standard\n");
  mtk::histo::params_t params;
  params.num_buckets = 20;
  mtk::histo::print_histogram(params, vec);

  std::printf("Abs\n");
  mtk::histo::print_abs_histogram(vec, 10);

  const auto [mean, var] = mtk::histo::utils::calc_mean_and_var(vec.data(), vec.size());
  std::printf("mean = %e, var = %e\n", mean, var);
}

template <class T>
auto histo_descrete_int_test(const std::size_t len) -> std::enable_if_t<std::is_integral_v<T>, void> {
  std::printf("## %s, Test for dtype = %s\n", __func__, typeid(T).name());
  std::vector<T> vec(len);

  for (std::size_t i = 0; i < len; i++) {
    vec[i] = (i * i) % 13;
    if (std::is_signed<T>::value && i % 2 == 0) {
      vec[i] *= -1;
    }
  }

  std::printf("Standard\n");
  mtk::histo::params_t params;
  params.bucket_type = mtk::histo::bucket_type_t::discrete_int;
  mtk::histo::print_histogram(params, vec);

  std::printf("Abs\n");
  params.preprocess_type = mtk::histo::preprocess_type_t::abs;
  mtk::histo::print_histogram(params, vec);

  const auto [mean, var] = mtk::histo::utils::calc_mean_and_var(vec.data(), vec.size());
  std::printf("mean = %e, var = %e\n", mean, var);
}

int main() {
  histo_test<std::uint64_t>(10000000);
  histo_test<float        >(10000000);
  histo_descrete_int_test<std::uint64_t>(10000000);
  histo_descrete_int_test<std::int64_t >(10000000);
}
