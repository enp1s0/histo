#pragma once
#include <iostream>
#include <limits>
#include <vector>
#include <type_traits>
#include <omp.h>

namespace mtk {
namespace histo {
namespace detail {
template <class T, std::enable_if_t<std::is_signed<T>::value, bool> = true>
T abs(const T v) {return std::abs(v);}
template <class T, std::enable_if_t<std::is_unsigned<T>::value, bool> = true>
T abs(const T v) {return v;}
} // namespace detail

template <class T>
void print_abs_histogram(
  const T* const ptr,
  const std::size_t len,
  const std::size_t num_buckets,
  const std::size_t num_total_asterisks = 100
  ) {
  T max = 0, min = std::numeric_limits<T>::max();

#pragma omp parallel for reduction(max: max) reduction(min: min)
  for (std::size_t i = 0; i < len; i++) {
    const auto v = detail::abs(ptr[i]);
    max = std::max(v, max);
    min = std::min(v, min);
  }

  // bucket_width = (max - min) / num_buckets
  // [min + 0 * (max - min) / num_buckets, min + 1 * (max - min) / num_buckets]
  // (min + 1 * (max - min) / num_buckets, min + 2 * (max - min) / num_buckets]
  // (min + 3 * (max - min) / num_buckets, min + 4 * (max - min) / num_buckets]
  // ...
  // (                                   , max]

  std::vector<std::size_t> counter(num_buckets, 0);
#pragma omp parallel num_threads(16)
  {
    std::vector<std::size_t> local_counter(num_buckets, 0);
    for (std::size_t i = omp_get_thread_num(); i < len; i += omp_get_num_threads()) {
      const auto v = detail::abs(ptr[i]);
      auto index = static_cast<std::size_t>(num_buckets * (v - min) / (max - min));
      if (index >= num_buckets) {
        index = num_buckets - 1;
      }
      local_counter[index]++;
    }
    for (std::size_t i = 0; i < num_buckets; i++) {
#pragma omp critical
      {
        counter[i] += local_counter[i];
      }
    }
  }

  for (std::size_t i = 0; i < num_buckets; i++) {
    const auto range_min = min + i       * static_cast<double>(max - min) / num_buckets;
    const auto range_max = min + (i + 1) * static_cast<double>(max - min) / num_buckets;
    if (i == 0) {
      std::printf("[");
    } else {
      std::printf("(");
    }
    std::printf(
      "%.5e, %.5e](%10lu; %e%%): ",
      range_min, range_max,
      counter[i],
      static_cast<double>(counter[i]) / len * 100
      );
    for (std::size_t j = 0; j < num_buckets * static_cast<double>(counter[i]) / len; j++) {
      std::printf("*");
    }
    std::printf("\n");
  }
}

template <class T>
void print_abs_histogram(
  const std::vector<T> vec,
  const std::size_t num_buckets,
  const std::size_t num_total_asterisks = 100
  ) {
  mtk::histo::print_abs_histogram(
    vec.data(),
    vec.size(),
    num_buckets,
    num_total_asterisks
    );
}

} // namespace histo
} // namespace mtk
