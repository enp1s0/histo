#pragma once
#include <iostream>
#include <cstdint>
#include <cmath>
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

template <class T, class Func>
auto print_histogram_core(
    const T* const ptr,
    const std::size_t len,
    const std::size_t num_buckets,
    const std::size_t num_total_asterisks,
    Func pre_process,
    const double pre_min,
    const double pre_max
    ) -> std::enable_if_t<std::is_arithmetic_v<T>, void> {
  T max = std::numeric_limits<T>::min(), min = std::numeric_limits<T>::max();

  if (std::isnan(pre_max) || std::isnan(pre_min)) {
#pragma omp parallel for reduction(max: max) reduction(min: min)
    for (std::size_t i = 0; i < len; i++) {
      const auto v = pre_process(ptr[i]);
      max = std::max(v, max);
      min = std::min(v, min);
    }
  }
  if (!std::isnan(pre_min)) {
    min = pre_min;
  }
  if (!std::isnan(pre_max)) {
    max = pre_max;
  }

  // bucket_width = (max - min) / num_buckets
  // [min + 0 * (max - min) / num_buckets, min + 1 * (max - min) / num_buckets]
  // (min + 1 * (max - min) / num_buckets, min + 2 * (max - min) / num_buckets]
  // (min + 3 * (max - min) / num_buckets, min + 4 * (max - min) / num_buckets]
  // ...
  // (                                   , max]

  std::size_t num_out = 0;
  std::vector<std::size_t> counter(num_buckets, 0);
#pragma omp parallel reduction(+: num_out)
  {
    std::vector<std::size_t> local_counter(num_buckets, 0);
    for (std::size_t i = omp_get_thread_num(); i < len; i += omp_get_num_threads()) {
      const auto v = pre_process(ptr[i]);

      if (min <= v && v <= max) {
        auto index = static_cast<std::size_t>(num_buckets * (v - min) / (max - min));
        if (index >= num_buckets) {
          index = num_buckets - 1;
        }
        local_counter[index]++;
      } else {
        num_out++;
      }
    }
    for (std::size_t i = 0; i < num_buckets; i++) {
#pragma omp critical
      {
        counter[i] += local_counter[i];
      }
    }
  }

  std::uint32_t count_width = 0;
  for (std::size_t i = 0; i < num_buckets; i++) {
    count_width = std::max(
        count_width,
        static_cast<std::uint32_t>(
            std::floor(std::log10(counter[i])) + 1
            )
        );
  }

  std::size_t acc = 0;
  for (std::size_t i = 0; i < num_buckets; i++) {
    const auto range_min = min + i       * static_cast<double>(max - min) / num_buckets;
    const auto range_max = min + (i + 1) * static_cast<double>(max - min) / num_buckets;
    if (i == 0) {
      std::printf("[");
    } else {
      std::printf("(");
    }
    acc += counter[i];
    std::printf(
        "%+.5e, %+.5e](%*lu; %e%%; %6.2f%%): ",
        range_min, range_max,
        count_width,
        counter[i],
        static_cast<double>(counter[i]) / len * 100,
        static_cast<double>(acc) / len * 100
        );
    for (std::size_t j = 0; j < num_total_asterisks * static_cast<double>(counter[i]) / len; j++) {
      std::printf("*");
    }
    std::printf("\n");
  }
  std::printf("Out-of-range: %lu\n", num_out);
}

template <class T, class Func>
auto print_histogram_descrete_int_core(
    const T* const ptr,
    const std::size_t len,
    const std::size_t num_total_asterisks,
    Func pre_process,
    const double pre_min,
    const double pre_max
    ) -> std::enable_if_t<std::is_integral_v<T>, void> {
  T max = std::numeric_limits<T>::min(), min = std::numeric_limits<T>::max();

  if (std::isnan(pre_max) || std::isnan(pre_min)) {
#pragma omp parallel for reduction(max: max) reduction(min: min)
    for (std::size_t i = 0; i < len; i++) {
      const auto v = pre_process(ptr[i]);
      max = std::max(v, max);
      min = std::min(v, min);
    }
  }
  if (!std::isnan(pre_min)) {
    min = pre_min;
  }
  if (!std::isnan(pre_max)) {
    max = pre_max;
  }
  const std::size_t num_buckets = max - min + 1;

  std::size_t num_out = 0;
  std::vector<std::size_t> counter(num_buckets, 0);
#pragma omp parallel reduction(+: num_out)
  {
    std::vector<std::size_t> local_counter(num_buckets, 0);
    for (std::size_t i = omp_get_thread_num(); i < len; i += omp_get_num_threads()) {
      const auto v = pre_process(ptr[i]);

      if (min <= v && v <= max) {
        auto index = v - min;
        local_counter[index]++;
      } else {
        num_out++;
      }
    }
    for (std::size_t i = 0; i < num_buckets; i++) {
#pragma omp critical
      {
        counter[i] += local_counter[i];
      }
    }
  }

  std::uint32_t count_width = 0;
  for (std::size_t i = 0; i < num_buckets; i++) {
    count_width = std::max(
        count_width,
        static_cast<std::uint32_t>(
            std::floor(std::log10(counter[i])) + 1
            )
        );
  }
  const std::uint32_t value_width = std::floor(std::log10(std::max(detail::abs(min), detail::abs(max))) + 1) + 1 /* = For sign char*/;

  std::size_t acc = 0;
  for (std::size_t i = 0; i < num_buckets; i++) {
    acc += counter[i];
    std::printf(
        "[%+*lld](%*lu; %e%%; %6.2f%%): ",
        value_width,
        static_cast<long long int>(i + min),
        count_width,
        counter[i],
        static_cast<double>(counter[i]) / len * 100,
        static_cast<double>(acc) / len * 100
        );
    for (std::size_t j = 0; j < num_total_asterisks * static_cast<double>(counter[i]) / len; j++) {
      std::printf("*");
    }
    std::printf("\n");
  }
  std::printf("Out-of-range: %lu\n", num_out);
}
} // namespace detail

enum class preprocess_type_t {
  identity = 0,
  abs = 1
};

enum class bucket_type_t {
  standard_range = 0,
  discrete_int = 1
};

struct params_t {
  double range_min = std::nan("");
  double range_max = std::nan("");
  std::size_t num_buckets = 0;
  std::size_t num_total_asterisks = 100;
  preprocess_type_t preprocess_type = preprocess_type_t::identity;
  bucket_type_t bucket_type = bucket_type_t::standard_range;
};

template <class T>
void print_abs_histogram(
    const T* vec,
    const std::size_t len,
    const std::size_t num_buckets,
    const std::size_t num_total_asterisks = 100
    ) {
  params_t params;
  params.num_buckets = num_buckets;
  params.num_total_asterisks = num_total_asterisks;
  params.preprocess_type = preprocess_type_t::abs;
  print_histogram(params, vec, len);
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

template <class T>
void print_histogram(
    const T* vec,
    const std::size_t len,
    const std::size_t num_buckets,
    const std::size_t num_total_asterisks = 100
    ) {
  params_t params;
  params.num_buckets = num_buckets;
  params.num_total_asterisks = num_total_asterisks;
  params.preprocess_type = preprocess_type_t::identity;
  print_histogram(params, vec, len);
}

template <class T>
void print_histogram(
    const std::vector<T> vec,
    const std::size_t num_buckets,
    const std::size_t num_total_asterisks = 100
    ) {
  mtk::histo::print_histogram(
      vec.data(),
      vec.size(),
      num_buckets,
      num_total_asterisks
      );
}

template <class T>
void print_histogram(
    const params_t& params,
    const T* const vec,
    const std::size_t len
    ) {
  if (params.bucket_type == bucket_type_t::standard_range) {
    std::size_t num_buckets = params.num_buckets == 0 ? 10 : params.num_buckets;
    if (params.preprocess_type == preprocess_type_t::identity) {
      mtk::histo::detail::print_histogram_core(
          vec,
          len,
          num_buckets,
          params.num_total_asterisks,
          [=](const T a) -> T {return a;},
          params.range_min,
          params.range_max
          );
    } else if (params.preprocess_type == preprocess_type_t::abs) {
      mtk::histo::detail::print_histogram_core(
          vec,
          len,
          num_buckets,
          params.num_total_asterisks,
          [=](const T a) -> T {return mtk::histo::detail::abs(a);},
          params.range_min,
          params.range_max
          );
    }
  } else if (params.bucket_type == bucket_type_t::discrete_int) {
    if constexpr (std::is_integral_v<T>) {
      if (params.num_buckets != 0) {
        std::printf("[histo warn]: Specified num_buckets is ignored in the discrete_int bucket type\n");
      }
      if (params.preprocess_type == preprocess_type_t::identity) {
        mtk::histo::detail::print_histogram_descrete_int_core(
            vec,
            len,
            params.num_total_asterisks,
            [=](const T a) -> T {return a;},
            params.range_min,
            params.range_max
            );
      } else if (params.preprocess_type == preprocess_type_t::abs) {
        mtk::histo::detail::print_histogram_descrete_int_core(
            vec,
            len,
            params.num_total_asterisks,
            [=](const T a) -> T {return mtk::histo::detail::abs(a);},
            params.range_min,
            params.range_max
            );
      }
    } else {
      std::fprintf(stderr, "[histo error]: discrete_int bucket type only supports integer data types");
    }
  }
}

template <class T>
void print_histogram(
    const params_t& params,
    const std::vector<T>& vec
    ) {
  print_histogram(params, vec.data(), vec.size());
}

namespace utils {
template <class T>
std::pair<double, double> calc_mean_and_var(
    const T* const ptr,
    const std::size_t len
    ) {
  double sum = 0;
#pragma omp parallel for reduction(+: sum)
  for (std::size_t i = 0; i < len; i++) {
    sum += ptr[i];
  }
  const auto mean = sum / len;

  sum = 0;
#pragma omp parallel for reduction(+: sum)
  for (std::size_t i = 0; i < len; i++) {
    const auto v = mean - ptr[i];
    sum += v * v;
  }
  const auto var = sum / (len - 1);

  return std::make_pair(mean, var);
}
template <class T>
std::pair<double, double> calc_mean_and_var(
    const std::vector<T>& vec
    ) {
  return calc_mean_and_var(vec.data(), vec.size());
}
} // namespace utils
} // namespace histo
} // namespace mtk
