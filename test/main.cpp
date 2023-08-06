#include <iostream>
#include <random>
#include <typeinfo>
#include <type_traits>
#include <histo/histo.hpp>

template <class T>
void histo_test(const std::size_t len) {
	std::printf("Test for %s\n", typeid(T).name());
	std::vector<T> vec(len);

	for (std::size_t i = 0; i < len; i++) {
		vec[i] = (i * i) % 11;
		if (std::is_signed<T>::value && i % 2 == 0) {
			vec[i] *= -1;
		}
	}

	std::printf("Standard\n");
	mtk::histo::print_histogram(vec, 10);
	std::printf("Abs\n");
	mtk::histo::print_abs_histogram(vec, 10);
}

int main() {
	histo_test<std::uint64_t>(100000);
	histo_test<float        >(100000);
}
