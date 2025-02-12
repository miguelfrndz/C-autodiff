#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include "autodiff.hpp"

using namespace autodiff;

dual log_normal_likelihood(dual x, dual mu, dual sigma) {
    dual a = (x - mu) / sigma;
    dual b = pow2(a);
    dual c = b * dual(-0.5);
    dual d = c - log(sigma);
    dual e = d - dual(0.5 * log(2 * M_PI));
    return e;
}

dual functionTest(dual x) {
    dual sin_term = sin(x);
    dual cos_term = cos(x);
    dual x2 = pow2(x);
    return sin_term * cos_term * x2;
}

int main(int argc, char const *argv[]) {
    (void) argc, (void) argv;

    // Forward-Mode Examples
    dual x_test = dual(1.0, 1.0, "x");
    dual f_test = functionTest(x_test);
    f_test.name = "test_function";
    std::cout << "sin(x)*cos(x)*x**2:" << std::endl;
    // f_test.print();
    std::cout << f_test;

    dual x = dual(10.0, "x");
    dual mu = dual(5.0, 1.0, "mu"); // mu is the variable of interest
    dual sigma = dual(2.0, "sigma");
    dual f = log_normal_likelihood(x, mu, sigma);
    f.name = "log_normal_likelihood";
    std::cout << "\nLog-Normal Likelihood:" << std::endl;
    f.print();

    // Reverse-Mode Examples
    // TODO: Not Yet Implemented...
    return 0;
}
