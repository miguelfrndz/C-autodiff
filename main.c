#include <stdio.h>
#include <math.h>
#include "autodiff.h"

dual log_norm_likelihood(dual x, dual mu, dual sigma) {
    dual a = dual_div(dual_sub(x, mu), sigma);
    dual b = dual_pow2(a);
    dual c = dual_mul(b, dual_new(-0.5, 0.0));
    dual d = dual_exp(c);
    dual e = dual_mul(d, dual_new(1.0 / (sqrt(2 * M_PI) * sigma.val), 0.0));
    return e;
}

dual functionTest(dual x) {
    dual sin_term = dual_sin(x);
    dual cos_term = dual_cos(x);
    dual x2 = dual_pow2(x);
    return dual_mul(dual_mul(sin_term, cos_term), x2);
}

int main(int argc, char const *argv[]) {
    (void) argc, (void) argv;
    printf("sin(x)*cos(x)*x**2:\n");
    dual x_test = dual_new(1.0, 1.0);
    dual f_test = functionTest(x_test);
    printf("\tf(%f) = %f\n", x_test.val, f_test.val);
    printf("\tf'(%f) = %f\n", x_test.val, f_test.dv);
    printf("\nLog-Normal Likelihood:\n");
    dual x = dual_new(1.0, 0.0);
    dual mu = dual_new(1.0, 1.0);
    dual sigma = dual_new(1.0, 0.0);
    dual f = log_norm_likelihood(x, mu, sigma);
    printf("\tf(%f) = %f\n", x.val, f.val);
    printf("\tf'(%f) = %f\n", x.val, f.dv);
    return 0;
}
