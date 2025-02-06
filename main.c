#include <stdio.h>
#include <math.h>
#include <string.h>
#include "autodiff.h"

dual log_norm_likelihood(dual x, dual mu, dual sigma) {
    dual a = dual_div(dual_sub(x, mu), sigma);
    dual b = dual_pow2(a);
    dual c = dual_mul(b, dual_new(-0.5, 0.0));
    dual d = dual_sub(c, dual_log(sigma));
    dual e = dual_sub(d, dual_new(0.5 * log(2 * M_PI), 0.0));
    return e;
}

Var* log_norm_likelihood_rev(Var *x, Var *mu, Var *sigma) {
    Var *a = var_div(var_sub(x, mu), sigma);
    Var *b = var_pow2(a);
    Var *c = var_mul(b, var_new(-0.5));
    Var *d = var_sub(c, var_log(sigma));
    Var *e = var_sub(d, var_new(0.5 * log(2 * M_PI)));
    return e;
}

dual functionTest(dual x) {
    dual sin_term = dual_sin(x);
    dual cos_term = dual_cos(x);
    dual x2 = dual_pow2(x);
    return dual_mul(dual_mul(sin_term, cos_term), x2);
}

Var* functionTest_rev(Var *x) {
    Var *sin_term = var_sin(x);
    Var *cos_term = var_cos(x);
    Var *x2 = var_pow2(x);
    return var_mul(var_mul(sin_term, cos_term), x2);
}

int main(int argc, char const *argv[]) {
    (void) argc, (void) argv;

    // Forward-Mode Examples
    printf("sin(x)*cos(x)*x**2:\n");
    dual x_test = dual_new(1.0, 1.0);
    dual f_test = functionTest(x_test);
    printf("\tf(%f) = %f\n", x_test.val, f_test.val);
    printf("\tf'(%f) = %f\n", x_test.val, f_test.dv);

    printf("\nLog-Normal Likelihood:\n");
    dual x = dual_new(10.0, 0.0);
    dual mu = dual_new(5.0, 1.0);
    dual sigma = dual_new(2.0, 0.0);
    dual f = log_norm_likelihood(x, mu, sigma);
    printf("\tf(%f) = %f\n", x.val, f.val);
    printf("\tf'(%f) = %f\n", x.val, f.dv);

    // Reverse-Mode Examples
    printf("\n---Reverse-Mode Examples---\n");
    init_tape();
    printf("\nsin(x)*cos(x)*x**2:\n");
    Var *x_reverse = var_new(1.0);
    Var *f_reverse = functionTest_rev(x_reverse);
    backward_pass(f_reverse);
    var_print(x_reverse);
    free_tape();

    printf("\nLog-Normal Likelihood:\n");
    init_tape();
    x_reverse = var_new(10.0);
    strcpy(x_reverse->name, "x");
    Var *mu_reverse = var_new(5.0);
    strcpy(mu_reverse->name, "mu");
    Var *sigma_reverse = var_new(2.0);
    strcpy(sigma_reverse->name, "sigma");
    f_reverse = log_norm_likelihood_rev(x_reverse, mu_reverse, sigma_reverse);
    backward_pass(f_reverse);
    var_print(x_reverse);
    var_print(mu_reverse);
    var_print(sigma_reverse);
    free_tape();
    return 0;
}
