#include <math.h>
#include "autodiff.h"

dual dual_new(double val, double dv) {
    dual d = {.val = val, .dv = dv};
    return d;
}

dual dual_add(dual a, dual b) {
    return dual_new(a.val + b.val, a.dv + b.dv);
}

dual dual_sub(dual a, dual b) {
    return dual_new(a.val - b.val, a.dv - b.dv);
}

dual dual_mul(dual a, dual b) {
    return dual_new(a.val * b.val, a.dv * b.val + a.val * b.dv);
}

dual dual_div(dual a, dual b) {
    return dual_new(a.val / b.val, (a.dv * b.val - a.val * b.dv) / (b.val * b.val));
}

dual dual_sin(dual a) {
    return dual_new(sin(a.val), cos(a.val) * a.dv);
}

dual dual_cos(dual a) {
    return dual_new(cos(a.val), -sin(a.val) * a.dv);
}

dual dual_tan(dual a) {
    return dual_new(tan(a.val), a.dv / (cos(a.val) * cos(a.val)));
}

dual dual_exp(dual a) {
    return dual_new(exp(a.val), exp(a.val) * a.dv);
}

dual dual_log(dual a) {
    return dual_new(log(a.val), a.dv / a.val);
}

dual dual_pow(dual a, double b) {
    return dual_new(pow(a.val, b), b * pow(a.val, b - 1) * a.dv);
}

dual dual_pow2(dual a) {
    return dual_new(a.val * a.val, 2 * a.val * a.dv);
}

dual dual_pow3(dual a) {
    return dual_new(a.val * a.val * a.val, 3 * a.val * a.val * a.dv);
}

dual dual_pow4(dual a) {
    return dual_new(a.val * a.val * a.val * a.val, 4 * a.val * a.val * a.val * a.dv);
}