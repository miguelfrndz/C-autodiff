#ifndef AUTO_DIFF_H
#define AUTO_DIFF_H
typedef struct {
    double val;
    double dv;
} dual;

extern dual dual_new(double val, double dv);
extern dual dual_add(dual a, dual b);
extern dual dual_sub(dual a, dual b);
extern dual dual_mul(dual a, dual b);
extern dual dual_div(dual a, dual b);
extern dual dual_sin(dual a);
extern dual dual_cos(dual a);
extern dual dual_tan(dual a);
extern dual dual_exp(dual a);
extern dual dual_log(dual a);
extern dual dual_pow(dual a, double b);
extern dual dual_pow2(dual a);
extern dual dual_pow3(dual a);
extern dual dual_pow4(dual a);
#endif