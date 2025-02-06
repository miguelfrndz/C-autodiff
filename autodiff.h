#ifndef AUTO_DIFF_H
#define AUTO_DIFF_H

// Dual number struct (for forward-mode autodiff)
typedef struct {
    double val;
    double dv;
} dual;

// Function prototypes for forward-mode autodiff
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

#define TAPE_SIZE 1000 // tape size (for reverse mode)

typedef struct Var Var;
Var *tape[TAPE_SIZE];
int tape_index;

typedef struct Var {
    double value;           // value of this node
    double grad;            // accumulated adjoint gradient
    Var *left;              // pointer to left operand (or only operand for unary ops)
    Var *right;             // pointer to right operand (if any)
    void (*backward)(Var*); // function pointer for backward pass
    char name[];            // name of this node (for debugging)
} Var;

// Function prototypes for reverse-mode autodiff
extern Var* var_new(double value);
extern void var_print(Var *v);
extern void backward_pass(Var *v);
extern void init_tape();
extern void free_tape();
extern void var_add_backward(Var *v);
extern Var* var_add(Var *a, Var *b);
extern void var_sub_backward(Var *v);
extern Var* var_sub(Var *a, Var *b);
extern void var_mul_backward(Var *v);
extern Var* var_mul(Var *a, Var *b);
extern void var_div_backward(Var *v);
extern Var* var_div(Var *a, Var *b);
extern void var_sin_backward(Var *v);
extern Var* var_sin(Var *a);
extern void var_cos_backward(Var *v);
extern Var* var_cos(Var *a);
extern void var_tan_backward(Var *v);
extern Var* var_tan(Var *a);
extern void var_exp_backward(Var *v);
extern Var* var_exp(Var *a);
extern void var_log_backward(Var *v);
extern Var* var_log(Var *a);
extern void var_pow_backward(Var *v);
extern Var* var_pow(Var *a, Var *b);
extern void var_pow2_backward(Var *v);
extern Var* var_pow2(Var *a);
extern void var_pow3_backward(Var *v);
extern Var* var_pow3(Var *a);
extern void var_pow4_backward(Var *v);
extern Var* var_pow4(Var *a);

#endif