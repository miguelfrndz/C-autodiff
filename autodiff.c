#include <math.h>
#include <stdio.h>
#include <stdlib.h>
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

void init_tape() {
    tape_index = 0;
}

void free_tape() {
    for (int i = 0; i < tape_index; i++) {
        free(tape[i]);
    }
}

Var* var_new(double value) {
    Var *v = (Var*) malloc(sizeof(Var));
    if (!v) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    v->value = value;
    v->grad = 0.0;
    v->left = NULL;
    v->right = NULL;
    v->backward = NULL;
    if (tape_index >= TAPE_SIZE) {
        fprintf(stderr, "Tape overflow!\n");
        exit(1);
    }
    tape[tape_index++] = v;
    return v;
}

void var_print(Var *v) {
    if (v->name[0] != '\0') {
        printf("\tName: %s, Value: %f, Gradient: %f\n", v->name, v->value, v->grad);
    } else {
        printf("\tValue: %f, Gradient: %f\n", v->value, v->grad);
    }
}

void backward_pass(Var *v) {
    v->grad = 1.0;
    for (int i = tape_index - 1; i >= 0; i--) {
        if (tape[i]->backward) {
            tape[i]->backward(tape[i]);
        }
    }
}

void var_add_backward(Var *v) {
    if (v->left)  v->left->grad += v->grad;
    if (v->right) v->right->grad += v->grad;
}

Var* var_add(Var *a, Var *b) {
    Var *v = var_new(a->value + b->value);
    v->left = a;
    v->right = b;
    v->backward = var_add_backward;
    return v;
}

void var_sub_backward(Var *v) {
    if (v->left)  v->left->grad += v->grad;
    if (v->right) v->right->grad -= v->grad;
}

Var* var_sub(Var *a, Var *b) {
    Var *v = var_new(a->value - b->value);
    v->left = a;
    v->right = b;
    v->backward = var_sub_backward;
    return v;
}

void var_mul_backward(Var *v) {
    if (v->left)  v->left->grad += v->grad * v->right->value;
    if (v->right) v->right->grad += v->grad * v->left->value;
}

Var* var_mul(Var *a, Var *b) {
    Var *v = var_new(a->value * b->value);
    v->left = a;
    v->right = b;
    v->backward = var_mul_backward;
    return v;
}

void var_div_backward(Var *v) {
    if (v->left)  v->left->grad += v->grad / v->right->value;
    if (v->right) v->right->grad -= v->grad * v->left->value / (v->right->value * v->right->value);
}

Var* var_div(Var *a, Var *b) {
    Var *v = var_new(a->value / b->value);
    v->left = a;
    v->right = b;
    v->backward = var_div_backward;
    return v;
}

void var_sin_backward(Var *v) {
    if (v->left) v->left->grad += v->grad * cos(v->left->value);
}

Var* var_sin(Var *a) {
    Var *v = var_new(sin(a->value));
    v->left = a;
    v->backward = var_sin_backward;
    return v;
}

void var_cos_backward(Var *v) {
    if (v->left) v->left->grad -= v->grad * sin(v->left->value);
}

Var* var_cos(Var *a) {
    Var *v = var_new(cos(a->value));
    v->left = a;
    v->backward = var_cos_backward;
    return v;
}

void var_tan_backward(Var *v) {
    if (v->left) v->left->grad += v->grad / (cos(v->left->value) * cos(v->left->value));
}

Var* var_tan(Var *a) {
    Var *v = var_new(tan(a->value));
    v->left = a;
    v->backward = var_tan_backward;
    return v;
}

void var_exp_backward(Var *v) {
    if (v->left) v->left->grad += v->grad * exp(v->left->value);
}

Var* var_exp(Var *a) {
    Var *v = var_new(exp(a->value));
    v->left = a;
    v->backward = var_exp_backward;
    return v;
}

void var_log_backward(Var *v) {
    if (v->left) v->left->grad += v->grad / v->left->value;
}

Var* var_log(Var *a) {
    Var *v = var_new(log(a->value));
    v->left = a;
    v->backward = var_log_backward;
    return v;
}

void var_pow_backward(Var *v) {
    if (v->left) v->left->grad += v->grad * v->right->value * pow(v->left->value, v->right->value - 1);
    if (v->right) v->right->grad += v->grad * pow(v->left->value, v->right->value) * log(v->left->value);
}

Var* var_pow(Var *a, Var *b) {
    Var *v = var_new(pow(a->value, b->value));
    v->left = a;
    v->right = b;
    v->backward = var_pow_backward;
    return v;
}

void var_pow2_backward(Var *v) {
    if (v->left) v->left->grad += v->grad * 2 * v->left->value;
}

Var* var_pow2(Var *a) {
    Var *v = var_new(a->value * a->value);
    v->left = a;
    v->backward = var_pow2_backward;
    return v;
}

void var_pow3_backward(Var *v) {
    if (v->left) v->left->grad += v->grad * 3 * v->left->value * v->left->value;
}

Var* var_pow3(Var *a) {
    Var *v = var_new(a->value * a->value * a->value);
    v->left = a;
    v->backward = var_pow3_backward;
    return v;
}

void var_pow4_backward(Var *v) {
    if (v->left) v->left->grad += v->grad * 4 * v->left->value * v->left->value * v->left->value;
}

Var* var_pow4(Var *a) {
    Var *v = var_new(a->value * a->value * a->value * a->value);
    v->left = a;
    v->backward = var_pow4_backward;
    return v;
}