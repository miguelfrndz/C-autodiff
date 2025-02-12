#include <string>
#include <iostream>
#include <cmath>
#include "autodiff.hpp"

namespace autodiff {

    // ====================================================
    // Forward-mode autodiff: Dual number class definition
    // ====================================================

    dual::dual(double val, double dv) {
        this->val = val;
        this->dv = dv;
        this->name = "";
    }

    dual::dual(double val, double dv, const std::string &name) {
        this->val = val;
        this->dv = dv;
        this->name = name;
    }

    dual::dual(double val) {
        this->val = val;
        this->dv = 0.0;
        this->name = "";
    }

    dual::dual(double val, const std::string &name) {
        this->val = val;
        this->dv = 0.0;
        this->name = name;
    }

    void dual::print() const {
        if (name.empty()) {
            std::cout << "Name: (No Name Available)" << ", Value: " << val << ", Gradient: " << dv << std::endl;
        } else {
            std::cout << "Name: " << name << ", Value: " << val << ", Gradient: " << dv << std::endl;
        }
    }

    std::ostream& operator<<(std::ostream &os, const dual &d) {
        if (d.name.empty()) {
            os << "Name: (No Name Available)" << ", Value: " << d.val << ", Gradient: " << d.dv;
        } else {
            os << "Name: " << d.name << ", Value: " << d.val << ", Gradient: " << d.dv;
        }
        os << std::endl;
        return os;
    }

    dual dual::operator+(const dual &other) const {
        return dual(val + other.val, dv + other.dv);
    };

    dual dual::operator-(const dual &other) const {
        return dual(val - other.val, dv - other.dv);
    }

    dual dual::operator*(const dual &other) const {
        return dual(val * other.val, val * other.dv + dv * other.val);
    }

    dual dual::operator/(const dual &other) const {
        return dual(val / other.val, (dv * other.val - val * other.dv) / (other.val * other.val));
    }

    dual sin(const dual &x) {
        return dual(std::sin(x.val), x.dv * std::cos(x.val));
    }

    dual cos(const dual &x) {
        return dual(std::cos(x.val), -x.dv * std::sin(x.val));
    }

    dual tan(const dual &x) {
        return dual(std::tan(x.val), x.dv / (std::cos(x.val) * std::cos(x.val)));
    }

    dual exp(const dual &x) {
        return dual(std::exp(x.val), x.dv * std::exp(x.val));
    }

    dual log(const dual &x) {
        return dual(std::log(x.val), x.dv / x.val);
    }

    dual pow(const dual &x, double b) {
        return dual(std::pow(x.val, b), b * std::pow(x.val, b - 1) * x.dv);
    }

    dual pow2(const dual &x) {
        return dual(x.val * x.val, 2 * x.val * x.dv);
    }

    dual pow3(const dual &x) {
        return dual(x.val * x.val * x.val, 3 * x.val * x.val * x.dv);
    }

    dual pow4(const dual &x) {
        return dual(x.val * x.val * x.val * x.val, 4 * x.val * x.val * x.val * x.dv);
    }

    // ====================================================
    // Reverse-mode autodiff: Var class definition
    // ====================================================

    tape_t tape;

    // TODO: Not yet implemented...
}