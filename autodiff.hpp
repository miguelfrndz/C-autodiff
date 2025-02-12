#ifndef AUTO_DIFF_HPP
#define AUTO_DIFF_HPP

#include <vector>

namespace autodiff {
    // ====================================================
    // Forward-mode autodiff: Dual number class declaration
    // ====================================================

    class dual {
        public:
            double val;
            double dv;
            std::string name;
            
            dual(double val, double dv);
            dual(double val, double dv, const std::string &name);
            dual(double val);
            dual(double val, const std::string &name);

            void print() const;
            friend std::ostream& operator<<(std::ostream &os, const dual &d);

            dual operator+(const dual &other) const;
            dual operator-(const dual &other) const;
            dual operator*(const dual &other) const;
            dual operator/(const dual &other) const;
    };

    dual sin(const dual &x);
    dual cos(const dual &x);
    dual tan(const dual &x);
    dual exp(const dual &x);
    dual log(const dual &x);
    dual pow(const dual &x, double b);
    dual pow2(const dual &x);
    dual pow3(const dual &x);
    dual pow4(const dual &x);

    // ====================================================
    // Reverse-mode autodiff: Var class declaration
    // ====================================================

    class Var;
    using VarPtr = Var*;
    constexpr size_t TAPE_SIZE = 10000;
    using tape_t = std::vector<VarPtr>;
    extern tape_t tape;
}

#endif // AUTO_DIFF_HPP