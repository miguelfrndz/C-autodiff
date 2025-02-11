#ifndef AUTO_DIFF_HPP
#define AUTO_DIFF_HPP

namespace autodiff {
    // ====================================================
    // Forward-mode autodiff: Dual number class declaration
    // ====================================================

    /*
    Dual Variable for Forward-Mode Autodiff.
    */
    class dual {
        public:
            double val;
            double dv;
            std::string name;
            
            /*
            Constructor for dual variable.

            @param val Value of the dual variable.
            @param dv Derivative of the dual variable.
            */
            dual(double val, double dv);

            /*
            Constructor for dual variable with name.

            @param val Value of the dual variable.
            @param dv Derivative of the dual variable.
            @param name Name of the dual variable.
            */
            dual(double val, double dv, const std::string &name);

            dual (double val);

            dual (double val, const std::string &name);

            void print() const;

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
}

#endif // AUTO_DIFF_HPP