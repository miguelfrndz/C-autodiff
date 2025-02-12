# Auto-Differentiation Engine in C/C++
Basic implementation of a simple auto-differentiation engine in C/C++. Includes forward-mode (using the `dual` type) and reverse-mode (using the `Var` type).

## Example: Normal Log-Likelihood

As a practical example to test our code, let us consider the log-likelihood function of a Normal (Gaussian) distribution. For an observed variable $x$, the log-likelihood is given by:

$$
f(x,\mu, \sigma)=-\dfrac{1}{2}\left(\dfrac{x-\mu}{\sigma}\right)^2-\log(\sigma)-\dfrac{1}{2}\log(2\pi)
$$

where $\mu$ and $\sigma$ are the mean and standard deviation, respectively. Our goal is to compute the derivatives at the point $(x=10, \mu=5, \sigma=2)$.

### Forward-Mode (`dual`) in `C`

1) Define the log-likelihood function in terms of elementary operations:
```C
dual log_norm_likelihood(dual x, dual mu, dual sigma) {
    dual a = dual_div(dual_sub(x, mu), sigma);
    dual b = dual_pow2(a);
    dual c = dual_mul(b, dual_new(-0.5, 0.0));
    dual d = dual_sub(c, dual_log(sigma));
    dual e = dual_sub(d, dual_new(0.5 * log(2 * M_PI), 0.0));
    return e;
}
```
2) Initialize the variables of our function (note that in forward mode we set to 1 the directional derivative of the value of interest, in this case $\mu$ as we wish to compute $\frac{\partial f}{\partial \mu}$):
```C
dual x = dual_new(10.0, 0.0);
dual mu = dual_new(5.0, 1.0); // Variable of interest
dual sigma = dual_new(2.0, 0.0);
```
3) Run the *forward-mode* auto-differentiation:
```C
dual f = log_norm_likelihood(x, mu, sigma);
```
4) `f.val` contains the evaluation of the function at the point of interest, and `f.dv` contains $\frac{\partial f}{\partial \mu}$.

### Forward-Mode (`dual`) in `C++`

1) Define the log-likelihood function in terms of elementary operations (note that in this case, the usual operators have been overloaded for easier function definition):
```C++
dual log_normal_likelihood(dual x, dual mu, dual sigma) {
    dual a = (x - mu) / sigma;
    dual b = pow2(a);
    dual c = b * dual(-0.5); // We don't need to specify null gradients :)
    dual d = c - log(sigma);
    dual e = d - dual(0.5 * log(2 * M_PI));
    return e;
}
```
2) Initialize the variables of our function (note that in forward mode we set to 1 the directional derivative of the value of interest, in this case $\mu$ as we wish to compute $\frac{\partial f}{\partial \mu}$):
```C++
dual x = dual(10.0, "x"); // We don't need to include the gradient for the other variables :)
dual mu = dual(5.0, 1.0, "mu"); // Variable of interest
dual sigma = dual(2.0, "sigma");
```
3) Run the *forward-mode* auto-differentiation:
```C++
dual f = log_normal_likelihood(x, mu, sigma);
f.name = "log_normal_likelihood"; // Of course we can still name the dual :)
```
4) `f.val` contains the evaluation of the function at the point of interest, and `f.dv` contains $\frac{\partial f}{\partial \mu}$.

### Reverse-Mode (`Val`) in `C`
1) We initialize the gradient tape:
```C
init_tape();
```
2) Similarly to the forward-mode, we define the log-likelihood in terms of its elementary operations:
```C
Var* log_norm_likelihood_rev(Var *x, Var *mu, Var *sigma) {
    Var *a = var_div(var_sub(x, mu), sigma);
    Var *b = var_pow2(a);
    Var *c = var_mul(b, var_new(-0.5));
    Var *d = var_sub(c, var_log(sigma));
    Var *e = var_sub(d, var_new(0.5 * log(2 * M_PI)));
    return e;
}
```
3) Initialize the variables of our interest (optionally, we can name them):
```C
Var *x = var_new(10.0);
strcpy(x->name, "x");
Var *mu = var_new(5.0);
strcpy(mu->name, "mu");
Var *sigma = var_new(2.0);
strcpy(sigma->name, "sigma");
```
4) Run the *reverse-mode* auto-differentiation:
```C
Var *f = log_norm_likelihood_rev(x, mu, sigma);
backward_pass(f); //Execute backwards pass
```
5) The derivatives w.r.t. each variable are now available under the variables themselves. For example, $\frac{\partial f}{\partial \mu}$ is in:
```C
mu->grad;
```

6) Clear the tape and release the nodes in the expression graph:
```C
free_tape();
```

### Reverse-Mode (`Val`) in `C++`
Not yet implemented.

## How-to-Run?

- To build the `C` version:
```bash
make
```
- To build the `C++` version:
```bash
make cpp
```

## TODO: Pending Tasks
* Implement a parser to generate the expression graph based on any function definition (to overcome lack of operator overloading in C).
* Make a Python interface to deploy the library.
