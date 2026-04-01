# varPro

Model-independent variable selection via rule-based variable priority.

<!-- badges: start -->

[![cranlogs](https://cranlogs.r-pkg.org/badges/varpro)](https://cranlogs.r-pkg.org/badges/varpro)
[![CRAN status](https://www.r-pkg.org/badges/version/varpro)](https://cran.r-project.org/package=varpro)
[![active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/badges/latest/active.svg)

<!-- badges: end -->

`varPro` provides a framework for variable selection that avoids creating
artificial covariates (for example permutation variables or knockoffs).
Instead, it constructs release rules to evaluate each covariate's influence
on response behavior under unknown, potentially complex conditional
distributions.

While prediction accuracy is a central machine-learning objective, identifying
small sets of covariates with strong explanatory power is equally important.
Variable Priority (VarPro) uses rule-based statistics and sample averages,
making it broadly applicable across regression, classification, and survival
settings.

## Current Release

Current package release in this repository: 2.1.0 (2026-02-12).

## Installation

Install from GitHub:

```r
install.packages("devtools") # if needed

# randomForestSRC is a required dependency
devtools::install_github("kogalur/randomForestSRC")
devtools::install_github("kogalur/varPro")
```

## Reference

Lu M., Ishwaran H. Model-independent variable selection via the rule-based
variable priority. arXiv:2409.09003.
https://arxiv.org/abs/2409.09003
