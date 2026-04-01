# varPro

Model-independent variable selection via rule-based variable priority.

<!-- badges: start -->

[![CRAN status](https://www.r-pkg.org/badges/version/varpro)](https://cran.r-project.org/package=varpro)
[![Version](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kogalur/varPro/main/badges/version.json)](https://github.com/kogalur/varPro/blob/main/DESCRIPTION)
[![Downloads](https://cranlogs.r-pkg.org/badges/varpro)](https://cranlogs.r-pkg.org/badges/varpro)
[![R-CMD-check](https://github.com/kogalur/varPro/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/kogalur/varPro/actions/workflows/R-CMD-check.yaml)
[![codecov](https://codecov.io/gh/kogalur/varPro/graph/badge.svg)](https://codecov.io/gh/kogalur/varPro)
[![Project status: active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/badges/latest/active.svg)

<!-- badges: end -->

## Why varPro

`varPro` is a variable-selection framework designed for settings where
the data-generating process is complex or unknown. Instead of relying on
artificial covariates (such as permutations or knockoffs), it uses release
rules and rule-based statistics to estimate each feature's explanatory signal.

The workflow targets practical interpretation: identify a smaller set of
important variables while preserving strong predictive behavior.

Supported analysis families include:

- Regression
- Multivariate regression
- Classification
- Survival analysis

Current repository release: 2.1.0 (2026-02-12).

## Installation

Install the CRAN release:

```r
install.packages("varPro")
```

Install the development version from GitHub:

```r
install.packages("devtools")
devtools::install_github("kogalur/randomForestSRC")
devtools::install_github("kogalur/varPro")
```

## Quick Start

```r
library(varPro)

data(alzheimers)

fit <- varpro(
	formula = status ~ ., 
	data = alzheimers,
	nvar = 20,
	ntree = 300,
	seed = 42
)

print(fit)
importance.varpro(fit)
```

See package help pages for additional workflows:

- `?ivarpro` for iterative selection
- `?uvarpro` for unified modeling options
- `?partialpro` for partial dependence style summaries

## Documentation Workflow (roxygen2)

This repository now supports a roxygen2-based documentation process. Use these
commands during development:

```r
# Rebuild NAMESPACE and man/*.Rd from roxygen comments in R/*.R
devtools::document()

# Rebuild README.md from README.Rmd (if edited)
devtools::build_readme()
```

Suggested authoring pattern:

1. Update function-level roxygen comments in `R/` files.
2. Run `devtools::document()`.
3. Review generated updates in `NAMESPACE` and `man/`.
4. Run package checks before commit.

## Reference

Lu M., Ishwaran H. Model-independent variable selection via the rule-based
variable priority. arXiv:2409.09003. https://arxiv.org/abs/2409.09003
