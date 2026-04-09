# varPro 3.0.0

## Improvements

* Refactored `varpro.strength()` to reduce R-side post-processing overhead after the native `varProStrength` call, improving performance on large forests and large membership reconstructions.
* Improved scalability and stability of `varpro.strength(..., membership = TRUE)` for very large analyses.
* For RHF grow objects, `varpro.strength()` now uses the integrated hazard exposure values stored on the fitted object (`int.haz.oob`) as the default working response when available.
* Internal cleanup of native-output decoding and membership reconstruction logic.

## Bug fixes

* Fixed a failure that could occur on very large analyses when rebuilding membership lists in R after native execution, which could previously surface as an integer-overflow warning from `cumsum()` followed by a downstream missing-value error in membership reconstruction.

# varPro 2.1.0

* Major refactoring and enhancement to functions downstream from the entry `varpro()` function.

# varPro 2.0.0

* Improved `ivarPro`.
* Refactored code to improve speed.
* Eliminated or replaced `mclapply()` with PSOCK-based parallel execution, improving Windows compatibility.

# varPro 1.0.1

* CRAN compliance fixes.

# varPro 1.0.0

* Initial release.
