#' varPro package
#'
#' Model-independent variable selection via rule-based variable priority.
#'
#' @useDynLib varPro, .registration = TRUE
#' @importFrom gbm gbm predict.gbm gbm.perf
#' @importFrom BART wbart
#' @importFrom glmnet glmnet cv.glmnet
#' @importFrom graphics barplot boxplot bxp lines rug axis legend arrows mtext abline par points rect text
#' @importFrom grDevices hcl.colors
#' @importFrom parallel mclapply detectCores
#' @importFrom randomForestSRC rfsrc predict.rfsrc get.mv.formula get.mv.predicted get.auc get.pr.auc
#' @importFrom stats as.formula coef median qnorm pnorm quantile runif var setNames na.omit sd model.matrix prcomp loess loess.control lowess lm supsmu predict.lm rnorm ecdf qt glm t.test chisq.test prop.test binomial logLik pchisq cov hatvalues resid
#' @importFrom survival survdiff Surv
#' @importFrom utils combn capture.output
#' @rawNamespace export(importance.varpro)
#' @rawNamespace export(predict.isopro)
#' @rawNamespace export(predict.ivarpro)
#' @rawNamespace export(predict.uvarpro)
#' @rawNamespace export(predict.varpro)
#' @rawNamespace export(plot.partialpro)
#' @keywords internal
"_PACKAGE"
