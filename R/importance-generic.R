#' Importance generic
#'
#' Generic for importance methods.
#'
#' @param object Object to evaluate.
#' @param ... Additional arguments.
#' @return Method-specific importance output.
#' @export
importance <- function(object, ...) {
  UseMethod("importance")
}
