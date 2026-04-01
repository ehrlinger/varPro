#' Show package news
#'
#' Opens the package NEWS file.
#'
#' @param ... Unused.
#' @return None.
#' @export
varpro.news <- function(...) {
  newsfile <- file.path(system.file(package="varPro"), "NEWS")
  file.show(newsfile)
}
