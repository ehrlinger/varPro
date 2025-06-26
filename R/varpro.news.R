#' Show the NEWS file
#' 
#' Show the NEWS file of the \pkg{varPro} package.
#' 
#' 
#' @param ... Further arguments passed to or from other methods.
#' @return None.
#' @author Min Lu and Hemant Ishwaran
#' @keywords documentation
varpro.news <- function(...) {
  newsfile <- file.path(system.file(package="varPro"), "NEWS")
  file.show(newsfile)
}
