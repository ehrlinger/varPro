varpro.news <- function(...) {
  newsfile <- file.path(system.file(package="varPro"), "NEWS.md")
  file.show(newsfile)
}
