test_that("ivarpro path and predict produce structured output", {
  skip_on_cran()
  data(alzheimers)

  d <- alzheimers[1:180, ]
  set.seed(301)

  o <- varpro(
    Diagnosis ~ .,
    d,
    ntree = 12,
    nvar = 5,
    max.tree = 12,
    max.rules.tree = 10,
    split.weight = FALSE,
    parallel = FALSE,
    verbose = FALSE
  )

  iv <- ivarpro(
    o,
    ncut = 7,
    nmin = 8,
    nmax = 16,
    save.model = TRUE,
    save.data = TRUE
  )

  expect_s3_class(iv, "ivarpro")

  pr <- predict(iv)
  expect_s3_class(pr, "data.frame")
  expect_equal(nrow(pr), nrow(d))
})
