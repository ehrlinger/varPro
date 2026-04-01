test_that("uvarpro auto mode supports prediction", {
  skip_on_cran()
  data(alzheimers)

  x <- alzheimers[1:200, 1:6]
  set.seed(201)

  u <- uvarpro(
    x,
    method = "auto",
    ntree = 15,
    max.tree = 15,
    max.rules.tree = 8,
    verbose = FALSE
  )

  expect_s3_class(u, "uvarpro")

  pr <- predict(u, x[1:10, , drop = FALSE])
  expect_equal(dim(pr), c(10, 6))
  expect_true(is.numeric(attr(pr, "mse")))
})

test_that("uvarpro rnd mode rejects predict.uvarpro", {
  skip_on_cran()
  data(alzheimers)

  x <- alzheimers[1:160, 1:6]
  set.seed(202)

  u <- uvarpro(
    x,
    method = "rnd",
    ntree = 12,
    max.tree = 12,
    max.rules.tree = 8,
    verbose = FALSE
  )

  expect_error(
    predict(u, x[1:5, , drop = FALSE]),
    "only applies to unsupervised varpro objects using auto-encoder"
  )
})

test_that("isopro unsupervised scoring returns valid ranges", {
  skip_on_cran()
  data(alzheimers)

  x <- alzheimers[1:220, 1:8]
  set.seed(203)

  iso <- isopro(
    data = x,
    method = "unsupv",
    ntree = 30,
    nodesize = 2
  )

  expect_s3_class(iso, "isopro")

  q <- predict(iso, x[1:10, , drop = FALSE], quantiles = TRUE)
  raw <- predict(iso, x[1:10, , drop = FALSE], quantiles = FALSE)

  expect_equal(length(q), 10)
  expect_equal(length(raw), 10)
  expect_true(all(q >= 0 & q <= 1))
})
