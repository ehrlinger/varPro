test_that("importance validates object class", {
  expect_error(
    importance(list()),
    "no applicable method for 'importance'"
  )
})

test_that("predict methods validate object classes", {
  expect_error(
    predict.varpro(list(), newdata = data.frame(x = 1)),
    "object must be a varpro object"
  )

  expect_error(
    predict.uvarpro(list(), newdata = data.frame(x = 1)),
    "object must be an 'uvarpro' varpro object"
  )

  expect_error(
    predict.isopro(list(), newdata = data.frame(x = 1)),
    "object must be an 'isopro' varpro object"
  )
})

test_that("outpro validates object classes", {
  expect_error(
    outpro(list(), data.frame(x = 1)),
    "requires a 'varpro' object or an 'rfsrc' object"
  )
})

test_that("isopro validates missing object/formula/data combinations", {
  expect_error(
    isopro(list(), method = "unsupv"),
    "object must be a varpro object"
  )
})

test_that("ivarpro validates unsupported class", {
  expect_error(
    ivarpro(list()),
    "only works for objects of class 'varpro' or an 'rfsrc' grow object"
  )
})

test_that("get.iso.performance enforces binary response", {
  expect_error(
    get.iso.performance(y = c(0, 1, 2), p = c(0.2, 0.3, 0.4)),
    "y can only take 2 distinct values"
  )
})
