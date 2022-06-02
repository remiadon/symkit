import polars as pl


def pl_r2_score(y_preds: pl.DataFrame, y_true: pl.Series):
    diff = (
        y_preds - y_true
    )  # y_true - y_preds not possible in polars, but result is squarred
    y_minus_X_2 = diff.select([pl.Expr.pow(pl.col(col), 2) for col in diff.columns])
    sserr = y_minus_X_2.sum()
    y_minus_mean_2 = (y_true - y_true.mean()) ** 2
    sstot = y_minus_mean_2.sum()
    ratio = sserr / sstot
    R2 = ratio.select(
        [(pl.lit(1.0) - pl.col(col)).alias(str(col)) for col in ratio.columns]
    )
    return R2
