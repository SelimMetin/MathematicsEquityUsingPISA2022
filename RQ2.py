import pyreadstat
import pandas as pd
import numpy as np
import os

# =====================================================
# 1. File path
# =====================================================

student_sav_path = "CY08MSP_STU_QQQ.SAV"

if not os.path.exists(student_sav_path):
    raise FileNotFoundError(f"File not found: {student_sav_path}")

# =====================================================
# 2. Read metadata and define variables
# =====================================================

_, meta = pyreadstat.read_sav(student_sav_path, metadataonly=True)
columns = meta.column_names

math_pvs = [f"PV{i}MATH" for i in range(1, 11)]
main_weight = "W_FSTUWT"

rep_weights = [col for col in columns if col.startswith("W_FSTURWT")]
rep_weights = sorted(rep_weights, key=lambda x: int(x.replace("W_FSTURWT", "")))

usecols = ["CNT", "CNTSTUID", "CNTSCHID", "ESCS", main_weight] + math_pvs + rep_weights

missing = [col for col in usecols if col not in columns]
if missing:
    raise ValueError(f"Missing variables: {missing}")

# =====================================================
# 3. Read data
# =====================================================

df, meta = pyreadstat.read_sav(
    student_sav_path,
    usecols=usecols,
    apply_value_formats=False
)

df = df[df["CNT"].isin(["TUR", "GRC"])].copy()
df = df.dropna(subset=["ESCS"])

# Country coding: Greece = 0, Türkiye = 1
df["TURKEY"] = (df["CNT"] == "TUR").astype(int)

# =====================================================
# 3A. Mean-centre socioeconomic status
# =====================================================
# The weighted mean is used so that ESCS = 0 in the regression model
# represents the average socioeconomic status level of the analysed population.

weighted_escs_mean = np.sum(df["ESCS"] * df[main_weight]) / np.sum(df[main_weight])

df["ESCS_CENTERED"] = df["ESCS"] - weighted_escs_mean
df["TURKEY_ESCS_CENTERED"] = df["TURKEY"] * df["ESCS_CENTERED"]

print(df["CNT"].value_counts())
print("Replicate weights:", len(rep_weights))
print("Weighted ESCS mean used for centering:", round(weighted_escs_mean, 4))

# =====================================================
# 4. Helper functions
# =====================================================

def weighted_mean(data, value_col, weight_col):
    valid = data[[value_col, weight_col]].dropna()
    return np.sum(valid[value_col] * valid[weight_col]) / np.sum(valid[weight_col])


def weighted_quantile(values, weights, quantiles):
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cumulative_weight = np.cumsum(weights)
    total_weight = np.sum(weights)
    return np.interp(np.array(quantiles) * total_weight, cumulative_weight, values)


def weighted_ols(y, X, w):
    """
    Weighted least squares using numpy.
    """
    mask = ~np.isnan(y) & ~np.isnan(w) & np.all(~np.isnan(X), axis=1)
    y = y[mask]
    X = X[mask]
    w = w[mask]

    XTWX = X.T @ (w[:, None] * X)
    XTWy = X.T @ (w * y)

    try:
        beta = np.linalg.solve(XTWX, XTWy)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(XTWX, XTWy, rcond=None)[0]

    return beta


def combine_pv(estimates, sampling_variances):
    estimates = np.array(estimates)
    sampling_variances = np.array(sampling_variances)

    M = len(estimates)

    final_estimate = np.mean(estimates)
    sampling_variance = np.mean(sampling_variances)
    imputation_variance = np.sum((estimates - final_estimate) ** 2) / (M - 1)
    total_variance = sampling_variance + (1 + 1/M) * imputation_variance
    se = np.sqrt(total_variance)

    return final_estimate, se, final_estimate - 1.96 * se, final_estimate + 1.96 * se


# =====================================================
# 5. Table 6: Regression-based socioeconomic gradient
# =====================================================

predictor_names = [
    "Intercept",
    "Türkiye",
    "Socioeconomic status, centered",
    "Türkiye × socioeconomic status, centered"
]

pv_betas = []
pv_sampling_vars = []

for pv in math_pvs:
    y = df[pv].to_numpy()

    X = np.column_stack([
        np.ones(len(df)),
        df["TURKEY"].to_numpy(),
        df["ESCS_CENTERED"].to_numpy(),
        df["TURKEY_ESCS_CENTERED"].to_numpy()
    ])

    full_beta = weighted_ols(y, X, df[main_weight].to_numpy())

    replicate_betas = []

    for rw in rep_weights:
        rep_beta = weighted_ols(y, X, df[rw].to_numpy())
        replicate_betas.append(rep_beta)

    replicate_betas = np.array(replicate_betas)

    # PISA Fay-adjusted BRR scaling factor
    sampling_vars = 0.05 * np.sum((replicate_betas - full_beta) ** 2, axis=0)

    pv_betas.append(full_beta)
    pv_sampling_vars.append(sampling_vars)

pv_betas = np.array(pv_betas)
pv_sampling_vars = np.array(pv_sampling_vars)

table6_rows = []

for j, name in enumerate(predictor_names):
    estimate, se, ci_low, ci_high = combine_pv(
        pv_betas[:, j],
        pv_sampling_vars[:, j]
    )

    table6_rows.append({
        "Predictor": name,
        "Estimate": estimate,
        "SE": se,
        "95% CI Lower": ci_low,
        "95% CI Upper": ci_high
    })

table6 = pd.DataFrame(table6_rows)

# Add interpretation
table6["Interpretation"] = [
    "Estimated mathematics score for Greece at the average socioeconomic status level of the analysed sample.",
    "Estimated Türkiye–Greece difference at the average socioeconomic status level of the analysed sample.",
    "Estimated mathematics-score change for a one-unit increase in socioeconomic status in Greece.",
    "Difference in the socioeconomic-status slope between Türkiye and Greece."
]

# Add centering note as a column for transparency
table6["Centering note"] = f"ESCS was centred around the weighted mean of the analysed Türkiye-Greece sample: {weighted_escs_mean:.4f}."

print("\nTable 6: Regression-based socioeconomic gradient with centred socioeconomic status")
print(table6.round(3))

# =====================================================
# 6. Table 7: Low-SES vs High-SES achievement gap
# =====================================================

# Create country-specific weighted quartiles using final student weight.
# This part remains based on the original ESCS scale; centering does not affect quartile grouping.

df["SES_GROUP"] = pd.Series(pd.NA, index=df.index, dtype="object")
for country in ["GRC", "TUR"]:
    sub = df[df["CNT"] == country].copy()

    q25, q75 = weighted_quantile(
        sub["ESCS"].to_numpy(),
        sub[main_weight].to_numpy(),
        [0.25, 0.75]
    )

    df.loc[(df["CNT"] == country) & (df["ESCS"] <= q25), "SES_GROUP"] = "Low SES"
    df.loc[(df["CNT"] == country) & (df["ESCS"] >= q75), "SES_GROUP"] = "High SES"

df_gap = df[df["SES_GROUP"].isin(["Low SES", "High SES"])].copy()


def country_ses_gap(data, country, value_col, weight_col):
    sub = data[data["CNT"] == country]
    low = weighted_mean(sub[sub["SES_GROUP"] == "Low SES"], value_col, weight_col)
    high = weighted_mean(sub[sub["SES_GROUP"] == "High SES"], value_col, weight_col)
    return low, high, high - low


table7_rows = []

for country_code, country_name in [("GRC", "Greece"), ("TUR", "Türkiye")]:
    low_estimates = []
    high_estimates = []
    gap_estimates = []
    gap_sampling_vars = []

    for pv in math_pvs:
        low, high, gap = country_ses_gap(df_gap, country_code, pv, main_weight)

        replicate_gaps = []

        for rw in rep_weights:
            _, _, rep_gap = country_ses_gap(df_gap, country_code, pv, rw)
            replicate_gaps.append(rep_gap)

        replicate_gaps = np.array(replicate_gaps)
        sampling_var = 0.05 * np.sum((replicate_gaps - gap) ** 2)

        low_estimates.append(low)
        high_estimates.append(high)
        gap_estimates.append(gap)
        gap_sampling_vars.append(sampling_var)

    low_mean = np.mean(low_estimates)
    high_mean = np.mean(high_estimates)
    gap_estimate, gap_se, gap_ci_low, gap_ci_high = combine_pv(
        gap_estimates,
        gap_sampling_vars
    )

    table7_rows.append({
        "Country": country_name,
        "Low-SES mean": low_mean,
        "High-SES mean": high_mean,
        "Gap": gap_estimate,
        "SE": gap_se,
        "95% CI Lower": gap_ci_low,
        "95% CI Upper": gap_ci_high
    })

table7 = pd.DataFrame(table7_rows)

print("\nTable 7: Socioeconomic achievement gap")
print(table7.round(3))

# =====================================================
# 7. Save tables
# =====================================================

with pd.ExcelWriter("rq2_socioeconomic_results_centered.xlsx", engine="openpyxl") as writer:
    table6.to_excel(writer, sheet_name="Table6_Regression_Centered", index=False)
    table7.to_excel(writer, sheet_name="Table7_SES_Gap", index=False)

print("\nFile created:")
print("- rq2_socioeconomic_results_centered.xlsx")
