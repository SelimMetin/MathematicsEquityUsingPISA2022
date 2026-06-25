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

print("Reading:", student_sav_path)


# =====================================================
# 2. Read metadata and identify variables
# =====================================================

_, meta = pyreadstat.read_sav(student_sav_path, metadataonly=True)
columns = meta.column_names

math_pvs = [f"PV{i}MATH" for i in range(1, 11)]

main_weight = "W_FSTUWT"

# Automatically detect replicate weights
rep_weights = [col for col in columns if col.startswith("W_FSTURWT")]

# Sort replicate weights numerically
rep_weights = sorted(
    rep_weights,
    key=lambda x: int(x.replace("W_FSTURWT", ""))
)

print("Number of replicate weights found:", len(rep_weights))
print("First replicate weights:", rep_weights[:5])
print("Last replicate weights:", rep_weights[-5:])

required_cols = [
    "CNT",
    "CNTSTUID",
    "CNTSCHID",
    main_weight
] + math_pvs + rep_weights

missing = [col for col in required_cols if col not in columns]

if missing:
    raise ValueError(f"Missing required variables: {missing}")

print("All required variables were found.")


# =====================================================
# 3. Read only required columns
# =====================================================

df, meta = pyreadstat.read_sav(
    student_sav_path,
    usecols=required_cols,
    apply_value_formats=False
)

# Keep Türkiye and Greece only
df = df[df["CNT"].isin(["TUR", "GRC"])].copy()

print("\nSample sizes:")
print(df["CNT"].value_counts())


# =====================================================
# 4. Helper functions
# =====================================================

def weighted_mean(data, value_col, weight_col):
    valid = data[[value_col, weight_col]].dropna()
    return np.sum(valid[value_col] * valid[weight_col]) / np.sum(valid[weight_col])


def country_mean(data, country, value_col, weight_col):
    sub = data[data["CNT"] == country]
    return weighted_mean(sub, value_col, weight_col)


def country_difference(data, value_col, weight_col):
    """
    Difference is Türkiye minus Greece.
    """
    mean_tur = country_mean(data, "TUR", value_col, weight_col)
    mean_grc = country_mean(data, "GRC", value_col, weight_col)
    return mean_tur - mean_grc


def replicate_sampling_variance(data, value_col, estimate_type, country=None):
    """
    Estimate sampling variance using 80 PISA replicate weights.

    estimate_type:
    - "mean": country-level mean
    - "difference": Türkiye minus Greece
    """

    if estimate_type == "mean":
        full_estimate = country_mean(data, country, value_col, main_weight)
    elif estimate_type == "difference":
        full_estimate = country_difference(data, value_col, main_weight)
    else:
        raise ValueError("estimate_type must be 'mean' or 'difference'.")

    replicate_estimates = []

    for rw in rep_weights:
        if estimate_type == "mean":
            rep_est = country_mean(data, country, value_col, rw)
        else:
            rep_est = country_difference(data, value_col, rw)

        replicate_estimates.append(rep_est)

    replicate_estimates = np.array(replicate_estimates)

    # PISA uses 80 replicate weights with Fay-type replication.
    # The standard PISA multiplier for 80 replicate weights is 0.05.
    sampling_variance = 0.05 * np.sum((replicate_estimates - full_estimate) ** 2)

    return sampling_variance


def combine_pv_estimates(estimates, sampling_variances):
    """
    Combine estimates across plausible values.

    estimates: list of estimates from PV1...PV10
    sampling_variances: list of sampling variances from PV1...PV10
    """

    estimates = np.array(estimates)
    sampling_variances = np.array(sampling_variances)

    m = len(estimates)

    final_estimate = np.mean(estimates)

    # Within-imputation / sampling variance
    within_variance = np.mean(sampling_variances)

    # Between-imputation variance
    between_variance = np.sum((estimates - final_estimate) ** 2) / (m - 1)

    # Total variance
    total_variance = within_variance + (1 + 1 / m) * between_variance

    standard_error = np.sqrt(total_variance)

    ci_lower = final_estimate - 1.96 * standard_error
    ci_upper = final_estimate + 1.96 * standard_error

    return {
        "estimate": final_estimate,
        "sampling_variance": within_variance,
        "imputation_variance": between_variance,
        "total_variance": total_variance,
        "se": standard_error,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }


# =====================================================
# 5. Primary PISA-style analysis for RQ1
# =====================================================

results_by_pv = []

for pv in math_pvs:
    tur_mean = country_mean(df, "TUR", pv, main_weight)
    grc_mean = country_mean(df, "GRC", pv, main_weight)
    diff = tur_mean - grc_mean

    tur_samp_var = replicate_sampling_variance(
        df, pv, estimate_type="mean", country="TUR"
    )

    grc_samp_var = replicate_sampling_variance(
        df, pv, estimate_type="mean", country="GRC"
    )

    diff_samp_var = replicate_sampling_variance(
        df, pv, estimate_type="difference"
    )

    results_by_pv.append({
        "plausible_value": pv,
        "turkiye_mean": tur_mean,
        "greece_mean": grc_mean,
        "difference_turkiye_minus_greece": diff,
        "turkiye_sampling_variance": tur_samp_var,
        "greece_sampling_variance": grc_samp_var,
        "difference_sampling_variance": diff_samp_var
    })

pv_results_df = pd.DataFrame(results_by_pv)

# Combine across plausible values
turkiye_final = combine_pv_estimates(
    pv_results_df["turkiye_mean"],
    pv_results_df["turkiye_sampling_variance"]
)

greece_final = combine_pv_estimates(
    pv_results_df["greece_mean"],
    pv_results_df["greece_sampling_variance"]
)

difference_final = combine_pv_estimates(
    pv_results_df["difference_turkiye_minus_greece"],
    pv_results_df["difference_sampling_variance"]
)


# =====================================================
# 6. Create final RQ1 result tables
# =====================================================

country_results = pd.DataFrame([
    {
        "Country": "Türkiye",
        "Estimate": turkiye_final["estimate"],
        "SE": turkiye_final["se"],
        "95% CI Lower": turkiye_final["ci_lower"],
        "95% CI Upper": turkiye_final["ci_upper"]
    },
    {
        "Country": "Greece",
        "Estimate": greece_final["estimate"],
        "SE": greece_final["se"],
        "95% CI Lower": greece_final["ci_lower"],
        "95% CI Upper": greece_final["ci_upper"]
    }
])

difference_results = pd.DataFrame([
    {
        "Comparison": "Türkiye - Greece",
        "Difference": difference_final["estimate"],
        "SE": difference_final["se"],
        "95% CI Lower": difference_final["ci_lower"],
        "95% CI Upper": difference_final["ci_upper"]
    }
])

print("\nCountry-level PISA-style estimates:")
print(country_results.round(3))

print("\nDifference estimate:")
print(difference_results.round(3))


# =====================================================
# 7. Robustness checks
# =====================================================

# Create averaged mathematics score for robustness check only
df["MATH_AVG"] = df[math_pvs].mean(axis=1)


def replicate_se_single_score(data, value_col, estimate_type, country=None):
    if estimate_type == "mean":
        full_estimate = country_mean(data, country, value_col, main_weight)
    elif estimate_type == "difference":
        full_estimate = country_difference(data, value_col, main_weight)
    else:
        raise ValueError("estimate_type must be 'mean' or 'difference'.")

    replicate_estimates = []

    for rw in rep_weights:
        if estimate_type == "mean":
            rep_est = country_mean(data, country, value_col, rw)
        else:
            rep_est = country_difference(data, value_col, rw)

        replicate_estimates.append(rep_est)

    replicate_estimates = np.array(replicate_estimates)

    variance = 0.05 * np.sum((replicate_estimates - full_estimate) ** 2)
    se = np.sqrt(variance)

    return full_estimate, se


# Weighted averaged mathematics score
tur_avg, tur_avg_se = replicate_se_single_score(df, "MATH_AVG", "mean", "TUR")
grc_avg, grc_avg_se = replicate_se_single_score(df, "MATH_AVG", "mean", "GRC")
diff_avg, diff_avg_se = replicate_se_single_score(df, "MATH_AVG", "difference")

# Weighted first plausible value only
tur_pv1, tur_pv1_se = replicate_se_single_score(df, "PV1MATH", "mean", "TUR")
grc_pv1, grc_pv1_se = replicate_se_single_score(df, "PV1MATH", "mean", "GRC")
diff_pv1, diff_pv1_se = replicate_se_single_score(df, "PV1MATH", "difference")

# Unweighted averaged mathematics score
tur_unw = df[df["CNT"] == "TUR"]["MATH_AVG"].mean()
grc_unw = df[df["CNT"] == "GRC"]["MATH_AVG"].mean()
diff_unw = tur_unw - grc_unw

tur_unw_se = df[df["CNT"] == "TUR"]["MATH_AVG"].std(ddof=1) / np.sqrt(df[df["CNT"] == "TUR"]["MATH_AVG"].notna().sum())
grc_unw_se = df[df["CNT"] == "GRC"]["MATH_AVG"].std(ddof=1) / np.sqrt(df[df["CNT"] == "GRC"]["MATH_AVG"].notna().sum())
diff_unw_se = np.sqrt(tur_unw_se**2 + grc_unw_se**2)

robustness = pd.DataFrame([
    {
        "Method": "Plausible-value pooling with student weights and replicate weights",
        "Türkiye mean": turkiye_final["estimate"],
        "Greece mean": greece_final["estimate"],
        "Difference": difference_final["estimate"],
        "SE difference": difference_final["se"],
        "95% CI Lower": difference_final["ci_lower"],
        "95% CI Upper": difference_final["ci_upper"],
        "Role": "Primary estimate"
    },
    {
        "Method": "Weighted averaged mathematics score with replicate weights",
        "Türkiye mean": tur_avg,
        "Greece mean": grc_avg,
        "Difference": diff_avg,
        "SE difference": diff_avg_se,
        "95% CI Lower": diff_avg - 1.96 * diff_avg_se,
        "95% CI Upper": diff_avg + 1.96 * diff_avg_se,
        "Role": "Robustness check"
    },
    {
        "Method": "Weighted first plausible value only with replicate weights",
        "Türkiye mean": tur_pv1,
        "Greece mean": grc_pv1,
        "Difference": diff_pv1,
        "SE difference": diff_pv1_se,
        "95% CI Lower": diff_pv1 - 1.96 * diff_pv1_se,
        "95% CI Upper": diff_pv1 + 1.96 * diff_pv1_se,
        "Role": "Robustness check"
    },
    {
        "Method": "Unweighted averaged mathematics score",
        "Türkiye mean": tur_unw,
        "Greece mean": grc_unw,
        "Difference": diff_unw,
        "SE difference": diff_unw_se,
        "95% CI Lower": diff_unw - 1.96 * diff_unw_se,
        "95% CI Upper": diff_unw + 1.96 * diff_unw_se,
        "Role": "Sensitivity check"
    }
])

print("\nRobustness table:")
print(robustness.round(3))


# =====================================================
# 8. Save outputs
# =====================================================

with pd.ExcelWriter("rq1_pisa_design_consistent_results.xlsx", engine="openpyxl") as writer:
    country_results.to_excel(writer, sheet_name="Country_Estimates", index=False)
    difference_results.to_excel(writer, sheet_name="Difference", index=False)
    pv_results_df.to_excel(writer, sheet_name="PV_Level_Results", index=False)
    robustness.to_excel(writer, sheet_name="Robustness_Checks", index=False)

print("\nFile created:")
print("- rq1_pisa_design_consistent_results.xlsx")