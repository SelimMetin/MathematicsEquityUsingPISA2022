
"""
RQ3 school-context analysis for PISA 2022 Turkiye-Greece study.

This script estimates adjusted weighted regression models for RQ3:
To what extent are school-level socioeconomic disadvantage and migration-related
composition associated with mathematics performance in Turkiye and Greece?

Input files expected in the same folder:
- CY08MSP_STU_QQQ.SAV
- CY08MSP_SCH_QQQ.SAV

Output:
- rq3_school_context_results.xlsx
"""

import os
import numpy as np
import pandas as pd
import pyreadstat


# =====================================================
# 1. File paths
# =====================================================

student_sav_path = "CY08MSP_STU_QQQ.SAV"
school_sav_path = "CY08MSP_SCH_QQQ.SAV"

if not os.path.exists(student_sav_path):
    raise FileNotFoundError(f"Student file not found: {student_sav_path}")

if not os.path.exists(school_sav_path):
    raise FileNotFoundError(f"School file not found: {school_sav_path}")


# =====================================================
# 2. Read metadata and define variables
# =====================================================

_, stu_meta = pyreadstat.read_sav(student_sav_path, metadataonly=True)
stu_columns = stu_meta.column_names

math_pvs = [f"PV{i}MATH" for i in range(1, 11)]
main_weight = "W_FSTUWT"

# Detect PISA student replicate weights
rep_weights = [col for col in stu_columns if col.startswith("W_FSTURWT")]
rep_weights = sorted(rep_weights, key=lambda x: int(x.replace("W_FSTURWT", "")))

student_cols = [
    "CNT",
    "CNTSTUID",
    "CNTSCHID",
    "ESCS",
    main_weight,
] + math_pvs + rep_weights

school_cols = [
    "CNT",
    "CNTSCHID",
    "SC211Q01JA",  # percentage of students whose heritage language differs from the test language
    "SC211Q03JA",  # percentage of students from socioeconomically disadvantaged homes
    "SC211Q04JA",  # percentage of immigrant students, excluding refugees
    "SC211Q05JA",  # percentage of students whose parents immigrated
    "SC211Q06JA",  # percentage of refugee students
]

missing_student = [col for col in student_cols if col not in stu_columns]
if missing_student:
    raise ValueError(f"Missing student variables: {missing_student}")

_, sch_meta = pyreadstat.read_sav(school_sav_path, metadataonly=True)
sch_columns = sch_meta.column_names

missing_school = [col for col in school_cols if col not in sch_columns]
if missing_school:
    raise ValueError(f"Missing school variables: {missing_school}")

print(f"Replicate weights found: {len(rep_weights)}")


# =====================================================
# 3. Read student and school data
# =====================================================

print("Reading student data...")
student_df, _ = pyreadstat.read_sav(
    student_sav_path,
    usecols=student_cols,
    apply_value_formats=False,
)

print("Reading school data...")
school_df, _ = pyreadstat.read_sav(
    school_sav_path,
    usecols=school_cols,
    apply_value_formats=False,
)

# Keep Turkiye and Greece only
student_df = student_df[student_df["CNT"].isin(["TUR", "GRC"])].copy()
school_df = school_df[school_df["CNT"].isin(["TUR", "GRC"])].copy()

print("Student sample by country:")
print(student_df["CNT"].value_counts())

print("School sample by country:")
print(school_df["CNT"].value_counts())


# =====================================================
# 4. Prepare school-level variables and merge
# =====================================================

school_df = school_df.rename(columns={
    "SC211Q01JA": "PCT_HERITAGE_LANGUAGE_DIFF",
    "SC211Q03JA": "PCT_SOCIOECON_DISADVANTAGED",
    "SC211Q04JA": "PCT_IMMIGRANT_STUDENTS",
    "SC211Q05JA": "PCT_PARENTS_IMMIGRATED",
    "SC211Q06JA": "PCT_REFUGEE_STUDENTS",
})

# Ensure one school record per country-school pair
school_df = school_df.drop_duplicates(subset=["CNT", "CNTSCHID"]).copy()

df = student_df.merge(
    school_df,
    on=["CNT", "CNTSCHID"],
    how="left",
)

# Country coding: Greece = 0, Turkiye = 1
df["TURKEY"] = (df["CNT"] == "TUR").astype(int)

school_context_vars = [
    "PCT_SOCIOECON_DISADVANTAGED",
    "PCT_HERITAGE_LANGUAGE_DIFF",
    "PCT_IMMIGRANT_STUDENTS",
    "PCT_PARENTS_IMMIGRATED",
    "PCT_REFUGEE_STUDENTS",
]

# Rescale percentage variables to 10 percentage-point units.
# This makes coefficients easier to interpret.
for var in school_context_vars:
    df[var + "_10PP"] = df[var] / 10.0

print("Merged data shape:", df.shape)
print("Merged sample by country:")
print(df["CNT"].value_counts())


# =====================================================
# 5. Helper functions
# =====================================================

def weighted_ols(y, X, w):
    """
    Weighted least squares using matrix algebra.

    Returns:
        beta: coefficient vector
        valid_n: number of complete cases used
    """
    mask = ~np.isnan(y) & ~np.isnan(w) & np.all(~np.isnan(X), axis=1)

    y_clean = y[mask]
    X_clean = X[mask]
    w_clean = w[mask]

    if len(y_clean) == 0:
        raise ValueError("No valid observations after listwise deletion.")

    xtwx = X_clean.T @ (w_clean[:, None] * X_clean)
    xtwy = X_clean.T @ (w_clean * y_clean)

    try:
        beta = np.linalg.solve(xtwx, xtwy)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(xtwx, xtwy, rcond=None)[0]

    return beta, int(mask.sum())


def combine_pv(estimates, sampling_variances):
    """
    Combine estimates across plausible values.

    Total variance = average sampling variance + (1 + 1/M) * imputation variance.
    """
    estimates = np.array(estimates, dtype=float)
    sampling_variances = np.array(sampling_variances, dtype=float)

    M = len(estimates)

    final_estimate = np.mean(estimates)
    sampling_variance = np.mean(sampling_variances)
    imputation_variance = np.sum((estimates - final_estimate) ** 2) / (M - 1)
    total_variance = sampling_variance + (1 + 1 / M) * imputation_variance

    se = np.sqrt(total_variance)

    return {
        "estimate": final_estimate,
        "sampling_variance": sampling_variance,
        "imputation_variance": imputation_variance,
        "total_variance": total_variance,
        "se": se,
        "ci_lower": final_estimate - 1.96 * se,
        "ci_upper": final_estimate + 1.96 * se,
    }


def run_context_model(data, context_var):
    """
    Runs one adjusted model for one school-level contextual variable.

    Model:
        mathematics = intercept + country + student socioeconomic status + school context

    The school-context coefficient is interpreted as the estimated mathematics-score
    difference associated with a 10 percentage-point increase in the indicator.
    """
    estimates = []
    sampling_vars = []
    valid_ns = []

    for pv in math_pvs:
        model_cols = [
            pv,
            "TURKEY",
            "ESCS",
            context_var,
            main_weight,
        ] + rep_weights

        model_df = data[model_cols].dropna().copy()

        y = model_df[pv].to_numpy()

        X = np.column_stack([
            np.ones(len(model_df)),
            model_df["TURKEY"].to_numpy(),
            model_df["ESCS"].to_numpy(),
            model_df[context_var].to_numpy(),
        ])

        full_beta, valid_n = weighted_ols(
            y,
            X,
            model_df[main_weight].to_numpy(),
        )

        # The school-context coefficient is the fourth coefficient.
        full_context_beta = full_beta[3]

        replicate_context_betas = []

        for rw in rep_weights:
            rep_beta, _ = weighted_ols(
                y,
                X,
                model_df[rw].to_numpy(),
            )
            replicate_context_betas.append(rep_beta[3])

        replicate_context_betas = np.array(replicate_context_betas)

        # PISA Fay-adjusted BRR scaling factor with 80 replicate weights.
        sampling_var = 0.05 * np.sum((replicate_context_betas - full_context_beta) ** 2)

        estimates.append(full_context_beta)
        sampling_vars.append(sampling_var)
        valid_ns.append(valid_n)

    combined = combine_pv(estimates, sampling_vars)

    return {
        "Estimate": combined["estimate"],
        "SE": combined["se"],
        "95% CI Lower": combined["ci_lower"],
        "95% CI Upper": combined["ci_upper"],
        "Sampling variance": combined["sampling_variance"],
        "Imputation variance": combined["imputation_variance"],
        "Total variance": combined["total_variance"],
        "Valid student n": int(round(np.mean(valid_ns))),
    }


# =====================================================
# 6. Run RQ3 models
# =====================================================

context_labels = {
    "PCT_SOCIOECON_DISADVANTAGED_10PP": "Students from socioeconomically disadvantaged homes",
    "PCT_HERITAGE_LANGUAGE_DIFF_10PP": "Students whose heritage language differs from the test language",
    "PCT_IMMIGRANT_STUDENTS_10PP": "Immigrant students, excluding refugees",
    "PCT_PARENTS_IMMIGRATED_10PP": "Students whose parents immigrated",
    "PCT_REFUGEE_STUDENTS_10PP": "Refugee students",
}

rows = []

for var, label in context_labels.items():
    print(f"Running model for: {label}")
    result = run_context_model(df, var)

    rows.append({
        "School-level contextual indicator": label,
        "Estimate": result["Estimate"],
        "SE": result["SE"],
        "95% CI Lower": result["95% CI Lower"],
        "95% CI Upper": result["95% CI Upper"],
        "Valid student n": result["Valid student n"],
        "Interpretation": "Estimated mathematics-score change per 10 percentage-point increase, controlling for country and student socioeconomic background.",
        "Sampling variance": result["Sampling variance"],
        "Imputation variance": result["Imputation variance"],
        "Total variance": result["Total variance"],
    })

table8 = pd.DataFrame(rows)

# Rounded presentation table
presentation_cols = [
    "School-level contextual indicator",
    "Estimate",
    "SE",
    "95% CI Lower",
    "95% CI Upper",
    "Valid student n",
    "Interpretation",
]
table8_presentation = table8[presentation_cols].copy()

print("\nTable 8: RQ3 school-level contextual associations")
print(table8_presentation.round(3))


# =====================================================
# 7. Save output
# =====================================================

output_file = "rq3_school_context_results.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    table8_presentation.to_excel(writer, sheet_name="Table8_School_Context", index=False)
    table8.to_excel(writer, sheet_name="Technical_Details", index=False)

print(f"\nFile created: {output_file}")
