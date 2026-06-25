# ============================================================
# Supplementary Random-Intercept Multilevel Robustness Models
# RQ3: PISA 2022 Türkiye-Greece school-level contextual indicators
# Revised version: explicitly reads the Combined sheet or combines
# Turkiye + Greece sheets if Combined is unavailable.
# ============================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. File paths and sheet settings
# ------------------------------------------------------------

INPUT_FILE = Path("pisa2022_turkiye_greece_student_school_immigrant_merged.xlsx")
OUTPUT_FILE = Path("rq3_multilevel_robustness_results_revised.xlsx")

# The uploaded workbook contains these sheets:
# Turkiye, Greece, Combined, Student_Summary, School_Summary
# Use Combined for the two-country analysis.
PREFERRED_SHEET = "Combined"
FALLBACK_SHEETS = ["Turkiye", "Greece"]

# ------------------------------------------------------------
# 2. Read the correct sheet
# ------------------------------------------------------------

def read_analysis_data(input_file: Path) -> pd.DataFrame:
    """Read Combined sheet. If absent, concatenate Turkiye and Greece sheets."""
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    xls = pd.ExcelFile(input_file)
    print("Available sheets:", xls.sheet_names)

    if PREFERRED_SHEET in xls.sheet_names:
        print(f"Reading sheet: {PREFERRED_SHEET}")
        data = pd.read_excel(input_file, sheet_name=PREFERRED_SHEET)
    elif all(sheet in xls.sheet_names for sheet in FALLBACK_SHEETS):
        print(f"'{PREFERRED_SHEET}' not found. Combining sheets: {FALLBACK_SHEETS}")
        frames = [pd.read_excel(input_file, sheet_name=sheet) for sheet in FALLBACK_SHEETS]
        data = pd.concat(frames, ignore_index=True)
    else:
        raise ValueError(
            f"Could not find '{PREFERRED_SHEET}' or both fallback sheets {FALLBACK_SHEETS}. "
            f"Available sheets are: {xls.sheet_names}"
        )

    return data


df = read_analysis_data(INPUT_FILE)

# Clean column names just in case there are hidden spaces.
df.columns = [str(c).strip() for c in df.columns]

# ------------------------------------------------------------
# 3. Required variables and diagnostics
# ------------------------------------------------------------

pv_cols = [f"PV{i}MATH" for i in range(1, 11)]

school_context_vars = {
    "PCT_SOCIOECON_DISADVANTAGED": "Socioeconomically disadvantaged students",
    "PCT_HERITAGE_LANGUAGE_DIFF": "Heritage language differs from test language",
    "PCT_IMMIGRANT_STUDENTS": "Immigrant students, excluding refugees",
    "PCT_PARENTS_IMMIGRATED": "Students whose parents immigrated",
    "PCT_REFUGEE_STUDENTS": "Refugee students",
}

required_cols = [
    "CNT",
    "CNTSCHID",
    "ESCS",
    "W_FSTUWT",
    *pv_cols,
    *school_context_vars.keys(),
]

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

print("\nRaw country counts:")
print(df["CNT"].value_counts(dropna=False))
print("\nRows read:", len(df))

if len(df) < 13000:
    raise ValueError(
        "The dataset has fewer than 13,000 rows. It probably did not read the Combined sheet. "
        "Check the sheet name and input file."
    )

if df["CNT"].nunique(dropna=True) < 2:
    raise ValueError(
        "The dataset contains fewer than two countries. "
        "This analysis requires both Türkiye and Greece."
    )

# ------------------------------------------------------------
# 4. Country indicator and unique school identifier
# ------------------------------------------------------------

cnt_upper = df["CNT"].astype(str).str.upper().str.strip()

# Türkiye = 1, Greece = 0
# Works for common country codes/names: TUR, TURKIYE, TÜRKIYE, GRC, GREECE.
df["TURKIYE"] = np.where(
    cnt_upper.str.contains("TUR|TÜRK|TURKIYE|TÜRKIYE", regex=True),
    1,
    np.where(cnt_upper.str.contains("GRC|GREECE|GREEK|HELLAS|ELLADA", regex=True), 0, np.nan),
)

print("\nTURKIYE indicator counts:")
print(df["TURKIYE"].value_counts(dropna=False))

if df["TURKIYE"].isna().any():
    bad_values = df.loc[df["TURKIYE"].isna(), "CNT"].drop_duplicates().tolist()
    raise ValueError(f"Could not classify these CNT values as Türkiye or Greece: {bad_values}")

if df["TURKIYE"].nunique(dropna=True) < 2:
    raise ValueError(
        "TURKIYE indicator has no variation. "
        "The model requires both Türkiye and Greece."
    )

# Create unique school ID across countries.
# CNTSCHID may repeat across countries, so country code must be included.
df["school_id"] = df["CNT"].astype(str).str.strip() + "_" + df["CNTSCHID"].astype(str).str.strip()

print("\nUnique schools:", df["school_id"].nunique())

# ------------------------------------------------------------
# 5. Numeric conversion
# ------------------------------------------------------------

numeric_cols = ["ESCS", "W_FSTUWT", *pv_cols, *school_context_vars.keys()]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ------------------------------------------------------------
# 6. Center ESCS around weighted mean of analysed two-country sample
# ------------------------------------------------------------

escs_data = df[["ESCS", "W_FSTUWT"]].dropna()
weighted_escs_mean = np.average(escs_data["ESCS"], weights=escs_data["W_FSTUWT"])
df["ESCS_centered"] = df["ESCS"] - weighted_escs_mean

print(f"\nWeighted ESCS mean used for centering: {weighted_escs_mean:.4f}")

# ------------------------------------------------------------
# 7. Rescale school-level percentage indicators to 10 percentage-point units
# ------------------------------------------------------------

for var in school_context_vars:
    df[f"{var}_10"] = df[var] / 10.0

# ------------------------------------------------------------
# 8. Helper functions
# ------------------------------------------------------------

def fit_mixed_model(data: pd.DataFrame, formula: str, group_col: str = "school_id"):
    """Fit random-intercept multilevel model. Returns result or None."""
    methods = ["lbfgs", "powell", "cg", "nm"]

    for method in methods:
        try:
            model = smf.mixedlm(formula=formula, data=data, groups=data[group_col])
            result = model.fit(reml=False, method=method, maxiter=1000, disp=False)
            if getattr(result, "converged", True):
                return result
        except Exception:
            continue

    return None


def pool_pv_results(pv_results, term_name: str):
    """Rubin-style pooling across plausible values for a fixed-effect term."""
    estimates = []
    variances = []

    for res in pv_results:
        if res is not None and term_name in res.params.index and term_name in res.bse.index:
            estimates.append(float(res.params[term_name]))
            variances.append(float(res.bse[term_name]) ** 2)

    m = len(estimates)
    if m == 0:
        return {
            "estimate": np.nan,
            "se": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "successful_pv_models": 0,
        }

    estimates = np.array(estimates)
    variances = np.array(variances)

    q_bar = estimates.mean()
    u_bar = variances.mean()
    b = estimates.var(ddof=1) if m > 1 else 0.0
    total_var = u_bar + (1.0 + 1.0 / m) * b
    se = np.sqrt(total_var)

    return {
        "estimate": q_bar,
        "se": se,
        "ci_lower": q_bar - 1.96 * se,
        "ci_upper": q_bar + 1.96 * se,
        "successful_pv_models": m,
    }


def calculate_icc(result):
    """Calculate ICC from random-intercept model result."""
    if result is None:
        return np.nan, np.nan, np.nan

    try:
        school_var = float(result.cov_re.iloc[0, 0])
        residual_var = float(result.scale)
        icc = school_var / (school_var + residual_var)
        return school_var, residual_var, icc
    except Exception:
        return np.nan, np.nan, np.nan


def interpret_ci(estimate, lo, hi):
    if pd.isna(estimate):
        return "Model did not converge or estimate unavailable."
    if hi < 0:
        return "Negative and statistically significant."
    if lo > 0:
        return "Positive and statistically significant."
    return "Not statistically significant."

# ------------------------------------------------------------
# 9. Null random-intercept models for ICC
# ------------------------------------------------------------

icc_rows = []
print("\nRunning null random-intercept models for ICC...")

for pv in pv_cols:
    null_data = df[[pv, "school_id"]].dropna().copy()
    null_data = null_data.rename(columns={pv: "MATH"})

    res = fit_mixed_model(null_data, "MATH ~ 1")
    school_var, residual_var, icc = calculate_icc(res)

    icc_rows.append({
        "plausible_value": pv,
        "valid_students": len(null_data),
        "valid_schools": null_data["school_id"].nunique(),
        "school_level_variance": school_var,
        "student_level_residual_variance": residual_var,
        "ICC": icc,
        "converged": bool(res is not None and getattr(res, "converged", True)),
    })

icc_df = pd.DataFrame(icc_rows)

icc_summary = pd.DataFrame([{
    "mean_valid_students": icc_df["valid_students"].mean(),
    "mean_valid_schools": icc_df["valid_schools"].mean(),
    "mean_school_level_variance": icc_df["school_level_variance"].mean(),
    "mean_student_level_residual_variance": icc_df["student_level_residual_variance"].mean(),
    "mean_ICC_across_plausible_values": icc_df["ICC"].mean(),
    "successful_null_models": int(icc_df["converged"].sum()),
}])

print("\nICC summary:")
print(icc_summary)

# ------------------------------------------------------------
# 10. RQ3 random-intercept multilevel robustness models
# ------------------------------------------------------------

main_rows = []
pv_detail_rows = []

print("\nRunning RQ3 multilevel robustness models...")

for raw_var, label in school_context_vars.items():
    scaled_var = f"{raw_var}_10"
    print(f"\nIndicator: {label}")

    pv_results = []

    for pv in pv_cols:
        model_data = df[[pv, "TURKIYE", "ESCS_centered", scaled_var, "school_id"]].dropna().copy()
        model_data = model_data.rename(columns={pv: "MATH"})

        formula = f"MATH ~ TURKIYE + ESCS_centered + {scaled_var}"
        res = fit_mixed_model(model_data, formula)
        pv_results.append(res)

        pv_detail_rows.append({
            "school_level_indicator": label,
            "raw_variable": raw_var,
            "scaled_variable": scaled_var,
            "plausible_value": pv,
            "estimate": np.nan if res is None else res.params.get(scaled_var, np.nan),
            "se": np.nan if res is None else res.bse.get(scaled_var, np.nan),
            "valid_students": len(model_data),
            "valid_schools": model_data["school_id"].nunique(),
            "converged": bool(res is not None and getattr(res, "converged", True)),
        })

    pooled = pool_pv_results(pv_results, scaled_var)

    # Reporting n based on complete cases for PV1; should be same for all PVs unless PV missing differs.
    reporting_data = df[[pv_cols[0], "TURKIYE", "ESCS_centered", scaled_var, "school_id"]].dropna().copy()

    main_rows.append({
        "school_level_indicator": label,
        "variable": raw_var,
        "estimate_per_10_percentage_point_increase": pooled["estimate"],
        "SE": pooled["se"],
        "CI_lower": pooled["ci_lower"],
        "CI_upper": pooled["ci_upper"],
        "valid_students": len(reporting_data),
        "valid_schools": reporting_data["school_id"].nunique(),
        "successful_pv_models": pooled["successful_pv_models"],
        "interpretation": interpret_ci(pooled["estimate"], pooled["ci_lower"], pooled["ci_upper"]),
    })

results_df = pd.DataFrame(main_rows)
pv_details_df = pd.DataFrame(pv_detail_rows)

# ------------------------------------------------------------
# 11. Input diagnostics sheet
# ------------------------------------------------------------

country_diag = (
    df.groupby("CNT", dropna=False)
    .agg(
        rows=("CNT", "size"),
        schools=("school_id", "nunique"),
        mean_ESCS=("ESCS", "mean"),
        mean_MATH_AVG=("MATH_AVG", "mean") if "MATH_AVG" in df.columns else ("PV1MATH", "mean"),
    )
    .reset_index()
)

school_missing_rows = []
for raw_var, label in school_context_vars.items():
    tmp = df[["school_id", raw_var]].drop_duplicates()
    school_missing_rows.append({
        "school_level_indicator": label,
        "variable": raw_var,
        "valid_schools": int(tmp[raw_var].notna().sum()),
        "missing_schools": int(tmp[raw_var].isna().sum()),
    })
school_missing_df = pd.DataFrame(school_missing_rows)

# ------------------------------------------------------------
# 12. Save results
# ------------------------------------------------------------

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    results_df.to_excel(writer, sheet_name="RQ3_multilevel_results", index=False)
    icc_summary.to_excel(writer, sheet_name="ICC_summary", index=False)
    icc_df.to_excel(writer, sheet_name="ICC_by_PV", index=False)
    pv_details_df.to_excel(writer, sheet_name="PV_model_details", index=False)
    country_diag.to_excel(writer, sheet_name="Input_country_diagnostics", index=False)
    school_missing_df.to_excel(writer, sheet_name="School_missingness", index=False)

print("\nDone.")
print(f"Results saved to: {OUTPUT_FILE}")
print("\nMain multilevel robustness results:")
print(results_df)
