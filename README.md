# Mathematics Equity in Türkiye and Greece Using PISA 2022

This repository contains the analysis code and outputs for the manuscript **“Mathematics Equity in Türkiye and Greece Using PISA 2022.”** The study examines mathematics performance and educational equity in Türkiye and Greece using OECD PISA 2022 student-level and school-level data.

The repository is intended to support transparency and reproducibility by providing the scripts and outputs used for the three main research questions of the study:

* **RQ1:** To what extent do Türkiye and Greece differ in their estimated mean mathematics performance in PISA 2022?
* **RQ2:** How does student socioeconomic background relate to mathematics performance in Türkiye and Greece?
* **RQ3:** To what extent are school-level socioeconomic disadvantage and migration-related composition associated with mathematics performance in Türkiye and Greece?

## Data source

The analyses use publicly available and anonymized data from the **OECD PISA 2022 Database**. The raw PISA 2022 data are not included in this repository. Researchers who wish to reproduce the analyses should download the relevant student-level and school-level PISA 2022 data files directly from the OECD:

https://www.oecd.org/en/data/datasets/pisa-2022-database.html

## Repository contents

This repository includes:

* Code used for data preparation and variable selection
* Code used for the RQ1 country-level mathematics performance comparison
* Code used for the RQ2 socioeconomic gradient and achievement-gap analyses
* Code used for the RQ3 school-level contextual association analyses
* Output files, including tables and figures generated from the analyses

## Reproducibility notes

The primary analyses use PISA mathematics plausible values, final student weights, and replicate weights in order to account for the PISA complex survey design. Supplementary robustness checks are also included where applicable.

Because the raw OECD PISA 2022 data are not redistributed in this repository, users should first download the official PISA 2022 data files from the OECD and place them in the appropriate local data folder before running the scripts.

## Citation

If using or referring to this repository, please cite the associated manuscript:

Metin, S., & Franko, C. *Mathematics Equity in Türkiye and Greece Using PISA 2022.*

## License

The analysis code in this repository is shared for academic and reproducibility purposes. The OECD PISA 2022 data remain subject to the terms and conditions of the OECD data source.
