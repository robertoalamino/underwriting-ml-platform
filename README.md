# Underwriting ML Platform

**A reproducible machine learning project for credit risk modelling, probability calibration and underwriting decision optimisation.**

This project explores the development of an end-to-end machine learning (ML) platform for consumer credit underwriting, combining statistical modelling, predictive analytics, software engineering and MLOps.

The initial implementation focuses on estimating credit risk using historical LendingClub loan data, with particular attention to chronological validation, probability calibration and the reliability of predictions across different lending cohorts.

The project is being developed incrementally, from exploratory analysis and baseline modelling towards a reproducible, tested and deployable ML system incorporating experiment tracking, automated pipelines, monitoring and model governance.

**Current status:** Active development. The existing implementation provides a retrospective modelling baseline, not a production-ready underwriting decision system.

## Key Results

The initial modelling experiment uses regularised logistic regression to estimate credit risk using the historical LendingClub loan dataset, available from [Zenodo](https://zenodo.org/records/11295916)
([DOI: 10.5281/zenodo.11295916](https://doi.org/10.5281/zenodo.11295916)).

The model is trained on loans originated between 2007 and 2015, with 2016 used for validation and probability recalibration, and 2017 reserved for evaluating the resulting predictions.

The following results were obtained on the 2017 test cohort (169,117 loans):

| Metric            | Original model | Intercept correction | Intercept and slope correction |
| ----------------- | -------------: | -------------------: | -----------------------------: |
| ROC-AUC           |         0.6729 |               0.6729 |                         0.6729 |
| Average precision |         0.3640 |               0.3640 |                         0.3640 |
| Log loss          |         0.5223 |               0.5136 |                         0.5134 |

The original model systematically underestimated the observed default rate in the 2017 cohort. The mean predicted default probability was approximately 18%, compared with an observed default rate of 23.13%. Recalibration parameters estimated using the 2016 validation cohort improved probabilistic predictions on the independent 2017 test cohort, reducing log loss without affecting ROC-AUC. Following recalibration, the mean predicted default probability increased to approximately 22.33%.

These results demonstrate the importance of distinguishing between a model's ability to rank borrowers by risk and the reliability of its estimated probabilities.

The experiment is retrospective and subject to limitations arising from selective loan acceptance, outcome availability and differences between origination cohorts. These limitations are discussed in the
[Limitations](#limitations) section below.

## Dataset and Reproducibility

### Data Source

The project uses a curated LendingClub loan dataset available from Zenodo:

- Dataset: [LendingClub loan data](https://zenodo.org/records/11295916)
- DOI: [10.5281/zenodo.11295916](https://doi.org/10.5281/zenodo.11295916)
- Observations: 1,347,681 loans
- Target variable: `Default` (0 = Fully Paid, 1 = Charged Off)

The dataset contains historical information about accepted loans and their eventual outcomes. It does not include the full population of loan applicants, including rejected applications.

### Obtaining the Data

The dataset is not included in this repository.

1. Download the dataset from the Zenodo record linked above.
2. Create the directory `data/raw/` in the project root.
3. Place the CSV file in that directory using the following name:

   `data/raw/LC_loans_granting_model_dataset.csv`

The `data/raw/` directory is excluded from version control through
`.gitignore`.

### Python Environment

The project requires Python 3.11 or 3.12.

From the repository root, create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install jupyterlab matplotlib ipykernel pytest
```

The commands above use Git Bash on Windows. Virtual environment activation differs on other operating systems and shells.

### Reproducing the Experiments

The existing experiments are organised into two Jupyter notebooks:

1. `notebooks/01_data_exploration.ipynb`:
   exploratory analysis of the LendingClub dataset.

2. `notebooks/02_logistic_regression_baseline.ipynb`:
   preprocessing, temporal partitioning, logistic regression
   modelling, probability calibration and evaluation.

Run JupyterLab from the repository root:

```bash
jupyter lab
```

Execute the notebooks in numerical order, using the virtual environment as the Python kernel.

The modelling experiment uses the following chronological partitions:

| Partition  | Origination years | Number of loans |
| ---------- | ----------------- | --------------: |
| Training   | 2007–2015         |         829,347 |
| Validation | 2016              |         293,057 |
| Test       | 2017              |         169,117 |
| Diagnostic | 2018              |          56,160 |

The 2018 cohort is reserved for diagnostic investigation because its observed outcomes are particularly susceptible to loan-maturity bias.

### Automated Tests

Run the existing automated tests from the repository root:

```bash
python -m pytest
```

The current test suite checks the temporal data-splitting logic. Additional tests and automated experiment reproduction will be introduced as the project develops.

## Decision Problem

The ultimate objective of the platform is to support consumer credit underwriting decisions by combining predictive modelling with economic decision optimisation.

### Predictive Modelling

The initial modelling task estimates the probability of a loan eventually being charged off, conditional on the information available about the loan and borrower:

$P(\text{Default} = 1 \mid X)$

where $X$ represents the model's input features.

This probability is estimated using historical data from accepted LendingClub loans with observed outcomes. Consequently, the model does not directly estimate the default probability for the entire population of credit applicants.

The current implementation focuses on three aspects:

1. **Discrimination:** How effectively does the model distinguish between loans with different observed outcomes?

2. **Calibration:** How closely do predicted probabilities correspond to observed default rates?

3. **Temporal generalisation:** How reliably does the model perform on lending cohorts originating after the training period?

### Underwriting Decision Optimisation

Predicting credit risk is only one component of an underwriting decision system.

A complete system must also consider the economic consequences of approving or rejecting an application, including expected returns, potential losses and lending constraints.

The intended decision framework therefore separates:

1. **Risk estimation:** Predict the probability of an adverse credit outcome.

2. **Probability calibration:** Assess and, where necessary, correct the reliability of the estimated probabilities.

3. **Economic evaluation:** Estimate the expected value of alternative lending decisions.

4. **Decision optimisation:** Select lending decisions that maximise an appropriate economic objective subject to
   operational, regulatory and risk constraints.

The current project implements the initial predictive modelling and calibration stages. Economic decision optimisation remains part of the development roadmap.

## Repository Structure

The repository currently has the following structure:

```text
underwriting-ml-platform/
├── docs/
│   └── problem_definition.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_logistic_regression_baseline.ipynb
├── src/
│   └── underwriting/
│       ├── __init__.py
│       ├── data.py
│       └── preprocessing.py
├── tests/
│   └── test_data.py
├── .gitignore
├── pyproject.toml
└── README.md
```

## Limitations

The current implementation is a retrospective modelling study using historical LendingClub loans. Several limitations must be considered when interpreting its results.

### Selection Bias

The most important limitation is that the dataset contains accepted loans with observed outcomes, rather
than the complete population of credit applications.

Consequently, the model is trained on borrowers who passed historical lending selection procedures. Its predictions cannot automatically be generalised to rejected applicants or to alternative underwriting policies.

### Outcome Definition and Loan Maturity

The target variable distinguishes loans classified as *Fully Paid* from those classified as *Charged Off*.

This is not equivalent to estimating the probability of default within a fixed time horizon. Loans originated in different periods may have different opportunities to reach their final observed outcomes.

The 2018 cohort is therefore treated separately for diagnostic purposes because, being the most recent cohort and last available year of data, its outcome distribution is particularly susceptible to loan-maturity bias.

### Temporal Distribution Shifts

Borrower characteristics, lending policies and economic conditions may change over time.

The project uses chronological training, validation and test partitions to investigate generalisation across lending cohorts. However, successful performance on historical cohorts does not guarantee equivalent performance under future conditions.

### Model Scope

The current baseline uses regularised logistic regression and a limited set of borrower and loan characteristics.

The reported results establish an initial reference point rather than the best achievable predictive performance.

In particular, the project has not yet implemented a complete economic decision model, a comprehensive fairness assessment, production monitoring or its deployment infrastructure.

### Interpretation

The model's predictions represent statistical correlations within the available historical data. **They should not be interpreted as causal effects or used directly to make real-world lending decisions without further validation, governance and appropriate regulatory safeguards.**

## Development Roadmap

The project is being developed incrementally, combining statistical modelling with software engineering and MLOps practices. 

The project's roadmap is organised into the five stages listed below.

### 1. Data Exploration and Baseline Modelling

**Status: Initial implementation completed**

- [x] Define the underwriting problem and modelling objectives.
- [x] Explore the historical LendingClub dataset.
- [x] Implement chronological training, validation and test partitions.
- [x] Develop a preprocessing pipeline for numerical and categorical features.
- [x] Train a regularised logistic regression baseline.
- [x] Evaluate discrimination and probabilistic performance.
- [x] Investigate probability calibration across temporal cohorts.

### 2. Reproducible ML Engineering

**Status: In progress**

- [x] Establish the Python project structure.
- [x] Introduce automated tests for temporal data partitioning.
- [ ] Extract preprocessing, modelling, calibration and evaluation functionality into reusable Python modules.
- [ ] Implement a reproducible training and evaluation pipeline.
- [ ] Introduce experiment tracking and model versioning.
- [ ] Expand automated testing and data validation.

### 3. Underwriting Decision Optimisation

**Status: Planned**

- [ ] Develop an economic framework for credit approval decisions.
- [ ] Incorporate expected returns, losses and lending constraints.
- [ ] Investigate risk-based approval thresholds.
- [ ] Evaluate predictive performance against economic outcomes.
- [ ] Analyse the sensitivity of decisions to modelling assumptions.

### 4. Deployment and MLOps

**Status: Planned**

- [ ] Containerise the application using Docker.
- [ ] Introduce automated CI/CD workflows.
- [ ] Implement API-based model serving.
- [ ] Investigate cloud deployment using Microsoft Azure.
- [ ] Develop data and model monitoring.
- [ ] Implement drift detection and model performance reporting.

### 5. Explainability and Model Governance

**Status: Planned**

- [ ] Introduce model explainability and feature attribution.
- [ ] Investigate fairness and potential sources of bias.
- [ ] Develop model documentation and governance procedures.
- [ ] Define monitoring, validation and retraining policies.

The roadmap represents the intended development as it is at the moment rather than a commitment
to a fixed implementation schedule. Priorities may change and evolve as the project progresses.