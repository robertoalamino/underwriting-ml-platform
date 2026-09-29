# Underwriting ML Platform- Problem Definition

## 1. Business problem

The purpose of the system is to suppor underwriting decisions for consumer credit applications.

For each applicant, the system should estimate credit risk and use that estimate, together with economic and policy considerations, to support an underwriting decision.

The primary decision is whether extending credit to an applicant represents and acceptable risk-adjusted economic opportunity.

## 2. Machine-learning problem

For an applicant represented by features x, the predictive model estimates

P(default|x)

The model therefore estimates the probability of default rather than directly making an approval decision.

## 3. Decision problem

The underlying decision is treated separately from the predictive model.

A simplified decision may be

- approve;
- decline.

A more realistic system may later introduce

- approve;
- refer for manual review;
- decline.

The decision layer may incorporate:

- predicted probability of default;
- expected profit or loss;
- lending policy;
- risk appetite;
- regulatory constraints;
- fairness considerations;
- model uncertainty.

## 4. Economic decision

In a simplified setting let

G = gain from an approved customer who does not default,

L = loss from an approved customer who defaults,

p = predicted probability of default.

The expected value of the approval is

EV(approve) = (1-p)G -pL.

An economically rational approval policy therefore depends on both predicted risk and the asymmetric economic consequences of the decision.

## 5. Project objective

The objective is not simply maximise objective accuracy.

The project will develop an end-to-end machine learning system that connects

data -> risk estimation -> decision optimisation -> business value

while addressing production concerns including reproducibility, testing, deployment, monitoring, explainability and governance.