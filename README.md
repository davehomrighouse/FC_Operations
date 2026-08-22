---
title: FC Operations - Next-Hour Backlog Risk
emoji: 📦
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: false
---

# FC Operations — Next-Hour Backlog Risk Predictor

This application demonstrates a machine-learning approach for predicting next-hour backlog risk in a simulated fulfillment center environment.

The project uses operational conditions from the current and upcoming hour to estimate whether the fulfillment process is at risk of carrying excessive backlog into the next hour.

## Model Purpose

The model answers the question:

> **Given the current operating conditions, is there a risk of backlog during the next hour?**

The application is intended as a demonstration of predictive decision support using synthetic fulfillment-center data.

It is not intended to determine the causal effect of changing staffing levels or other operational interventions.

## User Inputs

The application accepts four operational inputs:

- **Current Backlog** — units remaining to be processed entering the next hour.
- **Planned Work** — expected units scheduled for the next hour.
- **Packers Assigned** — number of packers assigned for the upcoming hour.
- **Bottleneck Present** — indicates whether an operational bottleneck is currently present.

The application internally engineers an additional feature:

`work_pressure = current_backlog / planned_work`

This represents the size of the existing backlog relative to the next hour's planned workload.

## Predictive Model

The deployed model is a **Logistic Regression classifier** implemented using Scikit-learn.

The final model uses three predictive features:

- `work_pressure`
- `packers_assigned`
- `bottleneck_flag`

The complete Scikit-learn preprocessing and classification pipeline is serialized and loaded by the application for inference.

The classification threshold is stored separately as deployment metadata.

## Model Output

The application returns:

**Predicted Backlog Risk**

- `Backlog Risk`
- `No Backlog Risk`

**Probability of Backlog Risk**

The probability represents the Logistic Regression model's estimated probability of the positive `Backlog Risk` class.

## Model Selection

Random Forest and Logistic Regression models were evaluated during model development.

Model performance was compared using metrics including:

- Recall
- F2-score
- ROC AUC
- Cross-validation performance

Recall and F2-score were emphasized because failing to identify a developing backlog was considered more costly than generating an additional risk warning.

The final Logistic Regression model was selected because it provided comparable or slightly better predictive performance while offering a simpler and more interpretable model.

## Feature Selection

Feature reduction and cross-validation were used to evaluate whether a simpler feature set could retain the predictive performance of larger models.

Nested Logistic Regression models were also evaluated using likelihood-ratio tests, AIC, and BIC.

The final three-feature model was selected after analysis showed that adding `utilization_rate` provided little additional explanatory or predictive value.

## Important Interpretation

The application is a **risk prediction system**, not a staffing optimization system.

For example, `packers_assigned` is an input to the predictive model, but changing the number of packers in the interface should not be interpreted as estimating the causal effect of adding or removing workers.

A question such as:

> "What is the backlog risk given 10 assigned packers?"

is a prediction question supported by the model.

A question such as:

> "How much will backlog risk decrease if management adds five packers?"

is an intervention or counterfactual question and is outside the scope of the current model.

## Model Limitations

The dataset used for this project is **synthetic** and was designed to simulate fulfillment-center operating conditions. Results should not be interpreted as predictions for an actual fulfillment operation.

The training data does not contain observations where `planned_work = 0`.

Because `work_pressure` is calculated as:

`current_backlog / planned_work`

the model cannot generate a valid model-based prediction when planned work is zero and an existing backlog is present.

The application therefore handles unsupported operating conditions separately rather than assigning an artificial model probability.

This distinction prevents deterministic business rules from being represented as model-generated probabilities.

## Future Development

A future version of the project could extend the predictive system into an **agentic decision-support system**.

The current Logistic Regression model would remain responsible for detecting next-hour backlog risk, while specialized agents could evaluate operational conditions and suggest actions for management review, such as:

- reviewing available staffing capacity,
- investigating active bottlenecks,
- evaluating work diversion opportunities,
- reviewing incoming work release.

A later simulation or counterfactual model could evaluate the expected effects of proposed interventions before actions are recommended.

The intended progression is:

`Risk Prediction → Operational Interpretation → Agent Recommendations → Human Decision`

## Technology

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Gradio
- Hugging Face Spaces
- GitHub Actions

## Disclaimer

This application is a portfolio and educational project using synthetic data. It is not an operational system and should not be used to make real-world fulfillment, staffing, or production decisions.
