/////////////////////////////////////////////////

Build Machine Learning Models to Predict Glaucoma

///////////////////////////////////////////////

using angle closure glaucoma data, 
The response of interest is ANGLE.CLOSURE and the candidate predictors consist of the remaining data columns

(a) perform data manipulation, cleaning, and multiple imputation to replace missing values, 
(b) fit a spectrum of prediction models, 
(c) select a model (or combination of models) based on estimated prediction accuracy, 
(d) generate visualizations for model performance and variable impact.


Step 1 Data Manipulation


Step 2 Develop Prediction Models
Develop 5 prediction models for angle closure glaucoma with different parameters setting:
SVM (one.R)
KNN (two.R)
Random forest (three.R)
Logistic regression (three.R)
Neural network (four.R)


Step 3 Model and Tuning Parameter Selection
Use 10-fold cross-validation
The accuracy measure of interest (for model and tuning parameter selection) is AUC.


Step 4 Stacking (stacking.R)
Generate stacked ensemble model based upon the 5 selected (with optimized tuning parameters) prediction models.


Step 5 Validation (validation.R)
Validate all the above models.


Step 6 Visualizations
For each of the 5 base prediction models, generate plots of cross-validated AUC vs. tuning parameter values.
For each of the 7 prediction models (5 base prediction models + 2 stacked models), generate ROC curves (plots) annotated with the corresponding AUCs using the validation datasets.
