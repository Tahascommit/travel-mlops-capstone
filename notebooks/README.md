# Notebooks

The three notebooks follow the project template (know your data, variables, wrangling, 15+ charts, hypothesis tests, feature engineering, three models with tuning, explainability, saving the model).

| Notebook | Task |
|---|---|
| `Flight_Price_Prediction_Regression.ipynb` | flight price regression |
| `Gender_Classification.ipynb` | gender classification from the user's name |
| `Hotel_Recommendation_System.ipynb` | hotel recommendation |

They load the data from `../data/raw/` when run inside the repo and from GitHub when run on Colab, so they work in both places without uploading anything. Each one runs top to bottom without errors (about 1 to 2 minutes each).

The production versions of the models are trained by the scripts in `src/` with the same features and settings as the notebooks.
