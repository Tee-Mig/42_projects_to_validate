# ft_linear_regression

Simple implementation of linear regression with gradient descent to predict the price of a car from its mileage

## Goal

Understand how linear regression works by implementing the algorithm from scratch without machine learning libraries

## Technologies

- Python
- pandas
- numpy
- matplotlib

## Train the model

Train the model using a dataset containing km and price:

python train_model.py --thetas thetas.txt --file data.csv

This will train the model and save the parameters in thetas.txt.

## Predict prices

Predict prices from mileage values:

python predict_car.py --thetas thetas.txt --file data_to_predict.csv

The predictions will be saved in predictions.csv.

You can also run it without a file to enter the mileage manually:

python predict_car.py --thetas thetas.txt

## Evaluate the model

Evaluate the model performance:

python evaluate_model.py --thetas thetas.txt --file data.csv

This displays the MAE, RMSE and R² metrics.
