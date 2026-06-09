# House Price Prediction using Linear Regression

## Overview
This project implements a Linear Regression model in Python to predict house prices based on:

- Square Footage
- Number of Bedrooms
- Number of Bathrooms

The model is trained using a manually created dataset and provides estimated house prices for new houses based on user inputs.

## Features
- Linear Regression implementation from scratch
- Multiple input features
- House price prediction
- Model performance evaluation
- Displays R² Score, RMSE, and MAE
- Feature importance analysis

## Dataset

Sample Features:
- Square Footage (sqft)
- Bedrooms
- Bathrooms

Target:
- House Price ($)

## Technologies Used

- Python 3
- NumPy
- Statistics
- Linear Regression

## Model Performance

Example Results:

- R² Score: 0.999466
- RMSE: $2,964.91
- MAE: $2,215.54

## Example Prediction

Input:

```python
model.predict(2200, 4, 3)
```

Output:

```
$327,515
```

## How to Run

Clone the repository:

```bash
git clone https://github.com/sakshigund-tech/house-price-prediction.git
```

Navigate to the project folder:

```bash
cd house-price-prediction
```

Run the program:

```bash
python house_price_linear_regression.py
```

## Project Structure

```text
house-price-prediction/
│
├── house_price_linear_regression.py
├── README.md
```

## Future Improvements

- Use larger real-world datasets
- Add data visualization
- Build a web application using Flask
- Support CSV file input
- Compare multiple regression algorithms

## Author

Sakshi Gund

GitHub:
https://github.com/sakshigund-tech
