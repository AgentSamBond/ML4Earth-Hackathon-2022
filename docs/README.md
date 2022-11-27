## ML4Earth Hackathon 2022

Author: Aleksei Zhuravlev

[Colab](https://colab.research.google.com/drive/1U45ropn7iRLj8vIHm2IalE1MnOIh7Tkx?usp=sharing) | [Video](https://youtu.be/NJut9FMinhM)

The notebook and presentation are also uploaded to this repository.

## Inspiration

Satellite imaging is widely used in USA and EU for crop yield prediction because it provides cheap and reliable data. At the same time, demand for remote sensing in South America and Asia is high, but low-income communities cannot afford their own satellites!

## What it does

Given historical data for US counties for the previous years, as well as the remote sensing and weather data, the project goals are:

* Predict soybean yield for each county in the current year
* Evaluate the contribution of satellite data with Explainable AI
* Find which period of the year is important
* Find out whether US satellites can be used in other regions

## How we built it

The project was made in Google Colab with python, scikit-learn, tensorflow

## Challenges we ran into

Data preparation step was very time consuming.

## Accomplishments that we're proud of

* trained a linear regression model, achieving 9% cross-validation error, 10% test set error
* trained a deep neural network, which had slightly better results!
* Applied feature permutation: remote sensing data turned out to be important for soybeans yield prediction
* In the US, satellites are needed only for 10-15 weeks per year
* They can be redirected to other regions, e.g. South America, Africa

## What we learned

Time management, data preparation, training of deep neural networks, presentation skills

## What's next for Atrium

* Acquire the data for other US counties 
* Take into account geographical conditions, e.g. plains, mountains
* Extend analysis to other countries

