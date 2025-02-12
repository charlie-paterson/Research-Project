# Research-Project

This project aims to develop a machine learning algorithm to predict the refractive index of radar waves based on the input parameters of wave propagation and incidence angle. Using various machine learning models (such as Support Vector Machines and Fine Tree models), the goal is to improve the reliability and accuracy of radar systems by better understanding and predicting how radar waves interact with atmospheric conditions.

## Current Progress
I have successfully generated synthetic data to simulate radar wave propagation under varying atmospheric conditions, including factors like temperature inversions and humidity gradients. I have also implemented machine learning models, including Support Vector Regression (SVR) and Fine Tree models, to predict the refractive index based on the input data. The models are performing well, with the Fine Tree model yielding an R² score of 0.99, indicating excellent accuracy.

Currently, I am fine-tuning the models and optimizing them for real-world radar data once it's available. The next steps will involve testing the models on real-world data and further improving their performance in real-world conditions.

### Code
The code for this project is available in the following repository:

![Link to code here](First_ML_Radar_for_git.m)

Feel free to explore, contribute, or provide feedback. The code is organized into different modules based on data preprocessing, model training, and evaluation.



### Figures
<div style="text-align: center;">
  <img src="figure1.png" alt="Plot Description" width="300"/>
</div>

*Figure 1: This shows the plot from my fine tree regressor model where it shows an almost perfect fit. *


<div style="text-align: center;">
  <img src="Figures/SVMplot.png" alt="Plot Description" width="300"/>
</div>

## Challenges
* Adapting the models to handle noisy or incomplete real-world data.
* Fine-tuning hyperparameters for optimal performance.
* Handling the complexity of atmospheric effects in real radar data.

## Next Steps
* Integrate real-world radar data when available.
* Fine-tune models based on the new data.
* Conduct further validation to improve model robustness and reliability.
* Explore the use of other machine learning models and techniques to enhance predictions.
