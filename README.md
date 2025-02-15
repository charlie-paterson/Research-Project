# Research-Project

This project aims to develop a machine learning algorithm to predict the refractive index of radar waves based on the input parameters of wave propagation and incidence angle. Using various machine learning models (such as Support Vector Machines and Fine Tree models), the goal is to improve the reliability and accuracy of radar systems by better understanding and predicting how radar waves interact with atmospheric conditions.

## Current Progress
I have successfully generated synthetic data to simulate radar wave propagation under varying atmospheric conditions, including factors like temperature inversions and humidity gradients. I have also implemented machine learning models, including Support Vector Machine Regressor (SVM) and Fine Tree models, to predict the refractive index based on the input data. The models are performing well, with the Fine Tree model yielding an R² score of 0.99, indicating excellent accuracy.

Currently, I am fine-tuning the models and optimizing them for real-world radar data once it's available. The next steps will involve testing the models on real-world data and further improving their performance in real-world conditions.

### Code
For this project, I utilized MATLAB to generate synthetic data and implement two machine learning algorithms: the Fine Tree Regressor and the Support Vector Machine (SVM) Regressor. The synthetic data was carefully crafted to mimic realistic patterns, ensuring a diverse range of features and complexities that test the robustness of both models. This allowed me to simulate real-world conditions while maintaining full control over the data properties.

The code for this project is available in the following repository:

![Link to code here](Code/First_ML_Radar_for_git.m)

Feel free to explore, contribute, or provide feedback. The code is organized into different modules based on data preprocessing, model training, and evaluation.



### Figures

I utilized multiple figures to illustrate how well the Fine Tree Regressor model fits the actual values in the dataset. The plots reveal a near-perfect alignment between the predicted and actual data points, indicating the model's strong ability to capture both linear and non-linear relationships within the data.

<p align="center">
  <img src="Figures/errors.png" alt="Plot Description" width="300"/>
</p>

<p align="center"><em>Figure 1: This shows the plot from my fine tree regressor model, demonstrating an almost perfect fit.</em></p>

<br><br>

<p align="center">
  <img src="Figures/Rsquared.png" alt="Plot Description" width="300"/>
</p>

<p align="center"><em>Figure 1: This shows the plot from my fine tree regressor model, demonstrating an almost perfect fit.</em></p>

When compared to the Support Vector Machine (SVM) Regressor, the Fine Tree Regressor consistently demonstrated superior performance. While the SVM regressor struggled with overfitting and underfitting in certain regions—especially where the data exhibited complex, non-linear patterns—the fine tree model maintained high accuracy across all data ranges. This is largely due to the fine tree's inherent flexibility in partitioning the feature space, allowing it to adapt more effectively to subtle variations in the data.

Overall, the visual comparisons and performance metrics clearly establish the Fine Tree Regressor as the more effective model for this dataset, offering both precision and reliability in its predictions.

## Challenges
* Adapting the models to handle noisy or incomplete real-world data.
* Fine-tuning hyperparameters for optimal performance.
* Handling the complexity of atmospheric effects in real radar data.

## Next Steps
* Integrate real-world radar data when available.
* Fine-tune models based on the new data.
* Conduct further validation to improve model robustness and reliability.
* Explore the use of other machine learning models and techniques to enhance predictions.
