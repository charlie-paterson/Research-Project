# Research-Project

This project aims to develop a neural network model to predict the refractivity profile of radar waves based on sea clutter returns. By leveraging deep learning, the goal is to improve the accuracy and reliability of radar systems by modeling how radar waves interact with atmospheric conditions, enabling better real-time performance analysis.

## Current Progress
I have successfully generated synthetic data so that I can see what needs to be done for when I get the real data across. I have implemented a neural network model to predict the refractive index based on this data, achieving an R² score close to 1.00, indicating excellent accuracy.

Currently, I am fine-tuning the model and optimizing it with the synthetic data. The next steps involve testing on real-world conditions and further enhancing its performance for practical applications.

### Code
For this project, I used MATLAB to generate synthetic data and develop a neural network model to predict the refractivity profile. The data was designed to capture a proof of conceot, ensuring a diverse range of features for robust model training. This approach allows for simulation of real-world conditions while maintaining full control over data properties.

The code for this project is available in the following repository:

![Link to code here](Code/Neural_Net.m)

Feel free to explore, contribute, or provide feedback. The code is organized into different modules based on data preprocessing, model training, and evaluation.



### Figures

I utilized multiple figures to illustrate how well the neural network model fits the actual values in the dataset. The error histogram and performance metrics show a strong alignment between predicted and actual values, highlighting the model’s ability to accurately capture complex atmospheric relationships.

<p align="center">
  <img src="Figures/errors.png" alt="Plot Description" width="300"/>
</p>

<p align="center"><em>Figure 1: This shows the plot from my neural network model, demonstrating minimal error against the targets.</em></p>

The error histogram provides insight into the neural network model’s prediction accuracy by visualizing the distribution of errors between the predicted and actual refractive index values. The histogram shows a strong concentration of errors around zero, indicating minimal deviation and high model accuracy. This confirms the network’s ability to effectively learn atmospheric patterns and predict refractivity with precision.

<p align="center">
  <img src="Figures/Rsquared.png" alt="Plot Description" width="300"/>
</p>

<p align="center"><em>Figure 2: This shows the regression plots, demonstrating a perfect fit.</em></p>

The MATLAB neural network performance plot displays four key graphs that evaluate the model’s accuracy in predicting the refractivity profile. These include the regression plots, which compare predicted values against actual targets for training, validation, and test datasets. The R² scores in each plot are close to 1.00, indicating a strong correlation between predictions and actual values. The near-linear fit in all cases demonstrates the model’s effectiveness in capturing atmospheric refractivity patterns, confirming its reliability for radar wave propagation analysis.

<p align="center">
  <img src="Figures/epoch.png" alt="Plot Description" width="300"/>
</p>

<p align="center"><em>Figure 3: This shows how the mean squared error changed over the course of the training process.</em></p>

The MATLAB neural network performance plot includes a graph that illustrates how the Mean Squared Error (MSE) evolves during the training process. This graph tracks the MSE for the training, validation, and test datasets at each epoch, providing an indication of the model’s learning progression. As the training advances, the MSE gradually decreases, signaling that the model is minimizing the error between the predicted refractivity profile values and the actual targets. The consistent decrease in MSE for both training and validation sets suggests that the model is successfully learning and generalizing the patterns of atmospheric refractivity. The MSE curve's smooth decline and stabilization near the lower bound highlight the model’s effectiveness in capturing radar wave propagation patterns, indicating reliable performance for future predictions.

<p align="center">
  <img src="Figures/gradient.png" alt="Plot Description" width="300"/>
</p>

<p align="center"><em>Figure 4: This shows the gradient change over the course of the training process.</em></p>

The MATLAB neural network performance plot includes a graph that displays the gradient of the model’s error during the training process. This graph monitors how the gradient changes at each epoch, providing insight into how the optimization algorithm is adjusting the model’s weights to minimize the error. A decreasing gradient suggests that the model is converging, and the optimization is becoming more stable. If the gradient becomes very small, it indicates that the model is nearing its optimal set of weights, and further adjustments will have a minimal effect on reducing the error. The smooth decline of the gradient in the graph reflects the effectiveness of the training process, suggesting that the neural network is successfully learning and refining its ability to predict the refractivity profile. This behavior demonstrates the network's efficient optimization for radar wave propagation analysis, ensuring reliable predictions for future datasets.

<p align="center">
  <img src="Figures/evaluation.png" alt="Plot Description" width="300"/>
</p>

<p align="center"><em>Figure 5: This shows the evaluation metrics for the Neural Network.</em></p>

The training results figure in MATLAB displays the progress of the model’s performance throughout the training process. This figure typically shows how the training error (often the Mean Squared Error, MSE) evolves over each epoch. It provides valuable insight into how well the model is learning and adjusting its weights to minimize error.

Initially, the error tends to be higher, but as training progresses, the MSE decreases, indicating that the model is learning to make more accurate predictions. The graph typically shows both the training error and the validation error, allowing for the detection of overfitting. If the validation error begins to increase while the training error continues to decrease, it may signal that the model is overfitting to the training data. However, if both errors decrease and stabilize, it shows that the model is generalizing well and optimizing successfully.

The training results figure illustrates the model’s ability to converge toward an optimal solution, with error values gradually decreasing until they reach a minimum, indicating effective learning. This progression confirms that the neural network is successfully capturing the underlying patterns in the data, making it reliable for radar wave propagation analysis.

Overall, the visual comparisons and performance metrics clearly establish the Neural Network as an effective model for this dataset, offering both precision and reliability in its predictions.

## Challenges
* Adapting the models to handle noisy or incomplete real-world data.
* Fine-tuning hyperparameters for optimal performance.
* Handling the complexity of atmospheric effects in real radar data.

## Next Steps
* Integrate real-world radar data when available.
* Fine-tune models based on the new data.
* Conduct further validation to improve model robustness and reliability.
* Explore the use of other machine learning models and techniques to enhance predictions.
