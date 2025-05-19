# Towards-Equitable-Care-Innovative-Approaches-to-HCT-Survival-Prediction

## Project Overview
This project focuses on developing and comparing predictive models for post-Hematopoietic Cell Transplantation (HCT) survival outcomes. The primary goal is to create models that provide both accurate and fair predictions across diverse patient populations, addressing disparities related to socioeconomic status, race, and geographic factors.

## Challenge Description
Current predictive models for allogeneic HCT patient survival often fail to adequately address disparities, which can impact patient care quality, resource utilization, and trust in healthcare systems. This project aims to bridge these gaps by developing advanced models that maintain high prediction accuracy while ensuring fairness across diverse patient groups.

## Models Implemented
We implemented and compared two deep learning approaches:

### 1. Convolutional Neural Network (CNN)
- **Architecture**: Input layer (14×14) → Gaussian noise → Multiple Conv1D layers with regularization → Pooling layers → Fully connected layers → Sigmoid output
- **Key Features**:
  - Gaussian noise for improved generalization
  - He initialization for weights
  - L2 regularization
  - Dropout layers to prevent overfitting
  - Model checkpointing to save the best version
- **Training**: 300 epochs with batch size of 512
- **Performance**: 64% accuracy on hidden test data

### 2. Deep Neural Network (DNN)
- **Architecture**: Input layer → Multiple dense layers (128→128→64→64→64→32→32→16) → Sigmoid output
- **Training**: 35 epochs with batch size of 32
- **Performance**: 59% accuracy on hidden test data

## Results and Analysis
The CNN model outperformed the DNN model on the hidden test set, achieving 64% accuracy compared to the DNN's 59%. This suggests that the CNN's ability to capture spatial or sequential relationships in the medical data, combined with its more sophisticated regularization techniques, provided better generalization to unseen data.

## Key Differences Between Models
1. **Data Representation**:
   - CNN: Structured input as a 2D spatial representation (14×14)
   - DNN: Used flattened feature vectors

2. **Regularization**:
   - CNN: Implemented multiple regularization techniques (noise injection, dropout, L2)
   - DNN: Simpler architecture with fewer regularization components

3. **Training Approach**:
   - CNN: Used model checkpointing to save the best performing model
   - DNN: Trained for fewer epochs with smaller batch sizes

## Future Directions
1. **Model Enhancement**:
   - Experiment with hybrid architectures combining CNN and DNN elements
   - Explore additional regularization techniques specific to healthcare data
   - Implement fairness constraints directly in the model optimization

2. **Feature Engineering**:
   - Develop more sophisticated approaches to represent medical features spatially
   - Create domain-specific feature transformations based on medical knowledge

3. **Fairness Evaluation**:
   - Conduct thorough analysis of model performance across different demographic groups
   - Implement techniques to mitigate any discovered biases

## Conclusion
The CNN model demonstrated superior performance for HCT survival prediction, suggesting that considering potential relationships between medical features provides valuable predictive power. Future work will focus on enhancing model fairness while maintaining high accuracy to ensure equitable healthcare outcomes across diverse patient populations.

## Libraries and Tools
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Acknowledgments
This work uses synthetic data that mirrors real-world patient characteristics while protecting privacy, enabling the development of models that can eventually be applied in clinical settings.
