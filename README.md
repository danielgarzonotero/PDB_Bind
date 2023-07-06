# Project Title: Fully Connected Neural Network for Binding Affinity Prediction from Molecular Fingerprints

Name: Daniel Eduardo Garzon Otero

The need for accurate prediction of binding affinity is crucial in drug discovery and design. In this project, I aim to improve the accuracy of binding affinity prediction compared to existing methods by using molecular fingerprints and a fully connected neural network. The results of this project could contribute to the development of more effective drugs and therapies.

# Dataset
The PDB_Bind database, which contains information about protein-ligand interactions obtained from the Protein Data Bank (PDB). Specifically, I will use the PDBBind dataset provided by Dr. Camille Bilodeau. The dataset will be processed and prepared for training the neural network model.
![image](https://github.com/danielgarzonotero/PDB_Bind/assets/122416545/fa6b87aa-e9eb-4a39-8e28-a352cdae36b7)


# Goals
Dataset processing: Utilize the pandas library and a Python class to set up and preprocess the dataset.
Deep learning: Train a fully connected neural network model using the preprocessed and engineered data. PyTorch library will be used for implementing the neural network.
Hyperparameter selection: Depending on the model's performance, modify the hyperparameters or adjust the model architecture to improve accuracy.
Model evaluation: Evaluate the trained models using performance metrics such as R-squared, mean absolute error (MAE), root mean squared error (RMSE), and Pearson correlation coefficient. Sklearn and Scipy libraries will be used for calculating these metrics.
Visualization: Utilize the Matplotlib library to visualize the data and interpret the results of the analysis.

# Neural Network Model
A fully connected neural network (FCNN) is defined using PyTorch. The FCNN consists of three fully connected layers with ReLU activation functions and two dropout layers. Dropout is used as a regularization technique to prevent overfitting.

# Results

![image](https://github.com/danielgarzonotero/PDB_Bind/assets/122416545/b6aaa81c-25dc-4e82-a07c-d970475d2b7b)



![image](https://github.com/danielgarzonotero/PDB_Bind/assets/122416545/2bd6a86b-be12-4ab4-a78f-9bf48b96a75f)


![image](https://github.com/danielgarzonotero/PDB_Bind/assets/122416545/43818b39-a49f-43cc-94d2-095f8a698b9b)


# Conclusion 
Based on the results presented, can be concluded that the Daniel NN model outperformed all other models on the test set, achieving the lowest RMSE of 0.5719. This suggests that the Daniel NN model is a good approach for the prediction task. However, it is important to note that the performance of the model on the validation set was not as good as some training routine, which could indicate overfitting.

![image](https://github.com/danielgarzonotero/PDB_Bind/assets/122416545/9fba3186-f439-4dc0-8009-da60fea8be4e)


# Reference
Jha, A.R. (2020). Mastering PyTorch. Packt Publishing



