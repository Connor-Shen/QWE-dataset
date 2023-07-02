# QWE-dataset
In this project I tried to explore the QWE dataset and build a model to predict the potential churning customers. EDA and business analysis were also conducted apart from building the predict model.
In detail, first I implemented data-preprocessing and EDA towards QWE. EDA helps me better understand the dataset and I also found some potential but important features of the dataset, which significantly helped me in the procedure of model prediction.
In the part of churn prediction, I used models including:
1. Logistic Regression
2. Random Forest
3. SVM
4. Neural Network
5. Neural Network with GAN
GAN (Generative Adversarial Network) is a framework that consists of two neural networks: a generator network and a discriminator network. The generator network generates synthetic data samples, while the discriminator network learns to distinguish between real and fake data. GAN can generate synthetic samples that closely resemble the real data. The generator learns the complex data distribution, allowing it to produce samples that are more realistic and representative of the minority class.

Eventually I reached the highest AUC score of 0.77. Data Insights and business analysis were also conducted based on that.
