#Earthquake_Prediction_Using_Deep_Learning

In this project, we utilize an open-source global earthquake dataset available from the US Geological Survey (USGS) containing over 19700 records with details of past earthquakes, including geolocation, depth, magnitude, and timestamp. The data spans multiple decades and provides detailed seismic activity records.

We aim to develop a machine learning model to classify earthquake events based on historical seismic attributes using a dense neural network. The primary goal is to learn from patterns in earthquake magnitude, depth, and location to identify high-magnitude events.

The dataset is first preprocessed by combining and converting date and time fields into a Unix timestamp, and handling missing values. Then, we select meaningful features like latitude, longitude, depth, and magnitude for modeling. Outliers are handled, and the data is scaled using StandardScaler to improve convergence during training.

We use a simple fully connected neural network (Multilayer Perceptron) consisting of two hidden layers with ReLU activation followed by a softmax layer for binary classification. The model is compiled using categorical crossentropy as the loss function, Adam as the optimizer, and accuracy and mean absolute error as the evaluation metrics.

The model is trained using an 80/20 train-test split. Finally, we evaluate the model on the test set and visualize earthquake distributions and the predicted outcomes using matplotlib and seaborn. This work demonstrates a basic yet effective approach to using deep learning for classifying seismic events based on historical data.
