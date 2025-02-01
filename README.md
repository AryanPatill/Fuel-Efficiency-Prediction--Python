<h2> # Fuel-Efficiency-Prediction--Python </h2>

Fuel Efficiency Prediction - Regression using The Auto MPG dataset ( You can use any other dataset also )
In a regression problem, we aim to predict the output of a continuous value, like a price or a probability. Contrast this with a classification problem, where we aim to select a class from a list of classes (for example, where a picture contains an apple or an orange, recognizing which fruit is in the picture).

This notebook uses the classic Auto MPG Dataset and builds a model to predict the fuel efficiency of late-1970s and early 1980s automobiles. To do this, we'll provide the model with a description of many automobiles from that time period. This description includes attributes like: cylinders, displacement, horsepower, and weight.

DataSet - _https://archive.ics.uci.edu/dataset/9/auto+mpg_

**Stepwise Instructions:**

•	**Install Dependencies** – Installs seaborn for visualization. <br>
•	**Import Libraries** – Loads pandas, matplotlib, seaborn, and tensorflow.<br>
•	**Load Dataset** – Downloads the Auto MPG dataset from UCI.<br>
•	**Read & Prepare Data** – Assigns column names and loads the dataset.<br>
•	**Handle Missing Values** – Removes rows with missing data.<br>
•	**Process Categorical Data** – One-hot encodes the Origin column (USA, Europe, Japan).<br>
•	**Split Dataset** – 80% training, 20% test data.<br>
•	**Visualize Data** – Uses seaborn.pairplot() to analyze feature relationships.<br>
•	**Compute Statistics** – Generates descriptive stats for numerical columns.<br>
•	**Separate Features & Labels **– Extracts MPG as the target variable.<br>
•	**Normalize Data** – Standardizes features using Z-score normalization.<br>
•	**Build Model** – Defines a Neural Network with:<br>
     2 hidden layers (64 neurons, ReLU activation)
     1 output layer (regression output)
     RMSprop optimizer & MSE loss
•	**Initialize Model** – Creates the model instance.<br>
•	**View Model Summary **– Displays the architecture.<br>
•	**Test Initial Predictions** – Runs the model on sample input.<br>
•	**Train Model** – Trains for 1000 epochs with a validation split.<br>
•	**Monitor Training** – Stores training history and prints progress.<br>
•	**Plot Training Performance** – Graphs MAE and MSE over epochs.<br>
•	**Use Early Stopping **– Stops training when validation loss stagnates.<br>
•	**Evaluate Model** – Computes Mean Absolute Error (MAE) on test data.<br>
•	**Make Predictions** – Predicts MPG values and plots true vs. predicted.<br>
•	**Analyze Errors** – Visualizes prediction errors using a histogram.<br>

Some small information about the learning from the code and information about it.

**In[12] **- Normalize the data
Look again at the train_stats block above and note how different the ranges of each feature are.
It is good practice to normalize features that use different scales and ranges. Although the model might converge without feature normalization, it makes training more difficult, and it makes the resulting model dependent on the choice of units used in the input.
Note: Although we intentionally generate these statistics from only the training dataset, these statistics will also be used to normalize the test dataset. We need to do that to project the test dataset into the same distribution that the model has been trained on

**In[13]** -This normalized data is what we will use to train the model.
Caution: The statistics used to normalize the inputs here (mean and standard deviation) need to be applied to any other data that is fed to the model, along with the one-hot encoding that we did earlier. That includes the test set as well as live data when the model is used in production.

**In[14]** - The Model
Build the model
Let's build our model. Here, we'll use a Sequential model with two densely connected hidden layers, and an output layer that returns a single, continuous value. The model building steps are wrapped in a function, build_model, since we'll create a second model, later on.

**In[20] **- This graph shows little improvement, or even degradation in the validation error after about 100 epochs. Let's update the model.fit call to automatically stop training when the validation score doesn't improve. We'll use an EarlyStopping callback that tests a training condition for every epoch. If a set amount of epochs elapses without showing improvement, then automatically stop the training.


**Conclusion**

This notebook introduced a few techniques to handle a regression problem.

  Mean Squared Error (MSE) is a common loss function used for regression problems (different loss functions are used for classification problems).
  Similarly, evaluation metrics used for regression differ from classification. A common regression metric is Mean Absolute Error (MAE).
  When numeric input data features have values with different ranges, each feature should be scaled independently to the same range.
  If there is not much training data, one technique is to prefer a small network with few hidden layers to avoid overfitting.
  Early stopping is a useful technique to prevent overfitting.


