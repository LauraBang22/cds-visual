# Assignment 3 in Visual analytics

### Assignment Discription
In this assignment I have solved the following tasks:
- Loads the Tobacco3482 data and generates labels for each image
- Train a classifier to predict document type based on visual features
- Present a classification report and learning curves for the trained classifier
- Your repository should also include a short description of what the classification report and learning curve show.

### Data
The data I have used in this assignment, is the dataset called Tobacco3482, which can be found [here](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg?resource=download).

The model I have used to train the data is the model called VGG16, which you can find more information about [here](https://www.geeksforgeeks.org/vgg-16-cnn-model/).

### Reproducebility 
I have created a requirements.txt file that can be run from the terminal using the code ```pip install -r requirements.txt```

To run the code of the entire script in the terminal you can use the code ```python src/Assignment3.py```

### Results
When running the code, the output is a classification report and a loss curve plot, which can be found in the **out** folder.

In this case I have run the code with 25 epochs. When looking at the classification report that gives me an accuracy of 46%, which I think is pretty good, considering the data is all different types of text and they are only in black and white.

However, wehen looking at the loss curve for the code is shows signs of it beginning to overfit in the last epochs. I could have run the code using only 20 epochs, which would have shown less overfitting. But the curve is not even close to flattening out, so there might be other approaches that could have been better. I might have gotten a better result if I have done some data augmentation, whilst still using the VGG16 model. Or maybe I should have used a different model, that might have worked better on this particular dataset or even trained my own classifier on the data.