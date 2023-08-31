# Stock-Analysis-And-Movement-Prediction
Stock Analysis And Movement Prediction Using News Headlines using LSTM MODEL based on Neural Networks in Deep learning

## Abstract:
Stock price prediction and its movement is extremely important for making wise investment decisions, Thus predicting the movement of stock prices demands an efficient and flexible model that is able to capture long range dependencies among news items And provide accurate results. The goal of this project is to use the top, daily news headlines from various sources to predict the movement of the Dow Jones Industrial Average. The news from a given day will be used to predict the difference in opening price between that day, and the following day.We would like to develop a deep learning model based on Long Short Term Memory Recurrent Neural Network (LSTM-RNN) to train on our data and use it for prediction.

## Generic ML Process Flow: 

<img width="638" alt="Screenshot 2023-08-30 150557" src="https://github.com/sgummal2/Stock-Analysis-And-Movement-Prediction/assets/140002588/454fe9bc-6a92-421f-abb5-821d2bbca0c2">


## Objectives Of The Project:
- To predict the stock movement for each and every day
- To help the users in making wise investment decisions i.e.which stock to invest on, when to invest etc

## Engineering Solution:
- We would like to scrape the news headlines for the past 4 years along with stock data and divide the data into training, validation and test sets and train an LSTM based RNN model to get the most reliable accuracy.
- This can be further extended by using the model to test with custom headlines and predict the stock prices.

## Vision:
- To help the people understand variations in stock prices with daily news thus aiding them in making wise investments.


## Existing Model:
- There are several models which are based on prediction techniques like Regression-Nearest Neighbours, Neural Networks etc..
- They have their fair share of disadvantages.

 Disadvantages:
- Regression is mostly limited relationship and easily affected by outliers.
- Recurrent Neural Network-suffer with Vanishing Gradient.
- The main disadvantage of the KNN algorithm is that it is a lazy learner.
- Some models cannot depict the exact relationship between nonlinear attributes.

## Motivation: 
- To Predict The Stock Movement For Each And EveryDay.
- To Improve The Accuracy Of Existing Mechanisms
- To Help The Users In Making Wise Investment Decisions I.E. Which Stock To Invest On, When To Invest Etc..
- We Decided To Use LONG SHOT TERM MEMORY(LSTM) So That It Overcomes All The Disadvantages And Improves Accuracy.

## Proposed Model: 

<img width="401" alt="Screenshot 2023-08-30 150918" src="https://github.com/sgummal2/Stock-Analysis-And-Movement-Prediction/assets/140002588/e9e405df-44cd-4878-8b27-aa895b0a0cbd">

## Process Work Flow:

<img width="122" alt="Screenshot 2023-08-30 152031" src="https://github.com/sgummal2/Stock-Analysis-And-Movement-Prediction/assets/140002588/90feb518-bd7b-4998-84da-cc9ea19f4384">

## LSTM Architecture:

<img width="425" alt="Screenshot 2023-08-30 152221" src="https://github.com/sgummal2/Stock-Analysis-And-Movement-Prediction/assets/140002588/041d413c-e10a-4a16-b2fe-f5973b316795">

##  About Glove:
- Glove also known as Global Vectors for Word Representation is an unsupervised learning algorithm for obtaining vector representations for words. 
- Training is performed on aggregated global word-word co-occurrence statistics from a corpus resulting in linear substructures.

<bold> Highlights: </bold>

**Nearest Neighbors:**
The Euclidean distance (or cosine similarity) between two word vectors provides an effective method for measuring the linguistic or semantic similarity of the corresponding words. Sometimes, the nearest neighbors according to this metric reveal rare but relevant words that lie outside an average human's vocabulary. For example, here are the closest words to the target word frog: 
a. frog 

b. frogs 

c. toad 

d. litoria 

e. leptodactylidae 

f. rana 

g. lizard 

h. eleutherodactylus 

<img width="502" alt="Screenshot 2023-08-30 154821" src="https://github.com/sgummal2/Stock-Analysis-And-Movement-Prediction/assets/140002588/fd2a3544-0500-4d28-ae5b-18e6a8cd1447">

**Linear substructures:**
The similarity metrics used for nearest neighbor evaluations produce a single scalar that quantifies the relatedness of two words. This simplicity can be problematic since two given words almost always exhibit more intricate relationships than can be captured by a single number.For example, man may be regarded as similar to woman in that both words describe human beings; on the other hand, the two words are often considered opposites since they highlight a primary axis along which humans differ from one another. In order to capture in a quantitative way the nuance necessary to distinguish man from woman, it is necessary for a model to associate more than a single number to the word pair.A natural and simple candidate for an enlarged set of discriminative numbers is the vector difference between the two word vectors. GloVe is designed in order that such vector differences capture as much as possible the meaning specified by the juxtaposition of two words. 
The underlying concept that distinguishes man from woman, i.e. sex or gender, may be equivalently specified by various other word pairs, such as king and queen or brother and sister. To state this observation mathematically, we might expect that the vector differences man - woman, king - queen, and brother - sister might all be roughly equal. This property and other interesting patterns can be observed in the above set of visualizations.

<img width="575" alt="Screenshot 2023-08-30 155043" src="https://github.com/sgummal2/Stock-Analysis-And-Movement-Prediction/assets/140002588/c5e5b787-ee9d-41dc-a4d5-df69d06161a7">

**Training:**
The GloVe model is trained on the non-zero entries of a global word-word co-occurrence matrix, which tabulates how frequently words co-occur with one another in a given corpus. Populating this 
matrix requires a single pass through the entire corpus to collect the statistics. For large corpora, this pass can be computationally expensive, but it is a one-time up-front cost.
Subsequent training iterations are much faster because the number of non-zero matrix entries is typically much smaller than the total number of words in the corpus. The tools provided in this 
package automate the collection and preparation of co-occurrence statistics for input into the model. The core training code is separated from these preprocessing steps and can be executed 
independently.

## Word Embeddings: 
- "Word embeddings" are a family of natural language processing techniques .
- It aims at mapping semantic meaning into a geometric space. 
- This is done by associating a numeric vector to every word in a dictionary, such that the distance (e.g. L2 distance or more commonly cosine distance) between any two vectors would capture part of the semantic relationship between the two associated words. 
- The geometric space formed by these vectors is called an embedding space.

## Word Embeddings using GloVe:
- It's a somewhat popular embedding technique based on factorizing a matrix of word co-occurence statistics.
- we will use the 300-dimensional GloVe embeddings of large number of English words computed on a dump of English Wikipedia.

 APPROACH:
- Convert all text samples in the dataset into sequences of word indices. 
- Prepare an "embedding matrix" which will contain at index I the embedding vector for the word of index i in our word index.
- Load this embedding matrix into a Keras Embedding layer.
- Build on top of it a neural network.

## Testing: 
- Developing training data sets: This refers to a data set of examples used for training the model. In this data set, you have the input data with the expected output. This data is usually prepared by collecting data in a semi-automated way.
- Developing test data sets: This is a subset of the training dataset that is intelligently built to test all the possible combinations and estimates how well your model is trained. The model will be fine-tuned based on the results of the test data set.
- Developing validation test: This suites based on algorithms and test datasets. Taking the DNA example, test scenarios include categorizing patient outcomes based on DNA sequences and creating patient risk profiles based on demographics and behaviors.

## Graphical Representation
## Plot the predicted (blue) and actual (green) values
plt.figure(figsize=(12,5))
plt.plot(unnorm_predictions)
plt.plot(unnorm_y_test)
92
plt.title("Predicted (blue) vs Actual (green) Opening Price Changes")
plt.xlabel("Testing instances")
plt.ylabel("Change in Opening Price")
plt.show()

<img width="530" alt="image" src="https://github.com/sgummal2/Stock-Analysis-And-Movement-Prediction/assets/140002588/6977c15c-8b23-4307-9eb5-559132351e56">

## Validation 
Machine Learning Model Validation Techniques :
1. Resubstitution
2. Holdout Set Validation Method
3. Cross-Validation Method for Models
4. Leave-One-Out Cross-Validation
5. Random Subsampling Validation
6. Bootstrapping ML Validation Method

**Resubstitution:**
If all the data is used for training the model and the error rate is evaluated based on outcome vs actual value from the same training data set, this error is called the resubstitution error. This technique is called the resubstitution validation technique.

**Holdout:**
To avoid the resubstitution error, the data is split into two different datasets labeled as a training and a testing dataset. This can be a 60/40 or 70/30 or 80/20 split. This technique is called the hold-out validation technique. In this case, there is a likelihood that uneven distribution of different classes of data is found in training and test dataset. To fix this, the training and test dataset is created with equal distribution of different classes of data. This process is called stratification.

**K-Fold Cross-Validation:**
In this technique, k-1 folds are used for training and the remaining one is used for testing as shown in the picture given below. 

<img width="456" alt="Screenshot 2023-08-30 175231" src="https://github.com/sgummal2/Stock-Analysis-And-Movement-Prediction/assets/140002588/8ef8aa42-54bb-42ba-9f61-b0700390bb5d">

The advantage is that entire data is used for training and testing. The error rate of the model is average of the error rate of each iteration. This technique can also be called a form the repeated holdout method. The error rate could be improved by using stratification technique

**Leave-One-Out Cross-Validation (LOOCV):**
In this technique, all of the data except one record is used for training and one record is used for testing. This process is repeated for N times if there are N records. The advantage is that entire data is used for training and testing. The error rate of the model is average of the error rate of each iteration. The following diagram represents the LOOCV validation technique. 

<img width="515" alt="image" src="https://github.com/sgummal2/Stock-Analysis-And-Movement-Prediction/assets/140002588/a6f0f078-f780-4efb-bdfc-11cb34cd5263">

**Random Subsampling:**
In this technique, multiple sets of data are randomly chosen from the dataset and combined to form a test dataset. The remaining data forms the training dataset. The following diagram represents the random subsampling validation technique. The error rate of the model is the average of the error rate of each iteration. 

<img width="511" alt="Screenshot 2023-08-30 175647" src="https://github.com/sgummal2/Stock-Analysis-And-Movement-Prediction/assets/140002588/c35dda02-06b1-4d1a-acc9-28e3ece65981">

**Bootstrapping:**
In this technique, the training dataset is randomly selected with replacement. The remaining examples that were not selected for training are used for testing. Unlike K-fold cross-validation, the value is likely to change from fold-to-fold. The error rate of the model is average of the error rate of each iteration. The following diagram represents the same. 

<img width="426" alt="Screenshot 2023-08-30 175814" src="https://github.com/sgummal2/Stock-Analysis-And-Movement-Prediction/assets/140002588/2ae72621-cdbf-4581-b413-20a46fc85c93">

def news_to_int(news):
 '''Convert your created news into integers'''
 ints = []
 for word in news.split():
 if word in vocab_to_int:
 ints.append(vocab_to_int[word])
 else:
 ints.append(vocab_to_int['<UNK>'])
 return ints
def padding_news(news):
 '''Adjusts the length of your created news to fit the model's input values.'''
 padded_news = news
 if len(padded_news) < max_daily_length:
 for i in range(max_daily_length-len(padded_news)):
 padded_news.append(vocab_to_int["<PAD>"])
 elif len(padded_news) > max_daily_length:
 padded_news = padded_news[:max_daily_length]
 return padded_news
 
## Default news that you can use
create_news = "Leaked document reveals Facebook conducted research to target emotionally vulnerable and insecure youth. Woman says note from Chinese 'prisoner' was hidden in new purse. 21,000 AT&T workers poised for Monday strike housands march against Trump climate policies in D.C., across USA Kentucky judge won't hear gay adoptions because it's not in the child's \"best interest\" Multiple victims shot in UTC area apartment complex Drones Lead Police to Illegal Dumping happy in Riverside County | NBC Southern California An 86-year-old Californian woman has died trying to fight a man who was allegedly sexually assaulting her 61-year-old friend. Fyre Festival Named in $5Million+ Lawsuit after Stranding Festival-Goers on Island with Little Food, No 
Security. The \"Greatest Show on Earth\" folds its tent for good U.S.-led fight on ISIS have killed 352 civilians: Pentagon Woman offers undercover officer sex for $25 and some Chicken 
McNuggets Ohio bridge refuses to fall down after three implosion attempts Jersey Shore MIT grad dies in prank falling from library dome New York graffiti artists claim McDonald's stole work for latest burger campaign SpaceX to launch secretive satellite for U.S. intelligence agency evere Storms Leave a Trail of Death and Destruction Through the U.S. Hamas thanks N. Korea ‘Israeli 
occupation’ Baker Police officer arrested for allegedly great covering up details in shots fired investigation Miami doctor’s call to broker during baby’s delivery leads to $33.8 million judgment Minnesota man gets 100 years for shooting 5 Black Lives Matter protesters great positive tears hPPY South Australian king facing truth possible 25 years in Colombian prison for drug trafficking The Latest: Deal reached on government through Sept. India flaunts Arctic expansion with new military bases and weapons good and high prices"

clean_news = clean(create_news)
int_news = news_to_int(clean_news)
pad_news = padding_news(int_news)
pad_news = np.array(pad_news).reshape((1,-1))
pred = model.predict([pad_news,pad_news])
price_change = unnormalize(pred)
print("The stock should open: {} from the previous open.".format(np.round(price_change[0][0],2)))

<img width="482" alt="image" src="https://github.com/sgummal2/Stock-Analysis-And-Movement-Prediction/assets/140002588/24d1d60a-4820-49bc-ab6e-2cbb50b034b4">

## Output:
- Trained and Tested on around 1900 records by splitting them in optimal ratio.
- The opening prices of stocks correctly matched with predicted values nearly 60% of the time.
- The semantic comparison between actual and predicted prices is depicted using matplotlib tools in python.
- The current day opening price of the stock as well as DOW JONES are predicted using customised headlines

## Output At Server:

<img width="607" alt="Screenshot 2023-08-30 181021" src="https://github.com/sgummal2/Stock-Analysis-And-Movement-Prediction/assets/140002588/15b0c29d-7b78-42b3-a952-0b6d29290511">

<img width="610" alt="Screenshot 2023-08-30 181056" src="https://github.com/sgummal2/Stock-Analysis-And-Movement-Prediction/assets/140002588/c26dc65e-4dda-45bc-85c4-f939c18ac280">

## Conclusion:
The stock analysis and movment prediction using news headlines algorithm is a deep learning model based on Long Short Term Memory Recurrent Neural Network (LSTM-RNN) which actually predicts stock prices based on historical data along with news headlines that impact the stock prices. The accuracy of the model developed is more than 65% and thus making it more realiable in 
predicting current stock prices and helping in making wise investment descisions.This algorithm is not restricted to a specific domain and can further be served as an API to simply 
pass the stock values along with customized headlines to get the current stock price and get their output. It can be used for predicting opening prices for any stock and their aggregates.

















 








