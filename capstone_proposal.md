# Machine Learning Engineer Nanodegree
## Capstone Proposal
Bárbara Barbosa
February 10st, 2019

## Proposal
_(approx. 2-3 pages)_

### Domain Background
_(approx. 1-2 paragraphs)_

Display advertising is a billion dollar effort and one of the central uses of machine learning on the Internet. However, its data and methods are usually kept under lock and key. In this research competition, CriteoLabs is sharing a week’s worth of data for you to develop models predicting ad click-through rate (CTR). Given a user and the page he is visiting, what is the probability that he will click on a given ad? Display advertising is ubiquitous but there is hardly any publicly available dataset to benchmark ML algorithms in that domain. Description obtained from [Kaggle challenge](https://www.kaggle.com/c/criteo-display-ad-challenge#description)

Since I will start to work with marketing data at my job this is the perfect challenge for me, to test my skills.

### Problem Statement
_(approx. 1 paragraph)_

The goal of this problem is to benchmark the most accurate ML algorithms for CTR estimation, so the company can determine what user they show an ad. There are 3 good solutions (the first 3 winners) for the challenge.

### Datasets and Inputs
_(approx. 2-3 paragraphs)_

The dataset can be obtained at (http://labs.criteo.com/2014/02/download-kaggle-display-advertising-challenge-dataset/) and consists of a weak of data from Criteo, who has:

 - 6 data centers on 3 continents
 - More than 7000 servers and a few tens of Petabytes of storage on our HPC cluster
 - 30B HTTP requests and 3B unique banners displayed per day
 - Peak traffic of 800K HTTP requests per second
 - Respond to bids in 80ms or less, 24/7

The datasets and file descriptions are obtained from kaggle: https://www.kaggle.com/c/criteo-display-ad-challenge/data

*Data fields*
**Label** - Target variable that indicates if an ad was clicked (1) or not (0).
**I1-I13** - A total of 13 columns of integer features (mostly count features).
**C1-C26** - A total of 26 columns of categorical features. The values of these features have been hashed onto 32 bits for anonymization purposes.
The semantic of the features is undisclosed.

When a value is missing, the field is empty.

### Solution Statement
_(approx. 1 paragraph)_

The solution for this problem consistis build a classifier algorithm that can predict the probability of an ad being clicked. This will be achivied using an AdaBoost classifier optimized by logloss, the evaluation metric. Also some feature enginering, combining the features will be performed. The evaluation will be made using logloss, the smaller the better.

### Benchmark Model
_(approximately 1-2 paragraphs)_

Our benchmark model will be the third place winner of the competition. That used a logistic regression with a quadratic/polynomial feature generalization. In it's paper (file:///home/barbarabarbosa/Downloads/Display-ad-challenge-Song.pdf) tha winner is not very specific in what features are combined to generate that combinations.

Instead of a regular logistic regression the winner used an implementation called Vowpal wabbit, since it only keeps the feature weights in RAM. The best model uses quadratic generalization for feature groups. Also the best model was implemented using an L2 optimization as the suggest solution.

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

The metric for evaluation will be log-loss (the smaller the better). The benchmark model has a log-loss of 0.44610 but since this is a kaggle competition, and the main objective is to win the score by super minor changes at the leaderboard, we will focus on obtain at least a model with 0.50 of log-loss. Which will give me a position of 500th at the leaderboard.

The logloss is defined by the equation bellow [from](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html). Except that in sickit-learn the equation is negative, so the greater the better (as other metrics).

-log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))

where yt is the true label and yp is the predicted probability.

This metric is frequently used for classification models where the final result is a probability. Log loss increases as the predicted probability diverges from the actual label.

### Project Design
_(approx. 1 page)_

This project will be divided in 5 sections:

    1 - Exploratory Data Analysis

    2 - Feature engineering

    3 - Model Training

    4 - Model evaluation and comparison

    5 - Final model selection and comprehension

#### 1 - Exploratory Data Analysis
Since the data isn't explained, we will have to check every feature for distribution analysis and a better undestanding of the problem.

This step will gave a better enlightment of what to do on the next step, as well as which features to use. The only information we have by now is that the dataset has some numerical and categorical features.

#### 2 - Feature engineering
In this section we hope to explore some possible relations with features (discovered in section 1), possible applying some normalization (like a log) or some combination (multiplication, division, sum) between features to apply in this step.

Besides that we can possible apply PCA for the numerical features, using the PCAs attributes with most variance and the categorical features all togheter.

Since the dataset is really big (4.6 Gb in a ziped file) I also plan to use Dask to perform the paralel computing I will need to run the feature engineer of this dataset in my machine.

#### 3 - Model Training

Since this is a classification problem and we have a great amount of data we plan to focus the training in 2 models

 - A logistic regression (because the 3rd place in this competition used this model)
 - A tree based algorithm, more specifically an Adaboost. (I would rather use a LightGBM, because is a gradient algorithm, and is one of the winner solutions, though)

Considering the logistic regression aproach we will use L2 regularization to remove useless atributes.
For the Adaboost we will have to take care with the amount of trees we will use, because the size of the dataset can be a huge problem and make the solution impracticable.

#### 4 - Model evaluation and comparison

Since the metric evaluated at the challenge is algorithm loss, we will use this metric to compare the models. We can also verify the AUC and accuracy, but the main metric will be log loss.

#### 5 - Final model selection and comprehension

As mentioned in section 4, we will use log-loss for model selection and if possible, we will perform a feature analysis of the winner model. Since the dataset is anonymised, this is not an easy task to perform, and maybe couldn't be possible.

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
