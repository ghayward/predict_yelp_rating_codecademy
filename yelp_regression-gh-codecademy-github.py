#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import pandas
import pandas as pd
#read in the yelp data
businesses = pd.read_json('yelp_business.json', lines=True)
reviews = pd.read_json('yelp_review.json', lines=True)
users = pd.read_json('yelp_user.json', lines=True)
checkins = pd.read_json('yelp_checkin.json', lines=True)
tips = pd.read_json('yelp_tip.json', lines=True)
photos = pd.read_json('yelp_photo.json', lines=True)


# In[2]:


#make the data more viewable
pd.options.display.max_columns = 60
pd.options.display.max_colwidth = 500


# In[3]:


#let's check out the businesses data
businesses.head()


# In[4]:


#let's check out the reviews data
reviews.head()


# In[5]:


#let's check out the user data
users.head()


# In[6]:


#let's check out the checkin data
checkins.head()


# In[7]:


#let's check out the tips data
tips.head()


# In[8]:


#let's check out the photos data
photos.head()


# In[9]:


#how many different businesses are in the dataset
print(businesses.business_id.nunique())
#or since each line is a unique business
print(len(businesses))


# In[10]:


#what are the different features (columns) in the reviews dataset
print(reviews.columns)


# In[11]:


#what is the range of values for features (columns) in the users datset
users.describe()


# In[12]:


#looking up an example: what is the yelp rating ('stars') of business_id = 5EvUIR4IzCWUOm0PsUZXjA ?
#let's use boolean indexing
businesses[businesses['business_id'] == '5EvUIR4IzCWUOm0PsUZXjA']['stars']
#i believe the first number (at least in jupyter notebooks) is the row index, and the secod number is the star count


# In[13]:


#we now want to merge (join) the data to get all the information together in one table
#we can merge (join) on the unique identifier: business_id, which is on all the tables
df = pd.merge(businesses, reviews, how='left', on='business_id')
print(len(df))


# In[14]:


#merge the rest of the tables
df = pd.merge(df, users, how='left', on='business_id')
df = pd.merge(df, checkins, how='left', on='business_id')
df = pd.merge(df, tips, how='left', on='business_id')
df = pd.merge(df, photos, how='left', on='business_id')
#we are using left merges (left joins) so we don't lose any rows if a subsquent table doesn't have the item
#now we confirm the length is still the same as the original, 188593
print(len(df))
#and let's check the columns
print(df.columns)


# In[15]:


#the below columns are not continuous numbers or binaryies...these will be ineffecetive for regression purposes
#we therefore move to remove them
features_to_remove = ['address','attributes','business_id','categories','city','hours','is_open','latitude','longitude','name','neighborhood','postal_code','state','time']
#dropping from the data frame; the inplace=True parameter should ensure that change is made on the live table (and not make a duplciate)
df.drop(labels=features_to_remove, axis=1, inplace=True)


# In[16]:


#we need to remove the nulls to make sure the regression model runs smoothly
#let's the check the data frame (table) for any N/As (NaNs or nulls)
df.isna().any()
#we can see a few columns with missing values


# In[17]:


#as we look at the columns with nulls, we can see they are geneally counts
#weekday checkins, weekend checkins, average tip length, number of tips, average caption length, number of pics
#we can thus basically conclude that if it's null, then that yelp business didn't have that category
#and that makes it effecively 0. example: if number of pics is null...they had 0 pics
#so we can just change those particular nulls to 0s to clean up our data
#we can do this with the .fillna() command
df.fillna({'weekday_checkins':0,
           'weekend_checkins':0,
           'average_tip_length':0,
           'number_tips':0,
           'average_caption_length':0,
           'number_pics':0},
          inplace=True) #and we use inplace=True "since we want to perform our changes in place and not return a new DataFrame"

#and let's do a check again
df.isna().any()


# In[18]:


#We are going to consider the dependant variable to be the 'stars' columns
#We want to know which other variables, 'indepedent', can have some impact on the stars column (so the dependent variable)
#We want to use the .corr() function in pandas to check which independent variables have the strongest correlation with
#the dependant varibale.
#Please note that this can be positive of negative
#A correlation of 1 is perfect positive correlation, and a coorelation of -1 is perfect negative correlation
#A correlation of 0 shows there is no correlation. 
#So that is our spectrum, so to speak. 
#We basically want to find out which columns appear to be the most important to investigate further, before we go deep
df.corr()


# In[19]:


#we can see in the above correlation that average_review_sentiment, average_review_length, and average_review_age 
#appear to correlate the most with the star rating on the yelp reviews on the business
#it looks like, on the other hand, one of the least correlative was the number_useful_votes (which, I think is like how many people found the review useful)
#we want to take a look at the relationship between each of these independent variables and the dependent variables
#by the way, I almost think a better lingo would be 'impact variables' (for independent variables) and 'impacted variables' for (dependent variables), althought this may not be the best because someone might think it implies causation
#also by the way, here's a note from Codecademy on what a 'review sentiment' is:
#####"What is average_review_sentiment, you ask? average_review_sentiment is the average sentiment score for\
#####all reviews on a business' Yelp page. The sentiment score for a review was calculated using the sentiment\
#####analysis tool VADER. VADER uses a labeled set of positive and negative words, along with codified rules of\
#####grammar, to estimate how positive or negative a statement is. Scores range from -1, most negative, to +1, most\
#####positive, with a score of 0 indicating a neutral statement. While not perfect, VADER does a good job at guessing\
#####the sentiment of text data!"
#And now back to this: 
#We want to see the relationship between these chosen possible signals and our dependent variable (maybe we can call it the 'result variable')

#we import pyplot from matplotlib for the scatter plot
from matplotlib import pyplot as plt


#looking at average_review_sentiment against stars 
plt.scatter(df.average_review_sentiment,df.stars, alpha=0.05)
plt.xlabel("Average Review Sentiment")
plt.ylabel("Star Rating")
plt.title("Yelp Average Review Sentiment vs. Star Rating")
plt.savefig('avg_review_sentiment_vs_star_yelp.png')
plt.show()


# In[20]:


#looking at average_review_length against stars here
plt.scatter(df.average_review_length,df.stars, alpha=0.05)
plt.xlabel("Average Review Length (characters)")
plt.ylabel("Star Rating")
plt.title("Yelp Average Review Length vs. Star Rating")
plt.savefig('avg_review_length_vs_star_yelp.png')
plt.show()


# In[21]:


#looking at average_review_age against stars here
plt.scatter(df.average_review_age,df.stars, alpha=0.05)
plt.xlabel("Average Review Age (days)")
plt.ylabel("Star Rating")
plt.title("Yelp Average Review Age vs. Star Rating")
plt.savefig('avg_review_age_vs_star_yelp.png')
plt.show()


# In[22]:


#looking at number_useful_votes against stars here
plt.scatter(df.number_useful_votes,df.stars, alpha=0.05)
plt.xlabel("Reviews's Number of 'Useful' Votes")
plt.ylabel("Star Rating")
plt.title("Yelp Review's Number of 'Useful' Votes vs. Star Rating")
plt.savefig('num_useful_votes_vs_star_yelp.png')
plt.show()


# In[23]:


#Codecademy questioN; why do you think `average_review_sentiment` correlates so well with Yelp rating?
#Looks like the more postivei someone's langauge is, the higher the star rating.\
#This makes sense. A high star review, is more likely to have more positive and happy text.


# In[24]:


#as we begin to build the multiple linear regressions model, we want to select a subset of the dataframe
#containing the most relevant (greatest correlations) from what we checked above
#thats's average sentiment, average length, and average age of the review
#for now, since average sentiment is so high, from a correlation perspective, we will want to check it our
#a bit more deeply later, and in the meantime, we will focus on average length and average age

features = df[['average_review_length', 'average_review_age']]

#and then we need to get ratings into it's own data frame since it's the dependent variable
ratings = df['stars']


# In[25]:


#the modeling is about to begin!
#to do it, we need to bring in the Scikit Learn library:
from sklearn.model_selection import train_test_split
#and the first step will be to slice up the data into training and testing sets, 80% and 20% of the whole respectively
#this gives us a way to test the model
#we will set the test size to 0.2 for 20%, and we'll set the random state to 1 (this random state number\
#basically initiates the random number generator, and the import thing is that you use the same random state number\
#throughout your work in a project, I think (See: https://stackoverflow.com/questions/42191717/python-random-state-in-splitting-dataset/42197534))

X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size = 0.2, random_state = 1)
#now we've split the data into training and testing sets! =)


# In[26]:


#time to import the linear regression command from Scikit-Learn
from sklearn.linear_model import LinearRegression
#we name the model (here we named it 'model' - creative lol)
model = LinearRegression()
#now we fit the model (seems like fashion industry talk)
model.fit(X_train,y_train)#this is fitting the model to the x's (thought we call it big X) and y's



# In[27]:


#let's see how we did and check the r^2
#the r^2 helps us determine how much variance in the dependent variable can be related to our independent variables
model.score(X_train, y_train)


# In[28]:


model.score(X_test, y_test)


# In[29]:


#An 8% R^2 is too low. Looks like we'll need to probably improve the model by adding more relevant features.


# In[30]:


#let's at least check the correlations (these would be the m's in the y=mx+b line of best)
#this next line of code was recommended by Codecademy
#basically it takes the features, zips them, sorts them in most predictive ot least predictive, then appliest the .coef() function to them
#via a lambda function (which is 'a way to define a function in a single line of code' -Codecademy)
####"After all that hard work, we can finally take a look at the coefficients on our different features!\
####The model has an attribute .coef_ which is an array of the feature coefficients determined by fitting\
####our model to the training data. To make it easier for you to see which feature corresponds to which\
####coefficient, we have provided some code in the cell that zips together a list of our features with the\
###coefficients and sorts them in descending order from most predictive to least predictive."

sorted(list(zip(['average_review_length','average_review_age'],model.coef_)),key = lambda x: abs(x[1]),reverse=True)


# In[31]:


#now let's use the mode to predict on the 20% test set and compare the predictions to the actual numbers for the 20% 
#test set
y_predicted = model.predict(X_test)
#then let's plot it to see how it does
plt.scatter(y_test, y_predicted)
plt.ylabel('Y Predicted')
plt.xlabel('Y Actual')
plt.ylim(1,5) #"set the y-limits of the current axes." (https://matplotlib.org/api/_as_gen/matplotlib.pyplot.ylim.html)
plt.title('Looking at how the model performed')
plt.savefig('looking_at_model_1.png')
plt.show()
#we would like to see something looking more like y=x (so a line at 45 degree angle)
#thus, as we sort of knew from seeing the 8% R^2 score, this is not the best model YET
#time for us to think deeper about average review sentiment haha


# In[32]:


#we are going to come up with some new ways of putting together features (columns) in the hopes that we can get a 
#better model
##subset of only average review sentiment (codecademy idea)
sentiment = ['average_review_sentiment']


# In[33]:


##subset of all features that have a response range [0,1] (codecademy idea)
binary_features = ['alcohol?','has_bike_parking','takes_credit_cards','good_for_kids','take_reservations','has_wifi']


# In[34]:


## subset of all features that vary on a greater range than [0,1] (codecademy idea)
numeric_features = ['review_count','price_range','average_caption_length','number_pics','average_review_age','average_review_length','average_review_sentiment','number_funny_votes','number_cool_votes','number_useful_votes','average_tip_length','number_tips','average_number_friends','average_days_on_yelp','average_number_fans','average_review_count','average_number_years_elite','weekday_checkins','weekend_checkins']


# In[35]:


##all features (codecademy idea)
all_features = binary_features + numeric_features


# In[36]:


# subset of features that might relate to the business becoming bigger and established
hit_big_leagues_features = ['average_number_friends', 'average_number_fans', 'average_days_on_yelp', 'review_count']


# In[37]:


#these next lines of code where suggested by Codecademy
#I will comment how they are working

#we import numpy
import numpy as np

#take a list of features to model as a parameter
#we can pass through the features list we made above
def model_these_features(feature_list):
    
    #this tells us to pull all the values for the 'stars' column
    ratings = df.loc[:,'stars']
    #this tells us to pull all the value for whatever columns we've selected in the above features's list
    features = df.loc[:,feature_list]
    
    #this is the spliting of the data from SK-learn
    X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size = 0.2, random_state = 1)
    
    # Codecademy: "don't worry too much about these lines, just know that they allow the model to work when
    # we model on just one feature instead of multiple features. Trust us on this one :)"
    # Me: The reshape command with parameters (-1,1) means that for however many rows we have, make sure I 
    # only have 1 column. You need to do this for the X dependents so that they is all the x's are properly set
    # up to an array for each row.
    # basically if you run a .shape, and you get length that is less than 2, than the length is 1 and it's 1 dimensional
    # like this -> numpy.array([1, 2, 3, 4, 5]) ... believe it or not, that's 5 rows, 0 columns
    # its python for a 1D, single index array
    # you then need to reshape it so you can get a column.... you'd really want [[1],[2],[3],[4],[5]], and for that
    # you want to run a .reshape(-1,1)
    # for more on the shape stuf see:
    ##### https://www.hackerrank.com/challenges/np-shape-reshape/problem
    ##### https://stackoverflow.com/questions/22053050/difference-between-numpy-array-shape-r-1-and-r
    if len(X_train.shape) < 2:
        X_train = np.array(X_train).reshape(-1,1)
        X_test = np.array(X_test).reshape(-1,1)
    
    #this just creates our model, and begins to fit it
    model = LinearRegression()
    model.fit(X_train,y_train)
    
    #this then gets an R^2 for our training part of the model and our testing part of the model
    print('Train Score:', model.score(X_train,y_train))
    print('Test Score:', model.score(X_test,y_test))
    
    # print the model features and their corresponding coefficients, from most predictive to least predictive
    ## Me: this part is some Codecademy magic: 
    ## basically, it is taking the list:
    ## first read this to understand 'key' --? https://stackoverflow.com/questions/13669252/what-is-key-lambda/13669294
    ## so here is what is happening
    ## each column (feature) in the feature list is an independent variable
    ## when we run model.coef_ you get the m slope for each of those variables in a best fit
    ## so that is the is the 'if you increase this independent variable by 1, the dependent variabble will decrease/increase by m'
    ## When we zip it together we get the independent variable followed by the coefficieint all in one bracket 'row' [ feature, coeff ]
    ## we have them in a bracket through the list command
    ## then we sort it
    ## but we apply a lambda to show hot to sort it 
    ## the lamda says for each x...which is the list now (from the list function)
    ## take the second element (because counting starts at 0) and take the absolute value of it
    ## so that is the absolute value of m, and we reverse sort that (reverse = true)
    ## also remember the syntax for sorted() .... Syntax : sorted(iterable, key, reverse)
    ## and key is how you set up custom sorting
    ## read more about that here: https://www.geeksforgeeks.org/sorted-function-python/
    ## so this will give you an "R" like, list of which features have the most impact on the dependent variable
    ## and it will list them in descending order (biggest first)...(Ascending order is the defualt, so it resverses it)
    print(sorted(list(zip(feature_list,model.coef_)),key = lambda x: abs(x[1]),reverse=True))
    
    # This takes the model that was trained on the 80%, and applies it to the 20% test x values for the 'predicted' y's
    y_predicted = model.predict(X_test)
    
    # This now graphs the actual y's from the test on the x axis, and the predicted y's from the model on the y axis
    # in a perfect model this would all lie on y=x, and it would make a 45 degree angle line
    plt.scatter(y_test,y_predicted)
    plt.xlabel('Yelp Rating')
    plt.ylabel('Predicted Yelp Rating')
    plt.ylim(1,5)
    plt.show()


# In[38]:


# create a model on sentiment here
model_these_features(sentiment)


# In[39]:


# create a model on all binary features here
model_these_features(binary_features)


# In[40]:


# create a model on all numeric features here
model_these_features(numeric_features)


# In[41]:


# create a model on all features here
model_these_features(all_features)


# In[42]:


# create a model on your feature subset here
model_these_features(hit_big_leagues_features)


# In[43]:


#looks like the best model so far was all_features
#let's print again to remind ourselves of what they were:
print(all_features)


# In[44]:


#this now takes all features, and re-trains our model on that...we are using much more firepower now than just age, and length of the review
features = df.loc[:,all_features]
ratings = df.loc[:,'stars']
X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size = 0.2, random_state = 1)
model = LinearRegression()
model.fit(X_train,y_train)


# In[45]:


#here, recommended by Codeademy is a selection using the boolean indexing technique from the .describe command. 
#we just wanted the mean, ax, and min for each feature (column)
pd.DataFrame(list(zip(features.columns,features.describe().loc['mean'],features.describe().loc['min'],features.describe().loc['max'])),columns=['Feature','Mean','Min','Max'])


# In[46]:


#here now we are going to input how we think or hope our restaurant would do for all these independent variables
#and we will see what the model would spit out for the star rating...
#if not sure of anything, we can just put in the mean
#we can tell what is binary, from if the max is 1
danielles_delicious_delicacies = np.array([1,1,1,0,1,1,31,1,3,1.5,1175,600,0.75,16,18,44,46,6,105,2000,12,122,1,45,50]).reshape(1,-1)


# In[47]:


#and after all our work, here is our prediction for a star rating, if we can hit our goals:
model.predict(danielles_delicious_delicacies)


# In[48]:


#looks like we can get about 4 stars!
#not bad for Yelp, where so many people posting mean reviews lol

