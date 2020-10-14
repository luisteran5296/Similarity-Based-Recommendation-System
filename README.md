<h1 align="center">Similarity Based Recommendation System</h1> 
_______________________________________________________________________________________________________________________________

<h5 align="center">Luis Terán</h5> 

A recommendation engine is an information filtering system uploading information tailored to users' interests, preferences, or behavioral history on an item. It is able to predict a specific user's preference on an item based on their profile.
With the use of product recommendation systems, the customers are able to find the items they are looking for easily and quickly. A few recommendation systems have been developed so far to find products the user has watched, bought or somehow interacted with in the past.

The recommendation engine is a splendid marketing tool especially for e-commerce and is also useful for increasing profits, sales and revenues in general. That's why personalized product recommendations are so widely used in the retail industry, eleven more highlighting the importance of recommendation engines in the e-commerce industry.

<img src="images/recom2.jpeg" alt="Figure 1" style="width: 600px;"/><p style="text-align:center;font-size: 11px;">Recommendation system</p>

Recommendation systems use a number of different technologies. We can classify these systems into two broad groups:
- Content-based systems: Content-based systems examine properties of the items recommended. For instance, if a Netflix user has watched many cowboy movies, then recommend a movie classified in the database as having the “cowboy” genre.
- Collaborative filtering systems: Collaborative filtering systems recommend items based on similarity measures between users and/or items. The items recommended to a user are those preferred by similar users. 

The aim of this project is to create a similarity based recommendation system using collaborative filtering completely capable of make concrete recommendations of movies according to previous rated books movies the user. The original dataset was obtained from:

</br><center> https://grouplens.org/datasets/movielens/latest/ </center>

This dataset (ml-latest-small) describes 5-star rating and free-text tagging activity from [MovieLens](http://movielens.org), a movie recommendation service. It contains 100836 ratings and 3683 tag applications across 9742 movies. These data were created by 610 users between March 29, 1996 and September 24, 2018. This dataset was generated on September 26, 2018. No demographic information is included. Each user is represented by an id, and no other information is provided.

The data files used in this project were:

- Movies Data File Structure (movies.csv): Movie information is contained in the file `movies.csv`. Each line of this file after the header row represents one movie, and has the (movieId, title, genre) format.
- Ratings Data File Structure (ratings.csv): All ratings are contained in the file `ratings.csv`. Each line of this file after the header row represents one rating of one movie by one user, and has the (userId, movieId, rating, timestamp) format.

The process followed by the project was:
1. Exploratory Data Analysis: 
2. User Item Matrix
3. Similar User Search
4. Recommendation System Implementation
5. Final Evaluation

## 1. Exploratory Data Analysis

### 1.1 General Analysis

First of all, we need to import both datasets (movies & ratings) and take a view of the data. There are 9,742 movies with id, title and genres, also 100,836 ratings of those movies. Before proceeding, we need to take care of some aspects of the dataset:

- Missing values: There were found 0 missing in any of the datasets.
- Duplicates: There were no duplicate observations in any dataset.
- Unnecessary variables: Since the recommender system is based only in previous evaluations, is only necessary to know the evaluation, which movie was evaluated and who made the evaluation ('rating', 'movieId', 'userId').


```python
# All the libraries needed for the project are loaded
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
import warnings

np.random.seed(0)
warnings.filterwarnings('ignore')
```


```python
# Importing movie dataset
movies = pd.read_csv("./data/movies.csv")
movies.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Showing number of rows (9742) and columns (3) of the movies dataset
movies.shape
```




    (9742, 3)




```python
# Looking for null values
movies.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9742 entries, 0 to 9741
    Data columns (total 3 columns):
    movieId    9742 non-null int64
    title      9742 non-null object
    genres     9742 non-null object
    dtypes: int64(1), object(2)
    memory usage: 152.3+ KB
    


```python
# Importing rating dataset
ratings = pd.read_csv("./data/ratings.csv")
ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
      <td>964982703</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>4.0</td>
      <td>964981247</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>6</td>
      <td>4.0</td>
      <td>964982224</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>47</td>
      <td>5.0</td>
      <td>964983815</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1</td>
      <td>50</td>
      <td>5.0</td>
      <td>964982931</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Showing number of rows (100836) and columns (4) of the movies dataset
ratings.shape
```




    (100836, 4)




```python
# Looking for null values
ratings.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100836 entries, 0 to 100835
    Data columns (total 4 columns):
    userId       100836 non-null int64
    movieId      100836 non-null int64
    rating       100836 non-null float64
    timestamp    100836 non-null int64
    dtypes: float64(1), int64(3)
    memory usage: 3.1 MB
    


```python
# Looking for duplicated values
ratings.duplicated().sum()
```




    0



### 1.2 Distribution of the ratings dataset

We'll start by understanding what are the most frequent rating evaluations in the ratings dataset and how are they distributed. The evaluation consists of 10 possible values between 0.5 and 5 (0.5 step increasing). We can see that 75% of the reviews are equal or greater than 3 and 50% is greater or equal to 4 stars. The low ratings are not frequent in the dataset and a 4 star rating is the most frequent evaluation.


```python
(pd.DataFrame(ratings['rating']
              .value_counts(sort=False))
              .sort_index()
              .plot(kind='bar', color='#4472C4', figsize=(15,5), alpha=0.6))
plt.title('Rating Distribution\n')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()
```


![png](outputImages/output_13_0.png)



```python
ratings['rating'].describe().to_frame()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>100836.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>3.501557</td>
    </tr>
    <tr>
      <td>std</td>
      <td>1.042529</td>
    </tr>
    <tr>
      <td>min</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>3.500000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
</div>



Now, let's count how those reviews are distributed among the users. We find out that the numbers of ratings by user ranges between 2 and 791, but almost the 75% of the users has only reviewed less than 100 movies.


```python
(pd.DataFrame(ratings.groupby('userId')
              .count()['rating'])
              .plot(kind='hist', color='#4472C4', figsize=(15,5), alpha=0.6))
plt.title('Number of ratings by user')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()
```


![png](outputImages/output_16_0.png)



```python
pd.DataFrame(ratings.groupby('userId').count()['rating']).describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>610.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>165.304918</td>
    </tr>
    <tr>
      <td>std</td>
      <td>269.480584</td>
    </tr>
    <tr>
      <td>min</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>35.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>70.500000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>168.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>2698.000000</td>
    </tr>
  </tbody>
</table>
</div>



One important aspect of the recommendation system is what makes a movie recommendable. One movie could have an average 5 star rating with only two reviews but that doesn't make that movie the most recommendable movie. For that reason, we'll define a threshold, the threshold will the minimum number of reviews for a movie to be considered as a reliable recommendations. We will drop movies below the threshold value so only popular movies are recommend, we don't have an explicit measure of how popular a movie but we can obtain how many reviews have received every movie. 


```python
# Reviews per movie plot
ratesMovies = pd.DataFrame(ratings['movieId'].value_counts(sort=False))
ratesMovies.plot(kind='hist', color='#4472C4', figsize=(15,5), alpha=0.6)
plt.title('Reviews per movie')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()
```


![png](outputImages/output_19_0.png)


As a first view, we realize that most of the movies don't have more than 30 reviews but there is no clear value to select as threshold.


```python
# Reviews per movie plot between 0 and 100
ratesMovies.plot.hist(xlim=(0,100), bins=300, color='#4472C4', figsize=(15,5), alpha=0.6)
plt.title('Number of reviews for movies')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()
```


![png](outputImages/output_21_0.png)


Taking a closer look we see that most of the movies have less than 30 reviews and after this value the frequency remains constant so this value we will be used as the threshold. Then, we need to remove all those reviews that belong to not popular movies. From the 100,836 initial reviews only 57,358 remained.


```python
# The threshold is defined
threshold = 30
```


```python
# Ratings before removing the less popular movies
len(ratings)
```




    100836




```python
# Removing less popular movies
ratesMovies.reset_index(inplace=True)
ratesMovies.columns=['movieId', 'reviews']
ratings = pd.merge(ratings, ratesMovies, on='movieId', how="left").copy()
ratings = ratings[ratings['reviews']>threshold].copy()
```


```python
# Ratings after removing the less popular movies
len(ratings)
```




    57358




```python
# We will also remove not popular movies from the movies dataset
remaining_movies = list(ratings.movieId.values)
movies = movies[movies['movieId'].isin(remaining_movies)].copy()
```

### 1.3 Distribution of the ratings dataset

As we already told, for the system the movie information and how the users have interacted determines how the movies are recommended. This means that the movies and their characteristics are not as important as the ones in the ratings dataset. Nevertheless, we'll take a quick view of the characteristics in the dataset in order to have a full picture of the data we are handling. 
Some extra information can be obtained by extracting the year of release for the movies. From that we can conclude that the majority of the movies in the dataset were released near the year 2000.


```python
# Extracting the year out of the title column
movies['year'] = movies['title'].str.extract(r"\((\d+)\)", expand=False)
movies['year'] = pd.to_numeric(movies['year'])
movies.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>1995</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
      <td>1995</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
      <td>1995</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
      <td>1995</td>
    </tr>
    <tr>
      <td>5</td>
      <td>6</td>
      <td>Heat (1995)</td>
      <td>Action|Crime|Thriller</td>
      <td>1995</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Year of release histogram
movies['year'].plot(kind = 'hist', xlim=(1900,2020), bins=300, color='#4472C4', figsize=(15,5), alpha=0.6)
plt.title('Year of release of movies')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.show()
```


![png](outputImages/output_30_0.png)


Another important characteristic parameter is the movie genre. More than one movie genre can appear for movie description so we will count the number of total occurrences of elements in the Genre column. The most frequent genres are Drama and Comedy, being approximately 4 times more frequent than Fantasy or Horror movies.


```python
# Genre frequency list
genres = list(movies.genres.values)
movie_genres = [movie.split('|') for movie in genres]
movie_genres = [x for sublist in movie_genres for x in sublist]
genres_freq = pd.DataFrame(pd.Series(movie_genres, name='Genre movies').value_counts().head(10))
genres_freq.reset_index(inplace=True)
genres_freq
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Genre movies</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Drama</td>
      <td>355</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Comedy</td>
      <td>345</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Action</td>
      <td>292</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Thriller</td>
      <td>242</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Adventure</td>
      <td>236</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Sci-Fi</td>
      <td>161</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Romance</td>
      <td>159</td>
    </tr>
    <tr>
      <td>7</td>
      <td>Crime</td>
      <td>150</td>
    </tr>
    <tr>
      <td>8</td>
      <td>Fantasy</td>
      <td>120</td>
    </tr>
    <tr>
      <td>9</td>
      <td>Children</td>
      <td>82</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Genre frequency barplot
fig = plt.figure(figsize=(16,6))
ax = sns.barplot(x="index", y="Genre movies", data=genres_freq)
plt.title('Frequency of movies genres', size=18)
plt.xticks(size=12)
plt.xlabel('Movie genre', size=12)
plt.yticks(size=12)
plt.ylabel('Movies count', size=12)
plt.grid(alpha=0.5)
plt.show()
```


![png](outputImages/output_33_0.png)


### 1.4 Top movies

Finally, since this is a real users dataset, just for fun we can see what are the best rated popular movies and what are worst rated popular movies. One interesting fact is that Drama genre is frequent in the top rated movies, also Crime movies, even though there are not frequent, are really well rated. On the other hand, action and comedy movies are frequent genres in the worst rated movies.


```python
# Obtaining mean rating for every movie
sorted_movies = ratings.groupby('movieId').mean()
sorted_movies.sort_values(['rating', 'reviews'], ascending=[False, False], inplace = True)
top10movies = list(sorted_movies.head(10).index)
bottom10movies = list(sorted_movies.tail(10).index)
```


```python
# Top rated movies
movies[movies['movieId'].isin(top10movies)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>277</td>
      <td>318</td>
      <td>Shawshank Redemption, The (1994)</td>
      <td>Crime|Drama</td>
      <td>1994</td>
    </tr>
    <tr>
      <td>602</td>
      <td>750</td>
      <td>Dr. Strangelove or: How I Learned to Stop Worr...</td>
      <td>Comedy|War</td>
      <td>1964</td>
    </tr>
    <tr>
      <td>659</td>
      <td>858</td>
      <td>Godfather, The (1972)</td>
      <td>Crime|Drama</td>
      <td>1972</td>
    </tr>
    <tr>
      <td>686</td>
      <td>904</td>
      <td>Rear Window (1954)</td>
      <td>Mystery|Thriller</td>
      <td>1954</td>
    </tr>
    <tr>
      <td>906</td>
      <td>1204</td>
      <td>Lawrence of Arabia (1962)</td>
      <td>Adventure|Drama|War</td>
      <td>1962</td>
    </tr>
    <tr>
      <td>914</td>
      <td>1213</td>
      <td>Goodfellas (1990)</td>
      <td>Crime|Drama</td>
      <td>1990</td>
    </tr>
    <tr>
      <td>922</td>
      <td>1221</td>
      <td>Godfather: Part II, The (1974)</td>
      <td>Crime|Drama</td>
      <td>1974</td>
    </tr>
    <tr>
      <td>975</td>
      <td>1276</td>
      <td>Cool Hand Luke (1967)</td>
      <td>Drama</td>
      <td>1967</td>
    </tr>
    <tr>
      <td>2226</td>
      <td>2959</td>
      <td>Fight Club (1999)</td>
      <td>Action|Crime|Drama|Thriller</td>
      <td>1999</td>
    </tr>
    <tr>
      <td>6315</td>
      <td>48516</td>
      <td>Departed, The (2006)</td>
      <td>Crime|Drama|Thriller</td>
      <td>2006</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Worst rated movies
movies[movies['movieId'].isin(bottom10movies)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>163</td>
      <td>193</td>
      <td>Showgirls (1995)</td>
      <td>Drama</td>
      <td>1995</td>
    </tr>
    <tr>
      <td>313</td>
      <td>355</td>
      <td>Flintstones, The (1994)</td>
      <td>Children|Comedy|Fantasy</td>
      <td>1994</td>
    </tr>
    <tr>
      <td>379</td>
      <td>435</td>
      <td>Coneheads (1993)</td>
      <td>Comedy|Sci-Fi</td>
      <td>1993</td>
    </tr>
    <tr>
      <td>396</td>
      <td>455</td>
      <td>Free Willy (1993)</td>
      <td>Adventure|Children|Drama</td>
      <td>1993</td>
    </tr>
    <tr>
      <td>607</td>
      <td>762</td>
      <td>Striptease (1996)</td>
      <td>Comedy|Crime</td>
      <td>1996</td>
    </tr>
    <tr>
      <td>1174</td>
      <td>1562</td>
      <td>Batman &amp; Robin (1997)</td>
      <td>Action|Adventure|Fantasy|Thriller</td>
      <td>1997</td>
    </tr>
    <tr>
      <td>1235</td>
      <td>1644</td>
      <td>I Know What You Did Last Summer (1997)</td>
      <td>Horror|Mystery|Thriller</td>
      <td>1997</td>
    </tr>
    <tr>
      <td>1373</td>
      <td>1882</td>
      <td>Godzilla (1998)</td>
      <td>Action|Sci-Fi|Thriller</td>
      <td>1998</td>
    </tr>
    <tr>
      <td>2029</td>
      <td>2701</td>
      <td>Wild Wild West (1999)</td>
      <td>Action|Comedy|Sci-Fi|Western</td>
      <td>1999</td>
    </tr>
    <tr>
      <td>2860</td>
      <td>3826</td>
      <td>Hollow Man (2000)</td>
      <td>Horror|Sci-Fi|Thriller</td>
      <td>2000</td>
    </tr>
  </tbody>
</table>
</div>



## 2. User Item Matrix

After that, we need to create the User Item Matrix, this is a matrix with all the movies in the dataset as columns and all the users as rows, so every element inside the matrix represents what is the evaluation a user has made for a particular movie. In the matrix, a column (specific movie) will contain all the evaluations the users have made for that movie meanwhile every row will contain all the ratings made from the same user.

<img src="images/uim.PNG" alt="Figure 1" style="width: 500px;"/><p style="text-align:center;font-size: 11px;">User Item Matrix</p>

For that, we start by selecting only the useful information from the ratings dataset ('userId', 'movieId', 'rating') and then use a pivot table to define users as rows, movies as columns and the ratings as values of the matrix. Due to the unseen movies, lots of the values will be presented as 'NaN', these 'NaN' values will be replaced by 0 to represent unseen movies since the scale of evaluation minimum value is 0.5.


```python
# Getting only significant columns for the UIM matrix
uim = ratings[['userId', 'movieId', 'rating']].copy()
uim.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>6</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>47</td>
      <td>5.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1</td>
      <td>50</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Creating UIM matrix using pivot
uim = uim.pivot(index='userId', columns='movieId', values='rating')
uim.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>movieId</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>10</th>
      <th>11</th>
      <th>16</th>
      <th>17</th>
      <th>...</th>
      <th>115617</th>
      <th>116797</th>
      <th>119145</th>
      <th>122882</th>
      <th>122886</th>
      <th>122904</th>
      <th>134130</th>
      <th>134853</th>
      <th>139385</th>
      <th>152081</th>
    </tr>
    <tr>
      <th>userId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>5</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 860 columns</p>
</div>




```python
# Filling NA's with 0
uim = uim.fillna(0)
uim.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>movieId</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>10</th>
      <th>11</th>
      <th>16</th>
      <th>17</th>
      <th>...</th>
      <th>115617</th>
      <th>116797</th>
      <th>119145</th>
      <th>122882</th>
      <th>122886</th>
      <th>122904</th>
      <th>134130</th>
      <th>134853</th>
      <th>139385</th>
      <th>152081</th>
    </tr>
    <tr>
      <th>userId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 860 columns</p>
</div>



## 3. Similar User Search

Once we have represented as rows all the reviews made by a particular user we need a way to compare the similarity between a new user and the users from the dataset, for this we will implement Cosine Similarity method.

Cosine similarity measures the similarity between two vectors of an inner product space. It is measured by the cosine of the angle between two vectors and determines whether two vectors are pointing in roughly the same direction. A vector can represent  thousands of attributes, in this case every attribute is the evaluation of a particular movie. Thus, each group of ratings is an object represented by what is called a term-frequency vector, this is every singular row in our User Item Matrix.

<img src="images/cos.png" alt="Figure 1" style="width: 500px;"/><p style="text-align:center;font-size: 11px;">Cosine similarity</p>

The cosine between two vectors (theta) can be also defined as the dot product divided by the module of them. Additionally, the resulting value will change according to the angle between them, as we can see from the picture when the angle is zero the two vectors are overlapping, thus they are really similar. However, when the angle is really open means the vectors are completely different from each other. When we compute the cosine of the angle it gives us values between 0 and 1. 
- As the values of cosine similarity gets closer to 1 (angle = 0°), the more similar the vectors.
- As the values of cosine similarity gets closer to 0 (angle = 90°), the less similar the vectors.

Considering this, we will create a function that iteratively will calculate the similarity from all the users present in the UIM matrix. Then, the function will return a list of the similar users and their respective cosine similarity, these similar users and similarities will be sorted by how similar they are to the studied user. 

We tested the function with the first user of our dataset, as we expected the greatest similarity is with himself. But also is similar with users 597, 366, 311 & 417.


```python
from sklearn.metrics.pairwise import cosine_similarity

# Function for finding similar users
# Receives:
# - user: The ratings made by the user to study
# - uim: The User Item Matrix with all other users to calculate similarity
def findSimilarUsers(user, uim):
    similarity = []
    for i,row in enumerate(uim.values):
        cos = cosine_similarity(user, row.reshape(1, -1))[0][0]
        similarity.append([i, cos])
    temp = pd.DataFrame(similarity, columns=['userId', 'similarity'])
    temp = temp.sort_values(by=['similarity'], ascending=False).copy()
    similar_users = list(temp['userId'].values)
    similarities = list(temp['similarity'].values)

    return (similar_users, similarities)
```


```python
# Test user is created by selecting the first user of the UIM matrix
user = uim.iloc[0].values.reshape(1, -1)
temp = findSimilarUsers(user, uim)
```


```python
# The top 5 similar users are:
temp[0][0:5]
```




    [0, 597, 366, 311, 467]




```python
# The cosine similarity obtained respectively for that users are:
temp[1][0:5]
```




    [0.9999999999999998,
     0.4855035808289292,
     0.4784148761010285,
     0.4756257951946654,
     0.4717509600003482]



## 4. Recommendation System Implementation

Now we have similar users to recommend movies but how can we select which movies to recommend? Well, for that problem we decided to create weight selection.

After we know which users are similar to the studied user, we first verify what movies has the user seen and remove them from the evaluations of the similar users in order to avoid recommending movies that the studied user has already seen.

<img src="images/imp1.png" alt="Figure 1" style="width: 800px;"/><p style="text-align:center;font-size: 11px;"><p style="text-align:center;font-size: 11px;">Selecting unseen movies</p>   
</br> 

From the remaining movies, we'll keep only those movies that similar users liked hence had given a high rate for the movies previously selected. Therefore, before doing this we need to define a value in order to select if a value will be removed or not, for the images presented a threshold of 4 was selected but for the actual project the defined threshold was 5. This will result in a matrix of movies that the studied user hasn't watched classified into the ones that probably would like (1) and the ones that would not (0).
   
<img src="images/imp2.png" alt="Figure 1" style="width: 400px;"/><p style="text-align:center;font-size: 11px;"><p style="text-align:center;font-size: 11px;">Filtering movies according to the rating</p>

<img src="images/imp3.png" alt="Figure 1" style="width: 400px;"/><p style="text-align:center;font-size: 11px;"><p style="text-align:center;font-size: 11px;">Classified matrix</p>

Next, we can use the calculated cosine similarity for every user as the weight of their recommendations, the possible movie recommendations (rows of the matrix) are multiplied by the cosine similarity values previously calculated, this value is different for every user.  
<img src="images/imp4.png" alt="Figure 1" style="width: 600px;"/><p style="text-align:center;font-size: 11px;"><p style="text-align:center;font-size: 11px;">Cosine similarity</p>

<img src="images/imp5.png" alt="Figure 1" style="width: 600px;"/><p style="text-align:center;font-size: 11px;"><p style="text-align:center;font-size: 11px;">Cosine similarity multiplied</p>

Finally, the final score of recommendation for every movie is obtained by the sum of the the individual scores of every user for that movie (1.6 for the image above), in other words, since every column of the matrix represents a singular movie, the sum of values for that column gives us the final score. This way, we are taking into consideration how many times a movie is well rated by a similar user, if similar users have evaluated the same movie with high ratings it will result in a highly recommendable movie. The importance (weigth) that is given for the opinion of every user is given by the similarity to the studied user.

**This is a simplification of the process for understanding purposes, the actual system implementation may change in some details. Nevertheless, the main idea is shown above.**


```python
# Function that returns the most suitable recommendations of movies
# Requires:
# - user: The ratings made by the user to study
# - uim: The User Item Matrix with all other users to calculate similarity
# - recommendations: Number of expected recommendations
# - analyzed_users: Number of similar users to analyze
# - findSimilarUsers: Function that finds similar users from the dataset

def findSimilarMovies(user, uim, recommendations=10, analyzed_users=10):
    # Looking for movies that the user has already seen
    seen = list(uim.columns[list((user>0)[0])])
    
    # Looking for similar users
    similars = findSimilarUsers(user, uim)
    
    # The Dataframe of results is ceated
    scores = pd.DataFrame(columns=['movieId', 'score'])
    dtypes = np.dtype([
          ('movieId', int),          
          ('score', float),          
          ])

    # For the top similar users (analyzed users) the process gets repeated
    for sim_user, sim_score in zip(similars[0][0:analyzed_users], similars[1][0:analyzed_users]):
        # Dropping movies that the studied user has already seen
        rec_movies = uim.iloc[sim_user].drop(seen)
        # Dropping low rated movies
        rec_index = list(rec_movies[rec_movies>4].index.values)        
        
        if (len(rec_index)>0):
            # For every recommended movies of a particualar similar user:
            for movie in rec_index:
                # If the movie is not in the dataframe, it will add it to the dataframe
                if (movie not in scores['movieId'].values):
                    scores.loc[len(scores)] = (movie, 10*sim_score)
                else: 
                # If the movies is already in the dataframe , it will increase its score
                    scores.loc[scores['movieId']==movie, 'score'] += 10*sim_score
        # The values are sorted by the score obtained
        scores.sort_values(by='score', ascending=False, inplace=True)
        
        # There could the case that not enough users are similar
        try:
            scores = scores.head(recommendations)
            scores['movieId'] = scores['movieId'].astype(int)
        except:
            scores['movieId'] = scores['movieId'].astype(int)
    
    # The movie id's and their corresponding scores are returned in lists
    return (list(scores.movieId.values), list(scores.score.values))   
```


```python
# movie Id and Score obtained for that recommendation
ids, scores = findSimilarMovies(user, uim)    
pd.DataFrame({'movieId': ids, 'Score': scores})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>293</td>
      <td>14.059126</td>
    </tr>
    <tr>
      <td>1</td>
      <td>4226</td>
      <td>9.519094</td>
    </tr>
    <tr>
      <td>2</td>
      <td>4973</td>
      <td>9.377341</td>
    </tr>
    <tr>
      <td>3</td>
      <td>112</td>
      <td>4.855036</td>
    </tr>
    <tr>
      <td>4</td>
      <td>6874</td>
      <td>4.855036</td>
    </tr>
    <tr>
      <td>5</td>
      <td>6711</td>
      <td>4.855036</td>
    </tr>
    <tr>
      <td>6</td>
      <td>5669</td>
      <td>4.855036</td>
    </tr>
    <tr>
      <td>7</td>
      <td>4848</td>
      <td>4.855036</td>
    </tr>
    <tr>
      <td>8</td>
      <td>3949</td>
      <td>4.855036</td>
    </tr>
    <tr>
      <td>9</td>
      <td>52973</td>
      <td>4.522305</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Looking for what movies belong to those movie ids
movies[movies['movieId'].isin(ids)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>99</td>
      <td>112</td>
      <td>Rumble in the Bronx (Hont faan kui) (1995)</td>
      <td>Action|Adventure|Comedy|Crime</td>
      <td>1995</td>
    </tr>
    <tr>
      <td>254</td>
      <td>293</td>
      <td>Léon: The Professional (a.k.a. The Professiona...</td>
      <td>Action|Crime|Drama|Thriller</td>
      <td>1994</td>
    </tr>
    <tr>
      <td>2945</td>
      <td>3949</td>
      <td>Requiem for a Dream (2000)</td>
      <td>Drama</td>
      <td>2000</td>
    </tr>
    <tr>
      <td>3141</td>
      <td>4226</td>
      <td>Memento (2000)</td>
      <td>Mystery|Thriller</td>
      <td>2000</td>
    </tr>
    <tr>
      <td>3544</td>
      <td>4848</td>
      <td>Mulholland Drive (2001)</td>
      <td>Crime|Drama|Film-Noir|Mystery|Thriller</td>
      <td>2001</td>
    </tr>
    <tr>
      <td>3622</td>
      <td>4973</td>
      <td>Amelie (Fabuleux destin d'Amélie Poulain, Le) ...</td>
      <td>Comedy|Romance</td>
      <td>2001</td>
    </tr>
    <tr>
      <td>4012</td>
      <td>5669</td>
      <td>Bowling for Columbine (2002)</td>
      <td>Documentary</td>
      <td>2002</td>
    </tr>
    <tr>
      <td>4529</td>
      <td>6711</td>
      <td>Lost in Translation (2003)</td>
      <td>Comedy|Drama|Romance</td>
      <td>2003</td>
    </tr>
    <tr>
      <td>4615</td>
      <td>6874</td>
      <td>Kill Bill: Vol. 1 (2003)</td>
      <td>Action|Crime|Thriller</td>
      <td>2003</td>
    </tr>
    <tr>
      <td>6481</td>
      <td>52973</td>
      <td>Knocked Up (2007)</td>
      <td>Comedy|Drama|Romance</td>
      <td>2007</td>
    </tr>
  </tbody>
</table>
</div>



## 5. Final Evaluation

Finally we've got a movie recommendation system, now we can make predictions of possible recommendations but we still we don't know if the predictions are made correctly. It's hard to really evaluate whether the system is predicting the right movies or not. The best way to evaluate is to make predictions with real users and see of the movies are correctly predicted, or split the data reducing the UIM matrix, cutting some movies the users has liked. A simpler way is by random gender selection, since the movies predicted are not content based, the prediction system never considers the gender of the movie (ratings dataset), so if we create a user that only likes movies that belong to the same genre, we estimate that the predictions made belong to that genre too.

<img src="images/recom.jpg" alt="Figure 1" style="width: 500px;"/><p style="text-align:center;font-size: 11px;">Recommendation system</p>

For this part of the project we'll make this prediction for two genres:
- Comedy movies
- Horror movies
For the creation of the movies, 10 random movies were selected that contain the genre "Comedy"/"Horror" in the movie genre description. Another important fact, is that as the user rates more movies, the higher the recommendation score could be, this is, as the user evaluates more movies the system creates better recommendations, for the cases presented we've only used 10 ratings.
As we expected, the general view states that most of the movies belong to the same genre the user liked or related. Nevertheless, there are some different genres in the recommendations this is because some movies don't have a unique genre, there are some movies that ave multiple genres. But, in a general way, we can see that for most of the predictions, the recommendations are related and are from the same kind.


### 5.1 User that only likes comedy movies


```python
comedy_movies = movies[movies['genres'].str.contains('Comedy', regex=False)].copy()
comedy_movies = comedy_movies.sample(10).copy()
comedy_movies
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>99</td>
      <td>112</td>
      <td>Rumble in the Bronx (Hont faan kui) (1995)</td>
      <td>Action|Adventure|Comedy|Crime</td>
      <td>1995</td>
    </tr>
    <tr>
      <td>1727</td>
      <td>2321</td>
      <td>Pleasantville (1998)</td>
      <td>Comedy|Drama|Fantasy</td>
      <td>1998</td>
    </tr>
    <tr>
      <td>1603</td>
      <td>2145</td>
      <td>Pretty in Pink (1986)</td>
      <td>Comedy|Drama|Romance</td>
      <td>1986</td>
    </tr>
    <tr>
      <td>820</td>
      <td>1080</td>
      <td>Monty Python's Life of Brian (1979)</td>
      <td>Comedy</td>
      <td>1979</td>
    </tr>
    <tr>
      <td>3568</td>
      <td>4886</td>
      <td>Monsters, Inc. (2001)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>2001</td>
    </tr>
    <tr>
      <td>8636</td>
      <td>119145</td>
      <td>Kingsman: The Secret Service (2015)</td>
      <td>Action|Adventure|Comedy|Crime</td>
      <td>2015</td>
    </tr>
    <tr>
      <td>383</td>
      <td>440</td>
      <td>Dave (1993)</td>
      <td>Comedy|Romance</td>
      <td>1993</td>
    </tr>
    <tr>
      <td>6016</td>
      <td>38061</td>
      <td>Kiss Kiss Bang Bang (2005)</td>
      <td>Comedy|Crime|Mystery|Thriller</td>
      <td>2005</td>
    </tr>
    <tr>
      <td>1005</td>
      <td>1307</td>
      <td>When Harry Met Sally... (1989)</td>
      <td>Comedy|Romance</td>
      <td>1989</td>
    </tr>
    <tr>
      <td>5938</td>
      <td>34162</td>
      <td>Wedding Crashers (2005)</td>
      <td>Comedy|Romance</td>
      <td>2005</td>
    </tr>
  </tbody>
</table>
</div>




```python
user1 = []
for col in uim.columns:   
    user1.append(5) if (col in list(comedy_movies['movieId'].values))  else user1.append(0)      
user1 = np.array(user1).reshape(1,-1)
```


```python
print ('User 1 has {} rated movies and {} unseen movies'.format(len(user1[user1==5]), len(user1[user1==0])))
```

    User 1 has 10 rated movies and 850 unseen movies
    


```python
ids, scores = findSimilarMovies(user1, uim)  
sc = pd.DataFrame({'movieId': ids, 'Score': scores}).copy()
recs = movies[movies['movieId'].isin(ids)].copy()
pd.merge(sc, recs).sort_values('Score', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>Score</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>5.089188</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>1995</td>
    </tr>
    <tr>
      <td>1</td>
      <td>356</td>
      <td>3.815754</td>
      <td>Forrest Gump (1994)</td>
      <td>Comedy|Drama|Romance|War</td>
      <td>1994</td>
    </tr>
    <tr>
      <td>2</td>
      <td>6942</td>
      <td>2.687132</td>
      <td>Love Actually (2003)</td>
      <td>Comedy|Drama|Romance</td>
      <td>2003</td>
    </tr>
    <tr>
      <td>3</td>
      <td>7451</td>
      <td>2.687132</td>
      <td>Mean Girls (2004)</td>
      <td>Comedy</td>
      <td>2004</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1215</td>
      <td>2.652919</td>
      <td>Army of Darkness (1993)</td>
      <td>Action|Adventure|Comedy|Fantasy|Horror</td>
      <td>1993</td>
    </tr>
    <tr>
      <td>5</td>
      <td>8641</td>
      <td>2.652000</td>
      <td>Anchorman: The Legend of Ron Burgundy (2004)</td>
      <td>Comedy</td>
      <td>2004</td>
    </tr>
    <tr>
      <td>6</td>
      <td>6188</td>
      <td>2.652000</td>
      <td>Old School (2003)</td>
      <td>Comedy</td>
      <td>2003</td>
    </tr>
    <tr>
      <td>7</td>
      <td>88163</td>
      <td>1.525983</td>
      <td>Crazy, Stupid, Love. (2011)</td>
      <td>Comedy|Drama|Romance</td>
      <td>2011</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1265</td>
      <td>1.525983</td>
      <td>Groundhog Day (1993)</td>
      <td>Comedy|Fantasy|Romance</td>
      <td>1993</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1258</td>
      <td>1.503717</td>
      <td>Shining, The (1980)</td>
      <td>Horror</td>
      <td>1980</td>
    </tr>
  </tbody>
</table>
</div>



### 5.2 User that only like horror movies


```python
horror_movies = movies[movies['genres'].str.contains('Horror', regex=False)].copy()
horror_movies = horror_movies.sample(10).copy()
horror_movies
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2078</td>
      <td>2762</td>
      <td>Sixth Sense, The (1999)</td>
      <td>Drama|Horror|Mystery</td>
      <td>1999.0</td>
    </tr>
    <tr>
      <td>6630</td>
      <td>56174</td>
      <td>I Am Legend (2007)</td>
      <td>Action|Horror|Sci-Fi|Thriller|IMAX</td>
      <td>2007.0</td>
    </tr>
    <tr>
      <td>2641</td>
      <td>3535</td>
      <td>American Psycho (2000)</td>
      <td>Crime|Horror|Mystery|Thriller</td>
      <td>2000.0</td>
    </tr>
    <tr>
      <td>920</td>
      <td>1219</td>
      <td>Psycho (1960)</td>
      <td>Crime|Horror</td>
      <td>1960.0</td>
    </tr>
    <tr>
      <td>5335</td>
      <td>8874</td>
      <td>Shaun of the Dead (2004)</td>
      <td>Comedy|Horror</td>
      <td>2004.0</td>
    </tr>
    <tr>
      <td>1997</td>
      <td>2657</td>
      <td>Rocky Horror Picture Show, The (1975)</td>
      <td>Comedy|Horror|Musical|Sci-Fi</td>
      <td>1975.0</td>
    </tr>
    <tr>
      <td>915</td>
      <td>1214</td>
      <td>Alien (1979)</td>
      <td>Horror|Sci-Fi</td>
      <td>1979.0</td>
    </tr>
    <tr>
      <td>1083</td>
      <td>1407</td>
      <td>Scream (1996)</td>
      <td>Comedy|Horror|Mystery|Thriller</td>
      <td>1996.0</td>
    </tr>
    <tr>
      <td>957</td>
      <td>1258</td>
      <td>Shining, The (1980)</td>
      <td>Horror</td>
      <td>1980.0</td>
    </tr>
    <tr>
      <td>2027</td>
      <td>2699</td>
      <td>Arachnophobia (1990)</td>
      <td>Comedy|Horror</td>
      <td>1990.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
user2 = []
for col in uim.columns:   
    user2.append(5) if (col in list(horror_movies['movieId'].values))  else user2.append(0)      
user2 = np.array(user2).reshape(1,-1)
```


```python
print ('User 2 has {} rated movies and {} unseen movies'.format(len(user2[user2==5]), len(user2[user2==0])))
```

    User 2 has 10 rated movies and 426 unseen movies
    


```python
ids, scores = findSimilarMovies(user2, uim)  
sc = pd.DataFrame({'movieId': ids, 'Score': scores}).copy()
recs = movies[movies['movieId'].isin(ids)].copy()
pd.merge(sc, recs).sort_values('Score', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>Score</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1387</td>
      <td>9.695435</td>
      <td>Jaws (1975)</td>
      <td>Action|Horror</td>
      <td>1975.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>253</td>
      <td>9.543883</td>
      <td>Interview with the Vampire: The Vampire Chroni...</td>
      <td>Drama|Horror</td>
      <td>1994.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1215</td>
      <td>7.747767</td>
      <td>Army of Darkness (1993)</td>
      <td>Action|Adventure|Comedy|Fantasy|Horror</td>
      <td>1993.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1997</td>
      <td>5.851641</td>
      <td>Exorcist, The (1973)</td>
      <td>Horror|Mystery</td>
      <td>1973.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5952</td>
      <td>3.922561</td>
      <td>Lord of the Rings: The Two Towers, The (2002)</td>
      <td>Adventure|Fantasy</td>
      <td>2002.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1252</td>
      <td>3.900174</td>
      <td>Chinatown (1974)</td>
      <td>Crime|Film-Noir|Mystery|Thriller</td>
      <td>1974.0</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1222</td>
      <td>3.891607</td>
      <td>Full Metal Jacket (1987)</td>
      <td>Drama|War</td>
      <td>1987.0</td>
    </tr>
    <tr>
      <td>7</td>
      <td>778</td>
      <td>2.219149</td>
      <td>Trainspotting (1996)</td>
      <td>Comedy|Crime|Drama</td>
      <td>1996.0</td>
    </tr>
    <tr>
      <td>8</td>
      <td>3471</td>
      <td>2.028370</td>
      <td>Close Encounters of the Third Kind (1977)</td>
      <td>Adventure|Drama|Sci-Fi</td>
      <td>1977.0</td>
    </tr>
    <tr>
      <td>9</td>
      <td>79132</td>
      <td>1.863237</td>
      <td>Inception (2010)</td>
      <td>Action|Crime|Drama|Mystery|Sci-Fi|Thriller|IMAX</td>
      <td>2010.0</td>
    </tr>
  </tbody>
</table>
</div>


