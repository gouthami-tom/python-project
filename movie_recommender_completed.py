import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer as count_vector
from sklearn.metrics.pairwise import cosine_similarity as cos_sim


def getTitle(index):
	return data_set[data_set.index == index]["title"].values[0]


def getIndex(title):
	return data_set[data_set.title == title]["index"].values[0]


def combineFeatures(row):
	try:
		return row['keywords'] + " " + row['cast'] + " " + row["genres"] + " " + row["director"]
	except:
		print("Error:", row)


def movieSuggestionsExtractor(movieName):
	# Get index of this movie from its title
	index = getIndex(movieName)
	suggestions = list(enumerate(cosine_sim[index]))

	# Get a list of similar movies in descending order of similarity score
	sorted_suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)

	# Print titles of first 10 movies
	for element in sorted_suggestions[:10]:
		print(getTitle(element[0]))


# Reading Data set
data_set = pd.read_csv("movie_dataset.csv")

# Select Features
features = ['keywords', 'cast', 'genres', 'director']

# Create a column in DF which combines all selected features
for feature in features:
	data_set[feature] = data_set[feature].fillna('')

data_set["combined_features"] = data_set.apply(combineFeatures, axis=1)

# Create count matrix from this new combined column
cv = count_vector()
count_matrix = cv.fit_transform(data_set["combined_features"])

# Compute the Cosine Similarity based on the count_matrix
cosine_sim = cos_sim(count_matrix)

movie_user_likes = input("Enter movie name for suggestions of similar kind : ")

if movie_user_likes in data_set.title.tolist():
	movieSuggestionsExtractor(movie_user_likes)
else:
	print("Movie not found in database :( \nPlease try with some other movie")
