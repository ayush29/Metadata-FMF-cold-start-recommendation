# Import the Required Libraries
import numpy as np
import pandas as pd
import math 
import random

# import form user defined libraries
import decision_tree2 as dtree
import optimization as opt

#get the data from TMDB dataset
def getUserFeatureMatrix(ratingcsv,moviefeaturecsv,maxUsers):
    df = pd.read_csv(moviefeaturecsv)
    df.fillna(0,inplace = True)
    df.set_index('id',inplace=True)
    
    df2 = pd.read_csv(ratingcsv,usecols=['userId','movieId','rating'])
    df2.fillna(0,inplace = True)
    df2['userId'] = pd.to_numeric(df2['userId'])
    df2['movieId'] = pd.to_numeric(df2['movieId'])
    df2['rating'] = pd.to_numeric(df2['rating'])
    
    commonMovieIds = df.index.intersection(df2['movieId'].unique(),sort = None).tolist()
    ratingdf = df2[df2['movieId'].isin(commonMovieIds)]
    featuresdf = df[df.index.isin(commonMovieIds)]
    
    
    #dropping users and keeping #maxUsers with most ratings
    users = df2.groupby(['userId'],sort = False).size().sort_values(ascending =False).index.values.tolist()
#     users.sort()
    
    users = users[:maxUsers]
#     print(max(users))
    ratingdf = ratingdf[ratingdf['userId'].isin(users)]
    
    num_users = len(users)
    num_movies = len(commonMovieIds)    
    print(num_users)
    print(num_movies)
    utility_mat = np.zeros((num_users,num_movies))
    for _,row in ratingdf.iterrows():            
        utility_mat[users.index(int(row['userId']))][commonMovieIds.index(int(row['movieId']))] = row['rating']
    
    movieFeaturesMat = featuresdf.as_matrix()
    
    userFeaturesMat = np.matmul(utility_mat, movieFeaturesMat)
    num_genre = 19
    num_prod_comp = 23
    num_prod_ctry = 32
    num_cast = 13
    
    for i in range(userFeaturesMat.shape[0]):
        genre_sum = np.linalg.norm(userFeaturesMat[i][:num_genre],None) + 0.0001        
        userFeaturesMat[i][:num_genre] = userFeaturesMat[i][:num_genre] /genre_sum
        
        cast_sum = np.linalg.norm(userFeaturesMat[i][num_genre:num_genre + num_cast],None) + 0.0001
        userFeaturesMat[i][num_genre:num_genre + num_cast] = userFeaturesMat[i][num_genre:num_genre + num_cast]/cast_sum

        
        prod_com_sum = np.linalg.norm(userFeaturesMat[i][num_genre + num_cast:num_genre + num_cast + num_prod_comp],None) + 0.0001
        userFeaturesMat[i][num_genre + num_cast:num_genre + num_cast + num_prod_comp] = userFeaturesMat[i][num_genre + num_cast:num_genre + num_cast + num_prod_comp] /prod_com_sum
        
        prod_ctry_sum = np.linalg.norm(userFeaturesMat[i][num_genre + num_cast + num_prod_comp: num_genre + num_cast + num_prod_comp + num_prod_ctry],None) + 0.0001
        userFeaturesMat[i][num_genre + num_cast + num_prod_comp: num_genre + num_cast + num_prod_comp + num_prod_ctry] = userFeaturesMat[i][num_genre + num_cast + num_prod_comp:num_genre + num_cast + num_prod_comp + num_prod_ctry]/prod_ctry_sum        

    userFeatureDF = pd.DataFrame(userFeaturesMat, columns = df.columns.tolist())
    userFeatureDF = userFeatureDF.loc[:, (userFeatureDF != 0).any(axis=0)]
  
        
    utilityDF = pd.DataFrame(utility_mat, columns = commonMovieIds )
    utilityDF = utilityDF.loc[:, (utilityDF != 0).any(axis=0)]
    
    return utilityDF,userFeatureDF
    


# get the data from the MovieLens Dataset
# def getRatingMatrix(filename):
#     # Open the file for reading data
#     file = open(filename, "r")

#     while 1:
#         # Read all the lines from the file and store it in lines
#         lines = file.readlines(1000000000)

#         # if Lines is empty, simply break
#         if not lines:
#             break

#         # Create a Dictionary of DIctionaries
#         User_Movie_Dict = {}

#         # Create a list to hold all the data
#         data = []

#         print ("Number of Lines: ", len(lines))

#         # For each Data Entry, get the Y and the Xs in their respective list
#         for line in lines:
#             # Get all the attributes by splitting on '::' and 
#             # use list comprehension to convert string to float
#             list1 = line.split("\n")[0].split("::")
#             list1 = [float(j) for j in list1]

#             # Add to the data
#             data.append(list1)

#         # Convert the data into numpyarray        
#         data_array = np.array(data)

#         # Get the indices of the maximum Values in each column
#         a = np.argmax(data, axis=0)
#         num_users = data[a[0]][0]
#         num_movies = data[a[1]][1]

#         # print "Max values Indices: ", a
#         print ("Number of Users: ", num_users)
#         print ("Number of Movies: ", num_movies)

#         # Creat and initialise Rating Matrix to hold all the rating values
#         ratingMatrix = np.zeros((int(num_users), int(num_movies)))

#         for list1 in data:
#             # print list1[0], " ", list1[1]
#             ratingMatrix[int(list1[0]) - 1][int(list1[1]) - 1] = list1[2]
            
#         # Return both the array and the dict
#         return (User_Movie_Dict, ratingMatrix)


def cf_movie_vector(rating_matrix, user_vectors, movie_vectors, K,lambda_r = 0.02):
    # Stores the user profile vectors
    movie_profiles = np.zeros((len(rating_matrix[0]), K))
    
    count = 0
    for j in range(len(rating_matrix[0])):
        first_term = np.zeros((K, K))
        
        for i in range(len(rating_matrix)):
            if rating_matrix[i][j] > 0:     
                first_term = np.add(first_term, np.outer(user_vectors[i], user_vectors[i]))

        # Take the inverse of the first term
        first_term = np.add(first_term, np.multiply(lambda_r, np.eye(K)))
        first_term = np.linalg.inv(first_term)

        second_term = np.zeros(K)

        for i in range(len(user_vectors)):
            if rating_matrix[i][j] > 0:
                second_term = np.add(second_term, np.multiply(rating_matrix[i][j], user_vectors[i]))

        movie_profiles[count] = np.dot(first_term, second_term)
        count = count + 1

    return movie_profiles


# Function to calculate the RMSE Error between the predicted and actual rating
def getRMSE(Actual_Rating, Predicted_Rating):
    # Calculate the Root Mean Squared Error(RMS)
    rmse = 0.0
    for i in range(len(Actual_Rating)):
        for j in range(len(Actual_Rating[0])):
            if Actual_Rating[i][j] > 0:
                rmse = rmse + pow((Actual_Rating[i][j] - Predicted_Rating[i][j]), 2)

    rms = rmse * 1.0 / len(Actual_Rating)
    rmse = math.sqrt(rmse)

    # Print and return the RMSE
    print ('Root Mean Squared Error(RMS) = ' , rms)
    return rms


# Used to randomly split the data
def random_split(data):
    # Split the data set into 75% and 25%
    SPLIT_PERCENT = 0.75
    
    # Get Random Indices to shuffle the rows around
    indices = np.random.permutation(data.shape[0])

    # Get the number of rows 
    num_rows = len(data[:, 0])

    # Get the indices for training and testing sets
    training_indices, test_indices = indices[: int(SPLIT_PERCENT * num_rows)], indices[int(SPLIT_PERCENT * num_rows) :]

    # return the training and the test set
    return training_indices,test_indices


# Returns the rating Matrix with approximated ratings for all users for all movies using fMf
def alternateOptimization(rating_matrix,feature_matrix, NUM_OF_FACTORS,MAX_DEPTH,ITERATIONS, LAMBDA):
    # Save and print the Number of Users and Movies
    NUM_USERS = rating_matrix.shape[0]
    NUM_MOVIES = rating_matrix.shape[1]
    print ("Number of Users", NUM_USERS)
    print ("Number of Movies", NUM_MOVIES)
    print ("Number of Latent Factors: ", NUM_OF_FACTORS)

    # Create the user and item profile vector of appropriate size.
    # Initialize the item vectors randomly, check the random generation
    user_vectors = np.zeros((NUM_USERS, NUM_OF_FACTORS), dtype=float)
    movie_vectors = np.random.rand(NUM_MOVIES, NUM_OF_FACTORS)

    i = 0    
    
    print ("Entering Main Loop of alternateOptimization")

    decTree = dtree.Tree(dtree.Node(None, 1), rating_matrix, NUM_OF_FACTORS,MAX_DEPTH)

    # Do converge Check
    while i < ITERATIONS:
        # Create the decision Tree based on movie_vectors
        #print "Creating tree..."

        decTree = dtree.Tree(dtree.Node(None, 1), rating_matrix, NUM_OF_FACTORS,MAX_DEPTH)
        
        #print "Creating Tree.. for i = ", i

        decTree.fitTree(decTree.root, rating_matrix,feature_matrix, movie_vectors, NUM_OF_FACTORS)

        #print "Tree Created for i = ", i
        #print "Getting the user vectors from tree"

        # Calculate the User vectors using dtree
        user_vectors = decTree.getUserVectors(feature_matrix, NUM_OF_FACTORS)

        #print "Got the user vectors from the decisionTree"
        #print "Optimizing movie_vectors.."

        # Optimize Movie vector using the calculated user vectors
        movie_vectors = cf_movie_vector(rating_matrix, user_vectors, movie_vectors, NUM_OF_FACTORS, LAMBDA)

        #print "movie_vectors Optimized"
        
        # Calculate Error for Convergence check
        i = i + 1

    # return the completed rating matrix    
    return (decTree, movie_vectors.T)


def printTopKMovies(test, predicted, K = 2):
    # Gives top K (2) recommendations
    K = 2

    zero_list = []
    movie_list = []

    for i in range(len(test)):
        for j in range(len(test[0])):
            if test[i][j] == 0:
                zero_list.append(predicted[i][j])
                movie_list.append(j)

            zero_array = np.array(zero_list)
            movie_array = np.array(movie_list)

            args = np.argsort(zero_array)
            movie_array = movie_array[args]

            print ("Top Movies Not rated by the user")
            print (movie_array[0:K-1])

def run(max_users=1500,Lambda=0.08, iterations=5,max_depth=5,num_factors=10):
    MAX_NUM_USERS = max_users
    
    # Get the Data
    ratingDF,featureDF =getUserFeatureMatrix("./the-movies-dataset/ratings.csv","./tmdb_movies_cross_features.csv",MAX_NUM_USERS)
    rating_matrix = ratingDF.as_matrix()
    feature_matrix = featureDF.as_matrix()
    
#     (User_Movie_Dict, data) = getRatingMatrix("ratings.dat")

    print ("Dimensions of the Dataset: ", rating_matrix.shape)
    
    # Split the data 75-25 into training and testing dataset
    (train_indices, test_indices) = random_split(rating_matrix)
    train_rating = rating_matrix[train_indices,:]
    test_rating = rating_matrix[test_indices,:]
    
    train_feature = feature_matrix[train_indices,:]
    test_feature = feature_matrix[test_indices,:]
    print ("Dimensions of the Training Set: ", train_rating.shape)
    print ("Dimensions of the Testing Set: ", test_rating.shape)

    # Split the testing dataset 75-25 into answer and evaluation dataset
    (answer_indices, evaluation_indices) = random_split(test_rating)
    answer_rating = test_rating[answer_indices,:]
    evaluation_rating = test_rating[evaluation_indices,:]
    
    answer_feature = test_feature[answer_indices,:]
    evaluation_feature = test_feature[evaluation_indices,:]
    print ("Dimensions of the Answer Set: ", answer_rating.shape)
    print ("Dimensions of the Evaluation Set: ", evaluation_rating.shape)
    
    # Set the number of Factors
    NUM_OF_FACTORS = num_factors
    MAX_DEPTH = max_depth
    ITERATIONS = iterations
    LAMBDA = Lambda
    (decisionTree, movie_vector) = alternateOptimization(train_rating,train_feature, NUM_OF_FACTORS,MAX_DEPTH,ITERATIONS, LAMBDA)
    
    user_vectors = decisionTree.getUserVectors(test_feature, NUM_OF_FACTORS)

    Predicted_Rating = np.dot(user_vectors, movie_vector)
    print ("Predicted_Rating for Test: ", Predicted_Rating) 
    print ("Test Rating: ", test_rating)
    rmse = getRMSE(test_rating, Predicted_Rating)
    print ("RMSE on Testing: ", rmse)
    return rmse
    
    
#     pd.DataFrame(test_rating, index = ratingDF.index.values[test_indices],columns = ratingDF.columns).to_csv("true_rating_TMDB.csv")
    
#     pd.DataFrame(Predicted_Rating, index = ratingDF.index.values[test_indices],columns = ratingDF.columns).to_csv("prediction_rating_TMDB.csv")

    # Top K new recommendations:
    # printTopKMovies(test, Predicted_Rating, 1)

    # Testing
    # testMatrix = np.array([[0, 5, 3, 4, 0], [3, 4, 0, 3, 1], [3, 0, 4, 0, 2], [4, 4, 4, 3, 0], [3, 5, 0, 4, 0]])
    # (indices_like, indices_dislike, indices_unknown) = splitUsers(data, 2293)
    
    # like = data[indices_like]
    # dislike = data[indices_dislike]
    # unknown = data[indices_unknown]

    # print "Dimensions of the Like: ", like.shape[0], like.shape[1]
    # print "Dimensions of the Dislike: ", dislike.shape[0], dislike.shape[1]
    # print "Dimensions of the Unknown: ", unknown.shape[0], unknown.shape[1]
    
    # Get the decision tree and the item profile vectors using alternate optimization
    #(decisionTree, item_vector) = alternateOptimization(train)
    # train = test = np.array([[0, 5, 3, 4, 0], [3, 4, 0, 3, 1], [3, 0, 4, 0, 2], [4, 4, 4, 3, 0], [3, 5, 0, 4, 0]])
    
