from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType
from tabulate import tabulate

def computeCosineSimilarity(data):
    # Compute xx, xy, and yy columns
    pairScores = data.selectExpr("movie1", "movie2", 
                                 "rating1 * rating1 as xx", 
                                 "rating2 * rating2 as yy", 
                                 "rating1 * rating2 as xy")

    # Calculate numerator, denominator, and numPairs
    calculateSimilarity = pairScores.groupBy("movie1", "movie2").agg(
        F.sum("xy").alias("numerator"),
        (F.sqrt(F.sum("xx")) * F.sqrt(F.sum("yy"))).alias("denominator"),
        F.count("xy").alias("numPairs")
    )

    # Calculate score and select necessary columns
    result = calculateSimilarity.withColumn("score",
        F.when(F.col("denominator") != 0, F.col("numerator") / F.col("denominator")).otherwise(0)
    ).select("movie1", "movie2", "score", "numPairs")

    return result

def getMovieName(movieNames, movieId):
    # Filter movieNames DataFrame for the specified movieId
    movie = movieNames.filter(F.col("movieID") == movieId).select("movieTitle").first()

    return movie.movieTitle if movie else None

if __name__ == "__main__":
    # Initialize Spark session
    spark = SparkSession.builder.appName("MovieSimilarities").master("local[*]").getOrCreate()

    # Define schema for movie names and ratings data
    movieNamesSchema = StructType([
        StructField("movieID", IntegerType(), True),
        StructField("movieTitle", StringType(), True)
    ])

    moviesSchema = StructType([
        StructField("userID", IntegerType(), True),
        StructField("movieID", IntegerType(), True),
        StructField("rating", IntegerType(), True),
        StructField("timestamp", LongType(), True)
    ])

    # Load movie names and ratings data
    movieNames = spark.read.option("sep", "|").option("charset", "ISO-8859-1").schema(movieNamesSchema).csv("ml-100k/u.item")
    ratings = spark.read.option("sep", "\t").schema(moviesSchema).csv("ml-100k/u.data").select("userId", "movieId", "rating")

    # Generate movie pairs and calculate cosine similarity
    moviePairs = ratings.alias("ratings1").join(ratings.alias("ratings2"), 
                                                (F.col("ratings1.userId") == F.col("ratings2.userId")) & 
                                                (F.col("ratings1.movieId") < F.col("ratings2.movieId"))) \
                                        .select(F.col("ratings1.movieId").alias("movie1"), 
                                                F.col("ratings2.movieId").alias("movie2"), 
                                                F.col("ratings1.rating").alias("rating1"), 
                                                F.col("ratings2.rating").alias("rating2"))
    
    # Compute cosine similarity and cache the result
    moviePairSimilarities = computeCosineSimilarity(moviePairs).cache()

    # Specify quality thresholds for recommendations
    scoreThreshold = 0.97
    coOccurrenceThreshold = 50

    # Example: Specify movie ID for which to find similar movies
    movieID = 50

    # Filter similar movies based on thresholds and specified movie ID
    filteredResults = moviePairSimilarities.filter(
        ((F.col("movie1") == movieID) | (F.col("movie2") == movieID)) &
        (F.col("score") > scoreThreshold) &
        (F.col("numPairs") > coOccurrenceThreshold)
    )

    # Sort and retrieve top similar movies
    results = filteredResults.sort(F.col("score").desc()).take(10)

    # Display top similar movies for the specified movie ID
    movieTitle = getMovieName(movieNames, movieID)
    if movieTitle:
        similar_movies_data = []
        for result in results:
            similarMovieID = result.movie1 if result.movie2 == movieID else result.movie2
            similarMovieTitle = getMovieName(movieNames, similarMovieID)
            if similarMovieTitle:
                similar_movies_data.append([similarMovieTitle, result.score, result.numPairs])

        # Print table using tabulate
        print(f"Top 10 similar movies for {movieTitle}:")
        print(tabulate(similar_movies_data, headers=["Movie Title", "Score", "Strength"]))
    else:
        print(f"Movie with ID {movieID} not found.")

    spark.stop()  # Stop Spark session
