# Install and Load libraries
if (!require(tidyverse)) install.packages('tidyverse'); library(tidyverse)
if (!require(caret)) install.packages('caret'); library(caret)
if (!require(data.table)) install.packages('data.table'); library(data.table)
if (!require(lubridate)) install.packages('lubridate'); library(lubridate)


###### Download and wrangle MovieLens Data ######

# Download MovieLens data
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# Generate ratings data frame
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

# Generate movies data frame
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# Set classes of variables in movies data frame
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

# Join movies data to ratings data
movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Remove unnecessary variables from global environment
rm(dl, ratings, movies, test_index, temp, movielens, removed)



###### Data Exploration ######

# Distribution of ratings (prop table and histogram)
ratings_distribution <- prop.table(table(edx$rating))

hist_rating <- hist(edx$rating, main = "Histogram of Ratings (edx)", xlab = "Rating")

# Distribution of movie IDs (histogram)
hist_movie <- hist(edx$movieId, main = "Histogram of Movie IDs (edx)", xlab = "movieId")

# Determine most highly rated movie 
highest_rated <- edx %>% group_by(title) %>% summarize(title, movieId, number_of_ratings = n()) %>% 
  arrange(-number_of_ratings) %>% head(1)

# Correlation between average rating per movie, user, date, genre and overall rating
correlations <- edx %>% group_by(movieId) %>%
  mutate(avg_rating_movie = mean(rating)) %>%
  ungroup() %>% group_by(userId) %>%
  mutate(avg_rating_user = mean(rating)) %>%
  ungroup() %>% group_by(genres) %>%
  mutate(avg_rating_genre = mean(rating)) %>%
  ungroup() %>% group_by(round_date(as_datetime(timestamp), "week")) %>%
  mutate(avg_rating_date = mean(rating)) %>% ungroup() %>%
  summarize(r_movie = cor(avg_rating_movie, rating),
            r_user = cor(avg_rating_user, rating),
            r_genre = cor(avg_rating_genre, rating),
            r_date = cor(avg_rating_date, rating))



###### Generate a Regularized Movie Effects Model ######
### Adjust for movie, user, genre, and date effects ###

# Generate RMSE Function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Split edx into training and test sets using the same method above
# Keep 90% of edx for training
set.seed(1, sample.kind="Rounding") 
edx_test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)

edx_temp <- edx[edx_test_index,]
edx_train <- edx[-edx_test_index,]

edx_test <- edx_temp %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

edx_removed <- anti_join(edx_temp, edx_test)
edx_train <- rbind(edx_train, edx_removed)

rm(edx_temp, edx_removed, edx_test_index)

# Generate a function that builds a model, makes predictions, and returns the RMSE
model_reg <- function(l, train, test) {
  
  # Determine average rating of all movies in the training set
  mu <- mean(train$rating)
  
  # Determine movie effect
  b_m <- train %>%
    group_by(movieId) %>%
    summarize(b_m = sum(rating - mu)/(n()+l)) 
  
  # Determine user effect
  b_u <- train %>% 
    left_join(b_m, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_m - mu)/(n()+l))
  
  # Determine genre effect
  b_g <- train %>%
    left_join(b_m, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_m - b_u - mu)/(n()+l))
  
  # Determine date effect, rounded by week
  b_d <- train %>%
    left_join(b_m, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_g, by="genres") %>%
    mutate(date = round_date(as_datetime(timestamp), "week")) %>%
    group_by(date) %>%
    summarize(b_d = sum(rating - b_m - b_u - b_g - mu)/(n()+l))
  
  # Make predictions on the testing set by joining the four effects above
  predicted_ratings <- test %>% 
    mutate(date = round_date(as_datetime(timestamp), "week")) %>%
    left_join(b_m, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_d, by = "date") %>%
    mutate(pred = mu + b_m + b_u + b_g + b_d) %>% .$pred
  
  # Return the RMSE 
  return(RMSE(test$rating, predicted_ratings))
}

# Use cross-validation to determine a lambda 
lambdas <- seq(4, 6, 0.25)

# Generate predictions and the RMSE for each lambda 
rmses <- vector(length = length(lambdas))
set.seed(1, sample.kind = "Rounding")
for (i in 1:length(lambdas)) {
  rmses[i] = model_reg(lambdas[i], edx_train, edx_test)
}

# Plot the lambdas versus the RMSEs
qplot(lambdas, rmses)

# Determine the lambda that generates the smallest RMSE
min_rmse <- min(rmses)
lambda <- lambdas[which.min(rmses)]



###### Make predictions on the validation set ######
# Copy the model and lambda selected above and return the final RMSE

mu_edx <- mean(edx$rating)

movie_effect <- edx %>%
  group_by(movieId) %>%
  summarize(b_m = sum(rating - mu_edx)/(n()+lambda)) 

user_effect <- edx %>% 
  left_join(movie_effect, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_m - mu_edx)/(n()+lambda))

genre_effect <- edx %>%
  left_join(movie_effect, by="movieId") %>%
  left_join(user_effect, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_m - b_u - mu_edx)/(n()+lambda))

date_effect <- edx %>%
  left_join(movie_effect, by="movieId") %>%
  left_join(user_effect, by="userId") %>%
  left_join(genre_effect, by="genres") %>%
  mutate(date = round_date(as_datetime(timestamp), "week")) %>%
  group_by(date) %>%
  summarize(b_d = sum(rating - b_m - b_u - b_g - mu_edx)/(n()+lambda))

predicted_ratings_validation <- validation %>% 
  mutate(date = round_date(as_datetime(timestamp), "week")) %>%
  left_join(movie_effect, by = "movieId") %>%
  left_join(user_effect, by = "userId") %>%
  left_join(date_effect, by = "date") %>%
  left_join(genre_effect, by = "genres") %>%
  mutate(pred = mu_edx + b_m + b_u + b_g + b_d) %>% .$pred

# Report Final RMSE 
validation_rmse <- RMSE(validation$rating, predicted_ratings_validation)
validation_rmse



###### Save data as .Rdata for Markdown Report ######

save(ratings_distribution, file = "ratings_distribution.Rdata")
save(hist_rating, file = "hist_rating.Rdata")
save(hist_movie, file = "hist_movie.Rdata")
save(highest_rated, file = "highest_rated.Rdata")
save(correlations, file = "correlations.Rdata")
save(lambdas, file = "lambdas.Rdata")
save(rmses, file = "rmses.Rdata")
