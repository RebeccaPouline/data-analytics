# Load required libraries
library(tidyverse)
library(magrittr)
library(caret)
library(ggplot2)
library(GGally)
library(sqldf)
library(RSQLite)
library(randomForest)
library(gbm)

# Load the dataset
teledata <- read_csv("./data/teleCust1000t.csv")

# View the dataset structure
View(teledata)
colSums(is.na(teledata))
summary(teledata)
str(teledata)

# Cluster analysis
# Select and standardize only numeric variables
teledata_numeric <- teledata |>  
  dplyr::select(where(is.numeric)) |> 
  scale()

# Set seed for reproducibility
set.seed(123)

# Perform k-means clustering
km_out <- kmeans(teledata_numeric, centers = 3, nstart = 25)

# View clustering results
print(km_out$centers) # Centroids for each cluster
print(table(km_out$cluster)) # Size of each cluster

# Assign cluster to the data
teledata$cluster <- km_out$cluster

# Create boxplot for variable distribution over clusters
teledata |> 
  gather(key = "variable", value = "value", -cluster) |> 
  ggplot(aes(x = factor(cluster), y = value, fill = factor(cluster))) + 
  geom_boxplot() + 
  facet_wrap(~variable, scales = "free", ncol = 3) + 
  theme_minimal() +
  labs(title = "Distribution of Variables Over Clusters", x = "Cluster", y = "Value") +
  theme(legend.position = "none")

# Scatter plot for cluster analysis based on tenure and income
ggplot(teledata, aes(x = tenure, y = income, color = factor(cluster))) + 
  geom_point(alpha = 0.6) +
  theme_minimal() +
  labs(title = "Cluster Analysis Based on Tenure and Income", x = "Tenure", y = "Income", color = "Cluster")

# Load additional datasets for churn analysis
teledata1 <- read_csv("./data/churn-bigml-80.csv")
teledata2 <- read_csv("./data/churn-bigml-20.csv")

# Combine datasets using SQL
comb_teledata <- sqldf("SELECT * FROM teledata1
                      UNION ALL
                      SELECT * FROM teledata2")

# View combined data structure
View(comb_teledata)
colSums(is.na(comb_teledata))
summary(comb_teledata)
str(comb_teledata)

# Convert 'Churn' from logical to factor
comb_teledata$Churn <- factor(ifelse(comb_teledata$Churn, "Yes", "No"))

# Convert character columns to factors
comb_teledata <- comb_teledata |>  
  mutate_if(is.character, as.factor)

# Churn analysis
# Partition data into training and test sets
set.seed(123)
training_indices <- createDataPartition(y = comb_teledata$Churn, p = 0.80, list = FALSE)
train <- comb_teledata[training_indices, ]
test <- comb_teledata[-training_indices, ]

# Fit logistic regression model
fit_logit <- glm(Churn ~ ., data = train, family = "binomial")

# Predict churn probabilities on the test set
churn_probs <- predict(fit_logit, test, type = "response")

# Define cost parameters
cost_FN <- 300 # Cost for a false negative prediction
cost_FP <- 60 # Cost for a false positive prediction

# Initialize a dataframe to store results
thresh <- seq(0.01, 1, length.out = 100)
results <- data.frame(thresh = thresh, FP = numeric(length(thresh)), FN = numeric(length(thresh)), TP = numeric(length(thresh)), TN = numeric(length(thresh)), TotalCosts = numeric(length(thresh)))

# Loop to calculate total costs
for (i in seq_along(thresh)) {
  preds <- factor(ifelse(churn_probs > thresh[i], "Yes", "No"), levels = c("No", "Yes"))
  cm <- confusionMatrix(preds, test$Churn)
  results$FP[i] <- cm$table["No", "Yes"]
  results$FN[i] <- cm$table["Yes", "No"]
  results$TP[i] <- cm$table["Yes", "Yes"]
  results$TN[i] <- cm$table["No", "No"]
  results$TotalCosts[i] <- results$FN[i] * cost_FN + results$FP[i] * cost_FP
}

# Find the optimal threshold value
optimal_index <- which.min(results$TotalCosts)
optimal_thresh <- thresh[optimal_index]

# Print optimal threshold and its total costs
cat("Optimal threshold:", optimal_thresh, "\n")
cat("Costs at optimal threshold:", results$TotalCosts[optimal_index], "\n")

# Plot changes in FP and FN over threshold values
ggplot(results, aes(x = thresh)) +
  geom_line(aes(y = FP, color = "False Positives")) +
  geom_line(aes(y = FN, color = "False Negatives")) +
  labs(title = "FP and FN Over Different Threshold Values",
       x = "Threshold Value", y = "Count",
       color = "Type") +
  theme_minimal()
