# Load necessary library
library(hdi)

X <- read.csv("data/processed/X_regression.csv")
Y <- read.csv("data/processed/Y_regression.csv")

X <- X[, -1]  # Removes the first column from X
Y <- as.numeric(Y[, -1])  # Removes the first column from Y

print(X)
print(Y)

# Assuming X is your predictor matrix and y is your response vector

# Fit the de-biased Lasso model
lasso_result <- lasso.proj(X, Y)

# View the results
print(lasso_result)

# Extract p-values
p_values <- lasso_result$pval

# Display p-values
print(p_values)
