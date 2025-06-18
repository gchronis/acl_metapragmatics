library(tidyverse)
library(lme4)
library(nnet)      # For multinomial logistic regression
library(glmnet)    # For LASSO (variable selection with collinearity)
library(caret)     # For preprocessing
library(dplyr)     # For data manipulation
library(data.table)
library(corrplot) # For easy heatmap plotting
library(lsa) # for cosine
library(pheatmap)
library(pscl)
library(progress)

##############
# load data
###############

sentences <- fread("/home/gsc685/data/coca_downsampled_tokenized_sentences.csv")

toks <- fread("/home/gsc685/acl_metapragmatics/results/coca_top_bottom_whole_sentences.csv")


# List all CSV files in the directory
all_features <-fread("data/sentence_features/coca/whole_sentence_feature_vectors_roberta_buchanan_layer7.csv")


##################
# for each feature, get the 10 sentences with the highest and lowest values
##################


feature_names <- unique(all_features$feature)

# Create a list to store results
top_bottom_sentences <- list()

pb <- progress_bar$new(
  format = "  processing [:bar] :percent eta: :eta",
  total = length(feature_names), clear = FALSE, width = 60
)


# Loop over each feature
for (feat in feature_names) {
  pb$tick()  # Update progress bar

  # This filters rows where the 'feature' column == current feature
  dt  <- all_features[feature == feat]
  # Sort by predicted_value
  sorted <- dt[order(-predicted_value)]  # For descending order (highest first)

  # Take top 10 highest and lowest
  top10 <- sorted[1:10, ]
  bottom10 <- sorted[(nrow(sorted)-9):nrow(sorted), ]

  # Top 10
  top10 <- merge(top10, sentences, by.x = "sentence_id", by.y = "token_id", all.x = TRUE)
  bottom10 <- merge(bottom10, sentences, by.x = "sentence_id", by.y = "token_id", all.x = TRUE)
  # Add a column to indicate top or bottom
  top10$top_bottom <- "top"
  bottom10$top_bottom <- "bottom"
# Combine
  top_bottom_sentences[[feat]] <- rbind(top10, bottom10)
}

# Combine into one data.table and add a feature column
combined_dt <- rbindlist(top_bottom_sentences, idcol = "feature")
# Save to CSV
fwrite(combined_dt, "/home/gsc685/acl_metapragmatics/results/top_bottom_features_full_sentences.csv")


# break it down by POS




# now go the other direction---for each pos, get the features with highest and lowest values on average
# are there features that are more common in one POS than another?


# are there features that are specific to different dependency labels?

