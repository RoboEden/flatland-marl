library(tidyverse)
library(ggplot)
install.packages("prcomp")
library(stats)

original_rollout_data <- read.csv("./rollout_history/combined_rollout_tree_embedding.csv")

summary(original_rollout_data)

original_rollout_data <- original_rollout_data %>%
  mutate(
    X0.1 = as.logical(X0.1),
    X1.1 = as.logical(X1.1),
    X2.1 = as.logical(X2.1),
    X3.1 = as.logical(X3.1),
    X4.1 = as.logical(X4.1),
    left_switch = X1.1 & X2.1,
    right_switch = X3.1 & X2.1,
    decision_possible = case_when(
      left_switch ~ "left_switch",
      right_switch ~ "right_switch",
      TRUE ~ "no_switch"
    )
  )

summary(original_rollout_data)


ggplot(original_rollout_data %>% select(X, X2, left_switch), mapping = aes(
  x = X1,
  y = X2,
  color = left_switch)) + 
  geom_point()


# box plots of distribution for each variable

rollout_data_long <- pivot_longer(original_rollout_data %>% select(-c(X)), X0:X127)

summary(rollout_data_long)
head(rollout_data_long)

ggplot(rollout_data_long, mapping = aes(x = name, y = value, color = left_switch)) + 
         geom_boxplot()

rollout_data_long_summary <- rollout_data_long %>% group_by(name, decision_possible) %>% 
  summarize(mean_value = mean(value)) %>% 
  group_by(name) %>%
  mutate(max_value = max(mean_value), 
         min_value = min(mean_value), 
         delta = abs(max_value - min_value)
         )

ggplot(rollout_data_long_summary %>% filter(delta >0.3), mapping = aes(x = decision_possible, y = mean_value, group = name, color = name)) + 
  geom_line()



pca <- prcomp(original_rollout_data %>% select(X0:X127))

plot(pca)
pcacharts(pca)
pca$x

summary(pca)

pca_df <- cbind(data.frame(pca$x), original_rollout_data$decision_possible) %>% 
  rename("decision_possible" = "original_rollout_data$decision_possible")

colnames(pca_df)

pca_df %>% group_by(decision_possible) %>% summarize(count = n())

ggplot(pca_df %>% select(PC1, PC2, decision_possible), mapping = aes(x = PC1, y = PC2, color = decision_possible)) + 
  geom_point(position = position_jitter(width = 0.05, height = 0.05))

pca_df %>% select(PC1, PC2, decision_possible)

print(pca$sdev^2)
biplot(pca)
