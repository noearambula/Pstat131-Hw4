library(tidymodels)
library(ISLR)
library(ISLR2)
library(tidyverse)
tidymodels_prefer()
titanic <- read_csv("titanic.csv")

set.seed(777)

titanic$survived =  factor(titanic$survived, levels = c("Yes", "No")) 
# Note can use parse_factor() in order to give a warning when there is a value not in the set

titanic$pclass =  factor(titanic$pclass)

class(titanic$survived)
class(titanic$pclass)

titanic

# Q1 SPLIT
titanic_split <- initial_split(titanic, strata = survived, prop = 0.8)

titanic_train <- training(titanic_split)
titanic_test <- testing(titanic_split)


titanic_recipe <- recipe(survived ~ pclass + sex + age + sib_sp + parch + fare, data = titanic_train) %>% 
  step_impute_linear(age, impute_with = imp_vars(sib_sp)) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_interact(~ starts_with("sex"):age + age:fare)

# Q2 
# Creating tuned recipe for use later in workflows

titanic_tuned_rec <- titanic_recipe %>%
  step_poly(pclass, sex,age , sib_sp, parch, fare, degree = tune())  # polynomial regression

# rsample object of the cross-validation resamples

titanic_folds <- vfold_cv(titanic_train, v = 10) # Here ISLR uses v instead of k but they are interchangeable, common values include 5 or 10
titanic_folds # creates the k-Fold data set

# tibble with hyperparameter values we are exploring
degree_grid <- grid_regular(degree(range = c(1, 10)), levels = 10)
degree_grid 

# Q4

  # Logistic regression
  # We will use the recipe created in Question 2 to create workflows

# Logistic regression
log_reg <- logistic_reg() %>% 
  set_engine("glm") %>% 
  set_mode("classification")

titanic_tuned_log_wf <- workflow() %>%  # log workflow
  add_recipe(titanic_tuned_rec) %>%
  add_model(log_reg)

   # LDA
# Linear discriminant analysis
lda_mod <- discrim_linear() %>% 
  set_mode("classification") %>% 
  set_engine("MASS")

titanic_tuned_lda_wf <- workflow() %>%  # lda workflow
  add_recipe(titanic_tuned_rec) %>%
  add_model(lda_mod)

   # QDA
# Quadratic discriminant analysis
qda_mod <- discrim_quad() %>% 
  set_mode("classification") %>% 
  set_engine("MASS")

titanic_tuned_qda_wf <- workflow() %>%  # qda workflow
  add_model(qda_mod) %>% 
  add_recipe(titanic_tuned_rec)

# Q5 FIT MODELS

   # LOG REG
tune_log <- tune_grid(  
  object = titanic_tuned_log_wf, 
  resamples = titanic_folds, 
  grid = degree_grid,
  control = control_grid(verbose = TRUE))

# tune_grid fits model within each fold for each value in degree grid
# the control option prints out progress which helps with models that take a long time to fit


  # LDA
tune_lda <- tune_grid(  
  object = titanic_tuned_lda_wf, 
  resamples = titanic_folds, 
  grid = degree_grid,
  control = control_grid(verbose = TRUE))

# tune_grid fits model within each fold for each value in degree grid
# the control option prints out progress which helps with models that take a long time to fit


  # QDA

tune_qda <- tune_grid(  
  object = titanic_tuned_qda_wf, 
  resamples = titanic_folds, 
  grid = degree_grid,
  control = control_grid(verbose = TRUE))

# tune_grid fits model within each fold for each value in degree grid
# the control option prints out progress which helps with models that take a long time to fit

# SAVING ALL THE FITS
save(tune_log, tune_lda, tune_qda,file = "fittedmodels.rda")
load(file = "fittedmodels.rda")

.notes
