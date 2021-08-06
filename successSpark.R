##################################################
##Config
##################################################
system("java -version") #"1.8.0_45"
install.packages("sparklyr")
packageVersion("sparklyr") #‘1.7.1’

library(sparklyr)
spark_install(version = "2.3")
#spark_available_versions()

#defines the local machine (or cluster elsewhere) with the right version of Spark initially loaded
sc <- spark_connect(master = "local", version = "2.3")

#######################################################
##Data
## here loads directly, column type useless with the txt format
sc_process<-spark_read_csv(sc, name="process_claims", path="file:///H://Process_families_ML_simplified.txt",  memory=TRUE,
                           quote = "\"", delimiter="\t", header=TRUE, dec=",", overwrite=TRUE)



#spec_compatible <- c(process_first= "integer", 
#        parent="integer", 
#         family_size="integer",
#        age_pat_years="numeric", 
#        age_pat_months="numeric", 
#        originality="numeric", 
#        generality="numeric", 
#        orig_count_cited="integer",
#        orig_count_classes="integer",
#        gen_count_citing="integer", 
#       gen_count_classes="integer", 
#       techdistance="numeric", 
#       university="integer", 
#       coassignees="integer", 
#       coassignees_bin="integer",
#       p_patentstock="integer", 
#       p_processstock_first="integer", 
#       p_processstock_share="numeric", 
#       p_d_patentstock="numeric", 
#       g_d_processstock_share="numeric", 
#       p_d_processstock_first="numeric", 
#       p_d_processstock_share="numeric",
#       g_patentstock="numeric", 
#       g_processstock_first="numeric", 
#       g_processstock_share="numeric", 
#       g_d_patentstock="numeric", 
#       g_d_processstock_first="numeric",
#       firm="integer", 
#       individual="integer",
#       gov="integer", 
#       us="integer", 
#       review="integer", 
#       priority90s="integer", 
#       priority20s="integer", 
#       prioritylate20s="integer", 
#       wipo_sector1="character",
#       wipo_sector1_EE="integer", 
#       wipo_sector1_Instruments="integer", 
#       wipo_sector1_ME="integer", 
#       wipo_sector1_Other="integer", 
#       cpc_section1_B="integer", 
#       cpc_section1_C="integer", 
#       cpc_section1_D="integer", 
#       cpc_section1_E="integer", 
#       cpc_section1_F="integer", 
#       cpc_section1_G= "integer", 
#       cpc_section1_H="integer")



dplyr::src_tbls(sc) #appears within the connection with the name defined above



#summarises the full db
summarize_all(sc_process, list(mean, sd))

tblstat<-sc_process %>% summarise_all(list(min, max))

#anomalies detection
library(DBI)
anom_preview <- dbGetQuery(sc, "SELECT * FROM process_claims WHERE (techdistance <0 OR originality<0 OR generality <0)") #summarises potential anomalies in the main env



#grouping per sectors
library(dplyr)
sc_process %>% group_by(wipo_sector1) %>% summarize_all(mean)

#launches correlation plot with numeric variables only
library(corrr)
cached_cars <- sc_process %>% dplyr::select(where(is.numeric)) %>% compute("cached_cars")
correlate(cached_cars, use = "pairwise.complete.obs", method = "pearson") %>%
  shave() %>%
  rplot() 

###defines the distribution of the key variable -- considering weights because overrepresentation of product claims
sc_process %>% group_by(process_first) %>% summarize(count = n())


#quick plot of the process mean across sectors -- simplified to the avg number of first process claims
library(ggplot2)
car_group <- sc_process %>%
  group_by(wipo_sector1) %>%
  summarise(mproc = mean(process_first, na.rm = TRUE)) %>%
  collect() %>%
  print()
#plots the trend across sector
ggplot(car_group, aes(wipo_sector1, mproc))+geom_col()

#overall distribution -- dynamic over decades
#library(dbplot)
#1990s
sc_process %>% group_by(priority90s) %>%
  summarise(mproc = mean(process_first, na.rm = TRUE))
#2000s
sc_process %>% group_by(priority20s) %>%
  summarise(mproc = mean(process_first, na.rm = TRUE))
#latest period
sc_process %>% group_by(prioritylate20s) %>%
  summarise(mproc = mean(process_first, na.rm = TRUE))


###########################################################
##Modelling
############################################################
##partitions
data_splits <- sdf_random_split(sc_process, training = 0.6, testing = 0.4, seed = 42)
proc_train <- data_splits$training
proc_test <- data_splits$testing

#alternative with sdf_partition
# partitions <- process_claims  %>%
# sdf_partition(training = 0.5, test = 0.5, seed = 1099)

proc_train$process_first<-as.double(proc_train$process_first)
proc_test$process_first<-as.double(proc_test$process_first)

################################################################
####Trains
################################################################
###model with all variables except the sector control from WIPO
ml_formula <- formula(process_first~ . - wipo_sector1)


###defines different models
################################################################
# Logit
ml_log <- ml_logistic_regression(proc_train, ml_formula)

# Decision Tree
ml_dt <- ml_decision_tree(proc_train, ml_formula)

## Random Forest
ml_rf <- ml_random_forest(proc_train, ml_formula)

## Gradient Boosted Tree
ml_gbt <- ml_gradient_boosted_trees(proc_train, ml_formula)

## Naive Bayes 
ml_nb <- ml_naive_bayes(proc_train, ml_formula)

## Neural Network
ml_nn <- ml_multilayer_perceptron_classifier(proc_train, ml_formula, layers = c(11,15,2))

###################################################################
##Validates
####################################################################
ml_models <- list(
  "Logistic" = ml_log,
  "Decision Tree" = ml_dt,
  "Random Forest" = ml_rf,
  "Gradient Boosted Trees" = ml_gbt,
  "Naive Bayes" = ml_nb,
  "Neural Net" = ml_nn
)

# Create a function for scoring for each model with its respective prediction
### here substitutes the sdf_predict from the previous spark version -- works outside the loop ...
score_test_data <- function(model, data=proc_test){
  pred <- ml_predict(model, data)
  select(pred, process_first, prediction)
}


# stores the results of the prediction performance for all models
ml_score <- lapply(ml_models, score_test_data)

########################################################################
##Lifting
########################################################################
# Lift function
calculate_lift <- function(scored_data) {
  scored_data %>%
    mutate(bin = ntile(desc(prediction), 10)) %>% 
    group_by(bin) %>% 
    summarize(count = sum(process_first)) %>% 
    mutate(prop = count / sum(count)) %>% 
    arrange(bin) %>% 
    mutate(prop = cumsum(prop)) %>% 
    select(-count) %>% 
    collect() %>% 
    as.data.frame()
}

# Initialize results: bugs but computes.
ml_gains <- data.frame(bin = 1:10, prop = seq(0, 1, len = 10), model = "Base")

# Calculate lift for each model
for(i in names(ml_score)){
  ml_gains <- ml_score[[i]] %>%
    calculate_lift %>%
    mutate(model = i) %>%
    rbind(ml_gains, .)
}


# Plot results. Trees and forest work better than other models
ggplot(ml_gains, aes(x = bin, y = prop, colour = model)) +
  geom_point() + geom_line() +
  ggtitle("Lift Chart for Predicting Process claims (first one) - Test Data Set") + 
  xlab("") + ylab("")


#########################################################################
####Accuracy evaluation
##########################################################################
# Function for calculating accuracy -- prediction stored as list
calc_accuracy <- function(data, cutpoint = 0.5){ #simplified the predicted proba in 1 vs 0
  data %>% 
    mutate(prediction = if_else(prediction > cutpoint, 1.0, 0.0)) %>%
    ml_classification_eval("prediction", "process_first", metric_name="accuracy")
} #here issue with process first which should be Double and not Integer...

claims_pipeline <- ml_pipeline(sc) %>%
  ft_dplyr_transformer(
    tbl = proc_train☺
  ) %>%
  ft_binarizer(
    input.col = "process_first",
    output.col = "process_claim_first"
  ) %>%
  ft_r_formula(process_claim_first ~ . -wipo_sector1) %>% 
  ml_logistic_regression()


# Create a pipeline and fit it
#pipeline <- ml_pipeline(sc) %>%
 # ft_binarizer("hp", "big_hp", threshold = 100) %>%
 # ft_vector_assembler(c("big_hp", "wt", "qsec"), "features") %>%
 # ml_gbt_regressor(label_col = "mpg")
#pipeline_model <- ml_fit(pipeline, mtcars_tbl)

#Once we have the pipeline model, we can export it via ml_write_bundle():
  
  # Export model
#  model_path <- file.path(tempdir(), "mtcars_model.zip")
#transformed_tbl <- ml_transform(pipeline_model, mtcars_tbl)
#ml_write_bundle(pipeline_model, transformed_tbl, model_path)
#spark_disconnect(sc)


# Calculate AUC and accuracy // issue with the variable type, process_first as Double 
#perf_metrics <- data.frame(
 # model = names(ml_score),
 # AUC = 100 * sapply(ml_score, ml_binary_classification_eval, "process_first", "prediction"),
 # Accuracy = 100 * lapply(ml_score, calc_accuracy),
 # row.names = NULL, stringsAsFactors = FALSE)


# Plot results
#library(tidyr)
#gather(perf_metrics, metric, value, Accuracy) %>%
 # ggplot(aes(reorder(model, value), value, fill = metric)) + 
 # geom_bar(stat = "identity", position = "dodge") + 
 # coord_flip() +
 # xlab("") +
 # ylab("Percent") +
 # ggtitle("Performance Metrics")

###computes by hand: issue in the lenght of mscore: AUC by default
#Model 2; random forest
rf_model <- proc_train %>% ml_random_forest(process_first~ . - wipo_sector1, type = "classification")
pred <- ml_predict(rf_model, proc_test)
ml_binary_classification_evaluator(pred) #71,05% under ROC
ml_classification_evaluator(pred , metric_name="accuracy") #71,33 correctly classified
#gathers the results of the predictions to compute the confusion matrix
library(tidyr)
rf_test_df <-pred %>% collect()
#rf_test_df <- collect(pred) too large

#confusion matrix
table(rf_test_df$process_first, rf_test_df$prediction) #limited power to explain patents with a first process claims

#model 3 "Decision Tree" =
ml_dt<- proc_train %>% ml_decision_tree(process_first~ . - wipo_sector1, type = "classification")
pred_dt <- ml_predict(ml_dt, proc_test)
ml_binary_classification_evaluator(pred_dt) #55,56% under ROC
ml_classification_eval(pred_dt, metric_name="accuracy") #72,65% correctly classified
table(pred_dt$prediction, pred_dt$process_first)

#model 4 "Gradient Boosted Trees" = best model 
  ml_gbt<- proc_train %>% ml_gradient_boosted_trees(process_first~ . - wipo_sector1, type = "classification")
  pred_gbt <- ml_predict(ml_gbt, proc_test)
  ml_binary_classification_evaluator(pred_gbt) #76,53% under ROC
  ml_classification_eval(pred_gbt, metric_name="accuracy") #74,55% correctly classifie
  rf_gbt_df <-pred_gbt %>% collect()
  #quick confusion matrix
  table(rf_gbt_df$prediction, rf_gbt_df$process_first) 

  ###full set of performance indicators with sdf_predict
  resgbt<-m_predict(ml_gbt, proc_test)
  confusionMatrix(pred_gbt) 
  
  #check with SVM on top but performances better with GBT 
  svm_model <- proc_train %>%
    ml_linear_svc(process_first~ . - wipo_sector1)
  svm_pred <- ml_predict(svm_model, proc_test) #74,55 under ROC
  ml_binary_classification_evaluator(svm_pred) #71% correctly classified
#############################################################################
###Features importance
###############################################################################
# Initialize results
feature_importance <- data.frame()

# Calculate feature importance for the trees
for(i in c("Decision Tree", "Random Forest", "Gradient Boosted Trees")){
  feature_importance <- ml_tree_feature_importance(ml_models[[i]]) %>%
    mutate(Model = i) %>%
    mutate(importance = as.numeric(importance)) %>%
    mutate(feature = as.character(feature)) %>%
    rbind(feature_importance, .)
}

# Plot results for trees which seem more relevant
feature_importance %>%
  ggplot(aes(reorder(feature, importance), importance, fill = Model)) + 
  facet_wrap(~Model) +
  geom_bar(stat = "identity") + 
  coord_flip() +
  xlab("") +
  ggtitle("Feature Importance")

###############################################################################
###check the results for GBT
# Create a pipeline
pipeline <- ml_pipeline(sc) %>%
  ft_r_formula(process_first ~ . - wipo_sector1) %>%
  ml_gbt_classifier()
pipeline

# Specify hyperparameter grid
grid <- list(
  gbt_classifier = list(
    #step_size = c(0.01, 0.1),
    max_depth = c(5, 10),
    #impurity = c("entropy", "gini"),
    max_iter=50
  )
)

# Create the cross validator object /// binary classification
cv <- ml_cross_validator(sc,
  estimator = pipeline,
  evaluator = ml_binary_classification_evaluator(sc),
 # estimator_param_maps = list(
  #  gbt_classifier = list(
      #step_size = c(0.01, 0.1),
   #   max_depth = c(5, 10),
      #impurity = c("entropy", "gini"),
    #  max_iter=50
    #)),
  num_folds = 3,
  parallelism = 4)

# Train the models
cv_model <- ml_fit(cv, proc_train)

# Print the metrics
ml_validation_metrics(cv_model)



spark_disconnect(sc)
################################################################

