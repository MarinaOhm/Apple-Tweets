# Apple Tweets Sentiment Analysis: Avg classification
bdanalytics  

**  **    
**Date: (Mon) Apr 27, 2015**    

# Introduction:  

Data: 
Source: 
    Training:   https://courses.edx.org/c4x/MITx/15.071x_2/asset/tweets.csv  
    New:        <newdt_url>  
Time period: 



# Synopsis:

Based on analysis utilizing <> techniques, <conclusion heading>:  

### ![](<filename>.png)

## Potential next steps include:
- Organization:
    - Categorize by chunk
    - Priority criteria:
        0. Ease of change
        1. Impacts report
        2. Cleans innards
        3. Bug report
        
- manage.missing.data chunk:
    - cleaner way to manage re-splitting of training vs. new entity

- fit.models chunk:
    - Prediction accuracy scatter graph:
    -   Add tiles (raw vs. PCA)
    -   Use shiny for drop-down of "important" features
    -   Use plot.ly for interactive plots ?
    
    - Change .fit suffix of model metrics to .mdl if it's data independent (e.g. AIC, Adj.R.Squared - is it truly data independent ?, etc.)
    - move model_type parameter to myfit_mdl before indep_vars_vctr (keep all model_* together)
    - create a custom model for rpart that has minbucket as a tuning parameter
    - varImp for randomForest crashes in caret version:6.0.41 -> submit bug report

- Probability handling for multinomials vs. desired binomial outcome
-   ROCR currently supports only evaluation of binary classification tasks (version 1.0.7)
-   extensions toward multiclass classification are scheduled for the next release

- Skip trControl.method="cv" for dummy classifier ?
- Add custom model to caret for a dummy (baseline) classifier (binomial & multinomial) that generates proba/outcomes which mimics the freq distribution of glb_rsp_var values; Right now glb_dmy_glm_mdl always generates most frequent outcome in training data
- glm_dmy_mdl should use the same method as glm_sel_mdl until custom dummy classifer is implemented

- Compare glb_sel_mdl vs. glb_fin_mdl:
    - varImp
    - Prediction differences (shd be minimal ?)

- Move glb_analytics_diag_plots to mydsutils.R: (+) Easier to debug (-) Too many glb vars used
- Add print(ggplot.petrinet(glb_analytics_pn) + coord_flip()) at the end of every major chunk
- Parameterize glb_analytics_pn
- Move glb_impute_missing_data to mydsutils.R: (-) Too many glb vars used; glb_<>_df reassigned
- Replicate myfit_mdl_classification features in myfit_mdl_regression
- Do non-glm methods handle interaction terms ?
- f-score computation for classifiers should be summation across outcomes (not just the desired one ?)
- Add accuracy computation to glb_dmy_mdl in predict.data.new chunk
- Why does splitting fit.data.training.all chunk into separate chunks add an overhead of ~30 secs ? It's not rbind b/c other chunks have lower elapsed time. Is it the number of plots ?
- Incorporate code chunks in print_sessionInfo
- Test against 
    - projects in github.com/bdanalytics
    - lectures in jhu-datascience track

# Analysis: 

```r
rm(list=ls())
set.seed(12345)
options(stringsAsFactors=FALSE)
source("~/Dropbox/datascience/R/mydsutils.R")
source("~/Dropbox/datascience/R/myplot.R")
source("~/Dropbox/datascience/R/mypetrinet.R")
# Gather all package requirements here
#suppressPackageStartupMessages(require())
#packageVersion("snow")

#require(sos); findFn("pinv", maxPages=2, sortby="MaxScore")

# Analysis control global variables
glb_trnng_url <- "https://courses.edx.org/c4x/MITx/15.071x_2/asset/tweets.csv"
glb_newdt_url <- "<newdt_url>"
glb_is_separate_newent_dataset <- FALSE    # or TRUE
glb_split_entity_newent_datasets <- TRUE   # or FALSE
glb_split_newdata_method <- "sample"          # "condition" or "sample" or "copy"
glb_split_newdata_condition <- "<col_name> <condition_operator> <value>"    # or NULL
glb_split_newdata_size_ratio <- 1 - 0.7               # > 0 & < 1
glb_split_sample.seed <- 123               # or any integer
glb_max_obs <- NULL # or any integer

glb_is_regression <- FALSE; glb_is_classification <- TRUE; glb_is_binomial <- TRUE

glb_rsp_var_raw <- "Avg"

# for classification, the response variable has to be a factor
glb_rsp_var <- "Negative.fctr"

# if the response factor is based on numbers e.g (0/1 vs. "A"/"B"), 
#   caret predict(..., type="prob") crashes
glb_map_rsp_raw_to_var <- function(raw) {
    relevel(factor(ifelse(raw <= -1, "Y", "N")), as.factor(c("Y", "N")), ref="N")
    #as.factor(paste0("B", raw))
    #as.factor(raw)    
}
glb_map_rsp_raw_to_var(c(-2, -1, 0, 1, 2))
```

```
## [1] Y Y N N N
## Levels: N Y
```

```r
glb_map_rsp_var_to_raw <- function(var) {
    #as.numeric(var) - 1
    #as.numeric(var)
    #levels(var)[as.numeric(var)]
    c("N", "Y")[as.numeric(var)]
}
glb_map_rsp_var_to_raw(glb_map_rsp_raw_to_var(c(-2, -1, 0, 1, 2)))
```

```
## [1] "Y" "Y" "N" "N" "N"
```

```r
if ((glb_rsp_var != glb_rsp_var_raw) & is.null(glb_map_rsp_raw_to_var))
    stop("glb_map_rsp_raw_to_var function expected")

glb_rsp_var_out <- paste0(glb_rsp_var, ".predict.") # model_id is appended later
glb_id_vars <- NULL # or c("<id_var>")

# List transformed vars  
glb_exclude_vars_as_features <- c("Tweet.fctr")    
# List feats that shd be excluded due to known causation by prediction variable
if (glb_rsp_var_raw != glb_rsp_var)
    glb_exclude_vars_as_features <- union(glb_exclude_vars_as_features, 
                                            glb_rsp_var_raw)
glb_exclude_vars_as_features <- union(glb_exclude_vars_as_features, 
                                      c(NULL)) # or c("<col_name>")
# List output vars (useful during testing in console)          
# glb_exclude_vars_as_features <- union(glb_exclude_vars_as_features, 
#                         grep(glb_rsp_var_out, names(glb_trnent_df), value=TRUE)) 

glb_impute_na_data <- FALSE            # or TRUE
glb_mice_complete.seed <- 144               # or any integer

glb_is_textual <- TRUE # vs. glb_is_numerical ???
#Sys.setlocale("LC_ALL", "C") # For english

# rpart:  .rnorm messes with the models badly
#         caret creates dummy vars for factor feats which messes up the tuning
#             - better to feed as.numeric(<feat>.fctr) to caret 
# Regression
if (glb_is_regression)
    glb_models_method_vctr <- c("lm", "glm", "rpart", "rf") else
# Classification
    if (glb_is_binomial)
        glb_models_method_vctr <- c("glm", "rpart", "rf") else  
        glb_models_method_vctr <- c("rpart", "rf")

glb_models_lst <- list(); glb_models_df <- data.frame()
# Baseline prediction model feature(s)
glb_Baseline_mdl_var <- NULL # or c("<col_name>")

glb_model_metric_terms <- NULL # or matrix(c(
#                               0,1,2,3,4,
#                               2,0,1,2,3,
#                               4,2,0,1,2,
#                               6,4,2,0,1,
#                               8,6,4,2,0
#                           ), byrow=TRUE, nrow=5)
glb_model_metric <- NULL # or "<metric_name>"
glb_model_metric_maximize <- NULL # or FALSE (TRUE is not the default for both classification & regression) 
glb_model_metric_smmry <- NULL # or function(data, lev=NULL, model=NULL) {
#     confusion_mtrx <- t(as.matrix(confusionMatrix(data$pred, data$obs)))
#     #print(confusion_mtrx)
#     #print(confusion_mtrx * glb_model_metric_terms)
#     metric <- sum(confusion_mtrx * glb_model_metric_terms) / nrow(data)
#     names(metric) <- glb_model_metric
#     return(metric)
# }

glb_tune_models_df <- 
   rbind(
    #data.frame(parameter="cp", min=0.00005, max=0.00005, by=0.000005),
                            #seq(from=0.01,  to=0.01, by=0.01)
    #data.frame(parameter="mtry", min=2, max=4, by=1),
    data.frame(parameter="dummy", min=2, max=4, by=1)
        ) 
# or NULL
glb_n_cv_folds <- 3 # or NULL

glb_clf_proba_threshold <- NULL # 0.5

# Model selection criteria
if (glb_is_regression)
    glb_model_evl_criteria <- c("min.RMSE.OOB", "max.R.sq.OOB", "max.Adj.R.sq.fit")
if (glb_is_classification) {
    if (glb_is_binomial)
        glb_model_evl_criteria <- c("max.Accuracy.OOB", "max.Kappa.OOB", "min.aic.fit") else
        glb_model_evl_criteria <- c("max.Accuracy.OOB", "max.Kappa.OOB")
}

glb_sel_mdl_id <- NULL # or "<model_id_prefix>.<model_method>"
glb_fin_mdl_id <- glb_sel_mdl_id # or "Final"

glb_out_pfx <- "Apple_Tweets_"

# Depict process
glb_analytics_pn <- petrinet(name="glb_analytics_pn",
                        trans_df=data.frame(id=1:6,
    name=c("data.training.all","data.new",
           "model.selected","model.final",
           "data.training.all.prediction","data.new.prediction"),
    x=c(   -5,-5,-15,-25,-25,-35),
    y=c(   -5, 5,  0,  0, -5,  5)
                        ),
                        places_df=data.frame(id=1:4,
    name=c("bgn","fit.data.training.all","predict.data.new","end"),
    x=c(   -0,   -20,                    -30,               -40),
    y=c(    0,     0,                      0,                 0),
    M0=c(   3,     0,                      0,                 0)
                        ),
                        arcs_df=data.frame(
    begin=c("bgn","bgn","bgn",        
            "data.training.all","model.selected","fit.data.training.all",
            "fit.data.training.all","model.final",    
            "data.new","predict.data.new",
            "data.training.all.prediction","data.new.prediction"),
    end  =c("data.training.all","data.new","model.selected",
            "fit.data.training.all","fit.data.training.all","model.final",
            "data.training.all.prediction","predict.data.new",
            "predict.data.new","data.new.prediction",
            "end","end")
                        ))
#print(ggplot.petrinet(glb_analytics_pn))
print(ggplot.petrinet(glb_analytics_pn) + coord_flip())
```

```
## Loading required package: grid
```

![](Apple_Tweets_files/figure-html/set_global_options-1.png) 

```r
glb_analytics_avl_objs <- NULL

glb_script_tm <- proc.time()
glb_script_df <- data.frame(chunk_label="import_data", 
                            chunk_step_major=1, chunk_step_minor=0,
                            elapsed=(proc.time() - glb_script_tm)["elapsed"])
print(tail(glb_script_df, 2))
```

```
##         chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed import_data                1                0   0.002
```

## Step `1`: import data

```r
glb_entity_df <- myimport_data(url=glb_trnng_url, 
    comment=ifelse(!glb_is_separate_newent_dataset, "glb_entity_df", "glb_trnent_df"), 
                                force_header=TRUE)
```

```
## [1] "Reading file ./data/tweets.csv..."
## [1] "dimensions of data in ./data/tweets.csv: 1,181 rows x 2 cols"
##                                                                                                   Tweet
## 1 I have to say, Apple has by far the best customer care service I have ever received! @Apple @AppStore
## 2                                          iOS 7 is so fricking smooth & beautiful!! #ThanxApple @Apple
## 3                                                                                         LOVE U @APPLE
## 4           Thank you @apple, loving my new iPhone 5S!!!!!  #apple #iphone5S pic.twitter.com/XmHJCU4pcb
## 5                    .@apple has the best customer service. In and out with a new phone in under 10min!
## 6                         @apple ear pods are AMAZING! Best sound from in-ear headphones I've ever had!
##   Avg
## 1 2.0
## 2 2.0
## 3 1.8
## 4 1.8
## 5 1.8
## 6 1.8
##                                                                                                              Tweet
## 41   I love how @apple makes it easy to transfer between Macs while I continue to work! pic.twitter.com/lga2KaXa4b
## 180                           Pretty sure if I could have any voice in the world it would be @apple's own #JonyIve
## 384                            @llombardo007 @Apple Will not realese 0.7.5 until after ios7 comes out(on the 18th)
## 601              #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO @apple @itune #IPOD GENERATION
## 858                   iPhone 5C = 5 cheap...  Plastic = not worth the money  Metal forever @Apple... Get it right!
## 1166                                                                                              freak YOU @APPLE
##       Avg
## 41    1.0
## 180   0.4
## 384   0.0
## 601   0.0
## 858  -0.6
## 1166 -2.0
##                                                                               Tweet
## 1176                                                                 freak u @apple
## 1177                                                                   freak @apple
## 1178 WHY CANT I freakING SEE PICTURES ON MY TL IM ANNOYED freak YOU @TWITTER @APPLE
## 1179                                             @APPLE YOU freakING COWS freak YOU
## 1180           @apple I hate you why is my phone not working I'm going to freak out
## 1181                              @aGounalakis that's nasty! @apple is a nasty brat
##      Avg
## 1176  -2
## 1177  -2
## 1178  -2
## 1179  -2
## 1180  -2
## 1181  -2
## 'data.frame':	1181 obs. of  2 variables:
##  $ Tweet: chr  "I have to say, Apple has by far the best customer care service I have ever received! @Apple @AppStore" "iOS 7 is so fricking smooth & beautiful!! #ThanxApple @Apple" "LOVE U @APPLE" "Thank you @apple, loving my new iPhone 5S!!!!!  #apple #iphone5S pic.twitter.com/XmHJCU4pcb" ...
##  $ Avg  : num  2 2 1.8 1.8 1.8 1.8 1.8 1.6 1.6 1.6 ...
##  - attr(*, "comment")= chr "glb_entity_df"
## NULL
```

```r
if (!glb_is_separate_newent_dataset) {
    glb_trnent_df <- glb_entity_df; comment(glb_trnent_df) <- "glb_trnent_df"
} # else glb_entity_df is maintained as is for chunk:inspectORexplore.data
    
if (glb_is_separate_newent_dataset) {
    glb_newent_df <- myimport_data(
        url=glb_newdt_url, 
        comment="glb_newent_df", force_header=TRUE)
    
    # To make plots / stats / checks easier in chunk:inspectORexplore.data
    glb_entity_df <- rbind(glb_trnent_df, glb_newent_df); comment(glb_entity_df) <- "glb_entity_df"
} else {
    if (!glb_split_entity_newent_datasets) {
        stop("Not implemented yet") 
        glb_newent_df <- glb_trnent_df[sample(1:nrow(glb_trnent_df),
                                          max(2, nrow(glb_trnent_df) / 1000)),]                    
    } else      if (glb_split_newdata_method == "condition") {
            glb_newent_df <- do.call("subset", 
                list(glb_trnent_df, parse(text=glb_split_newdata_condition)))
            glb_trnent_df <- do.call("subset", 
                list(glb_trnent_df, parse(text=paste0("!(", 
                                                      glb_split_newdata_condition,
                                                      ")"))))
        } else if (glb_split_newdata_method == "sample") {
                require(caTools)
                
                set.seed(glb_split_sample.seed)
                split <- sample.split(glb_trnent_df[, glb_rsp_var_raw], 
                                      SplitRatio=(1-glb_split_newdata_size_ratio))
                glb_newent_df <- glb_trnent_df[!split, ] 
                glb_trnent_df <- glb_trnent_df[split ,]
        } else if (glb_split_newdata_method == "copy") {  
            glb_trnent_df <- glb_entity_df
            comment(glb_trnent_df) <- "glb_trnent_df"
            glb_newent_df <- glb_entity_df
            comment(glb_newent_df) <- "glb_newent_df"
        } else stop("glb_split_newdata_method should be %in% c('condition', 'sample', 'copy')")   

    comment(glb_newent_df) <- "glb_newent_df"
    myprint_df(glb_newent_df)
    str(glb_newent_df)

    if (glb_split_entity_newent_datasets) {
        myprint_df(glb_trnent_df)
        str(glb_trnent_df)        
    }
}         
```

```
## Loading required package: caTools
```

```
##                                                                                                                                          Tweet
## 2                                                                                 iOS 7 is so fricking smooth & beautiful!! #ThanxApple @Apple
## 5                                                           .@apple has the best customer service. In and out with a new phone in under 10min!
## 8                                                                                                      the iPhone 5c is so beautiful <3 @Apple
## 11                                                                                  I love the new iOS so much!!!!! Thnx @apple @phillydvibing
## 16 Just watched the keynote of @apple latest iPhones. I just love the #iPhone5S and #iphone5c ??? I guess, i have a christmas gift already????
## 20                              Can't wait for my #orange phone upgrade in November :-) #apple iPhone 5s here I come ;-) @OrangeHelpers @apple
##    Avg
## 2  2.0
## 5  1.8
## 8  1.6
## 11 1.6
## 16 1.6
## 20 1.4
##                                                                                                                                        Tweet
## 171                                                                                                                        FOLLOW @APPLE NOW
## 256                                                       A #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO @apple @itune
## 474    {#IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO @apple @itune}
## 809  Waay too many people waiting for #AppleCare, in my @Apple era this would not have been acceptable #Commodity pic.twitter.com/hgzYTEx3ya
## 933                                              @apple #failed big time with their new iPhones. What happened to it being a "budget" phone?
## 1022                                                                       I don't like #iOS7 they're changing the look of everything @apple
##       Avg
## 171   0.4
## 256   0.2
## 474   0.0
## 809  -0.4
## 933  -0.8
## 1022 -1.0
##                                                                                                                                         Tweet
## 1158                                                                                                               I freaking hate you @apple
## 1167                                                                                                                         freak you @apple
## 1170                                                                                                      @apple worst customer service ever.
## 1173 We should boycott @Apple or freakin flame them or something like how they have all this money but can't make stable chargers #freakApple
## 1174                wtf @telstra @apple why would you have pre-order for the 5c the crap phone no one wants and not the 5s i hate you plz die
## 1177                                                                                                                             freak @apple
##       Avg
## 1158 -1.8
## 1167 -2.0
## 1170 -2.0
## 1173 -2.0
## 1174 -2.0
## 1177 -2.0
## 'data.frame':	356 obs. of  2 variables:
##  $ Tweet: chr  "iOS 7 is so fricking smooth & beautiful!! #ThanxApple @Apple" ".@apple has the best customer service. In and out with a new phone in under 10min!" "the iPhone 5c is so beautiful <3 @Apple" "I love the new iOS so much!!!!! Thnx @apple @phillydvibing" ...
##  $ Avg  : num  2 1.8 1.6 1.6 1.6 1.4 1.4 1.4 1.2 1.2 ...
##  - attr(*, "comment")= chr "glb_newent_df"
##                                                                                                                                        Tweet
## 1                                      I have to say, Apple has by far the best customer care service I have ever received! @Apple @AppStore
## 3                                                                                                                              LOVE U @APPLE
## 4                                                Thank you @apple, loving my new iPhone 5S!!!!!  #apple #iphone5S pic.twitter.com/XmHJCU4pcb
## 6                                                              @apple ear pods are AMAZING! Best sound from in-ear headphones I've ever had!
## 7 Omg the iPhone 5S is so cool it can read your finger print to unlock your iPhone 5S and to make purchases without a passcode #Apple @Apple
## 9                       #AttributeOwnership is exactly why @apple will always be #one! #apple #marketing #marketer #business #innovation #fb
##   Avg
## 1 2.0
## 3 1.8
## 4 1.8
## 6 1.8
## 7 1.8
## 9 1.6
##                                                                                                                                     Tweet
## 144                                                        How @Apple Is Improving Mobile App Security http://fw.to/Qo0DqXT  #Mobile #App
## 514 {#IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO @apple @itune}
## 653            Dear @Apple    im sure im not the only one that types "nerd" so lets stop putting a red line under it.    Thanks,  Diego??
## 700                                                                                               OS X 10.8.5 indirilmeye sunulmus @apple
## 816                             .@Apple discontinues Cards app and printing service, recommends using #iPhoto instead. http://ht.ly/oPzms
## 934                                                        I'm PISSED that Pages is now free @apple where is my refund for buying it! LOL
##      Avg
## 144  0.4
## 514  0.0
## 653 -0.2
## 700 -0.2
## 816 -0.4
## 934 -0.8
##                                                                               Tweet
## 1175                                                                Hate you @apple
## 1176                                                                 freak u @apple
## 1178 WHY CANT I freakING SEE PICTURES ON MY TL IM ANNOYED freak YOU @TWITTER @APPLE
## 1179                                             @APPLE YOU freakING COWS freak YOU
## 1180           @apple I hate you why is my phone not working I'm going to freak out
## 1181                              @aGounalakis that's nasty! @apple is a nasty brat
##      Avg
## 1175  -2
## 1176  -2
## 1178  -2
## 1179  -2
## 1180  -2
## 1181  -2
## 'data.frame':	825 obs. of  2 variables:
##  $ Tweet: chr  "I have to say, Apple has by far the best customer care service I have ever received! @Apple @AppStore" "LOVE U @APPLE" "Thank you @apple, loving my new iPhone 5S!!!!!  #apple #iphone5S pic.twitter.com/XmHJCU4pcb" "@apple ear pods are AMAZING! Best sound from in-ear headphones I've ever had!" ...
##  $ Avg  : num  2 1.8 1.8 1.8 1.8 1.6 1.6 1.6 1.6 1.6 ...
##  - attr(*, "comment")= chr "glb_trnent_df"
```

```r
if (!is.null(glb_max_obs)) {
    if (nrow(glb_trnent_df) > glb_max_obs) {
        warning("glb_trnent_df restricted to glb_max_obs: ", format(glb_max_obs, big.mark=","))
        org_entity_df <- glb_trnent_df
        glb_trnent_df <- org_entity_df[split <- 
            sample.split(org_entity_df[, glb_rsp_var_raw], SplitRatio=glb_max_obs), ]
        org_entity_df <- NULL
    }
#     if (nrow(glb_newent_df) > glb_max_obs) {
#         warning("glb_newent_df restricted to glb_max_obs: ", format(glb_max_obs, big.mark=","))        
#         org_newent_df <- glb_newent_df
#         glb_newent_df <- org_newent_df[split <- 
#             sample.split(org_newent_df[, glb_rsp_var_raw], SplitRatio=glb_max_obs), ]
#         org_newent_df <- NULL
#     }    
}

if (nrow(glb_trnent_df) == nrow(glb_entity_df))
    warning("glb_trnent_df same as glb_entity_df")
if (nrow(glb_newent_df) == nrow(glb_entity_df))
    warning("glb_newent_df same as glb_entity_df")

glb_script_df <- rbind(glb_script_df,
                   data.frame(chunk_label="cleanse_data", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##           chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed   import_data                1                0   0.002
## elapsed1 cleanse_data                2                0   0.503
```

## Step `2`: cleanse data

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="inspectORexplore.data", 
                              chunk_step_major=max(glb_script_df$chunk_step_major), 
                              chunk_step_minor=1,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                    chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed1          cleanse_data                2                0   0.503
## elapsed2 inspectORexplore.data                2                1   0.538
```

### Step `2`.`1`: inspect/explore data

```r
#print(str(glb_trnent_df))
#View(glb_trnent_df)

# List info gathered for various columns
# <col_name>:   <description>; <notes>

# Create new features that help diagnostics
#   Create factors of string variables
str_vars <- sapply(1:length(names(glb_trnent_df)), 
    function(col) ifelse(class(glb_trnent_df[, names(glb_trnent_df)[col]]) == "character",
                         names(glb_trnent_df)[col], ""))
if (length(str_vars <- setdiff(str_vars[str_vars != ""], 
                               glb_exclude_vars_as_features)) > 0) {
    warning("Creating factors of string variables:", paste0(str_vars, collapse=", "))
    glb_exclude_vars_as_features <- union(glb_exclude_vars_as_features, str_vars)
    for (var in str_vars) {
        glb_entity_df[, paste0(var, ".fctr")] <- factor(glb_entity_df[, var], 
                        as.factor(unique(glb_entity_df[, var])))
        glb_trnent_df[, paste0(var, ".fctr")] <- factor(glb_trnent_df[, var], 
                        as.factor(unique(glb_entity_df[, var])))
        glb_newent_df[, paste0(var, ".fctr")] <- factor(glb_newent_df[, var], 
                        as.factor(unique(glb_entity_df[, var])))
    }
}
```

```
## Warning: Creating factors of string variables:Tweet
```

```r
#   Convert factors to dummy variables
#   Build splines   require(splines); bsBasis <- bs(training$age, df=3)

add_new_diag_feats <- function(obs_df, ref_df=glb_entity_df) {
    require(plyr)
    
    obs_df <- mutate(obs_df,
#         <col_name>.NA=is.na(<col_name>),

#         <col_name>.fctr=factor(<col_name>, 
#                     as.factor(union(obs_df$<col_name>, obs_twin_df$<col_name>))), 
#         <col_name>.fctr=relevel(factor(<col_name>, 
#                     as.factor(union(obs_df$<col_name>, obs_twin_df$<col_name>))),
#                                   "<ref_val>"), 
#         <col2_name>.fctr=relevel(factor(ifelse(<col1_name> == <val>, "<oth_val>", "<ref_val>")), 
#                               as.factor(c("R", "<ref_val>")),
#                               ref="<ref_val>"),

          # This doesn't work - use sapply instead
#         <col_name>.fctr_num=grep(<col_name>, levels(<col_name>.fctr)), 
#         
#         Date.my=as.Date(strptime(Date, "%m/%d/%y %H:%M")),
#         Year=year(Date.my),
#         Month=months(Date.my),
#         Weekday=weekdays(Date.my)

#         <col_name>.log=log(<col.name>),        
#         <col_name>=<table>[as.character(<col2_name>)],
#         <col_name>=as.numeric(<col2_name>),

        .rnorm=rnorm(n=nrow(obs_df))
                        )

    # If levels of a factor are different across obs_df & glb_newent_df; predict.glm fails  
    # Transformations not handled by mutate
#     obs_df$<col_name>.fctr.num <- sapply(1:nrow(obs_df), 
#         function(row_ix) grep(obs_df[row_ix, "<col_name>"],
#                               levels(obs_df[row_ix, "<col_name>.fctr"])))
    
    print(summary(obs_df))
    print(sapply(names(obs_df), function(col) sum(is.na(obs_df[, col]))))
    return(obs_df)
}

glb_entity_df <- add_new_diag_feats(glb_entity_df)
```

```
## Loading required package: plyr
```

```
##     Tweet                Avg         
##  Length:1181        Min.   :-2.0000  
##  Class :character   1st Qu.:-0.6000  
##  Mode  :character   Median : 0.0000  
##                     Mean   :-0.1931  
##                     3rd Qu.: 0.2000  
##                     Max.   : 2.0000  
##                                      
##                                                                                                                                   Tweet.fctr  
##  {#IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO  @apple @itune}:   9  
##  {#IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO @apple @itune} :   9  
##  C #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO @apple @itune                                                    :   8  
##  #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO @apple @itune #IPOD GENERATION                                     :   7  
##  FOLLOW @APPLE NOW                                                                                                                     :   5  
##  #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO @apple @itune                                                      :   5  
##  (Other)                                                                                                                               :1138  
##      .rnorm        
##  Min.   :-3.04697  
##  1st Qu.:-0.71210  
##  Median :-0.02806  
##  Mean   :-0.02602  
##  3rd Qu.: 0.65972  
##  Max.   : 3.30433  
##                    
##      Tweet        Avg Tweet.fctr     .rnorm 
##          0          0          0          0
```

```r
glb_trnent_df <- add_new_diag_feats(glb_trnent_df)
```

```
##     Tweet                Avg         
##  Length:825         Min.   :-2.0000  
##  Class :character   1st Qu.:-0.6000  
##  Mode  :character   Median : 0.0000  
##                     Mean   :-0.1908  
##                     3rd Qu.: 0.2000  
##                     Max.   : 2.0000  
##                                      
##                                                                                                                                   Tweet.fctr 
##  {#IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO  @apple @itune}:  8  
##  {#IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO @apple @itune} :  7  
##  #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO @apple @itune #IPOD GENERATION                                     :  6  
##  #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO @apple @itune                                                      :  4  
##  C #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO @apple @itune                                                    :  4  
##  @apple                                                                                                                                :  3  
##  (Other)                                                                                                                               :793  
##      .rnorm         
##  Min.   :-3.825193  
##  1st Qu.:-0.714436  
##  Median : 0.002213  
##  Mean   :-0.024802  
##  3rd Qu.: 0.663992  
##  Max.   : 3.851758  
##                     
##      Tweet        Avg Tweet.fctr     .rnorm 
##          0          0          0          0
```

```r
glb_newent_df <- add_new_diag_feats(glb_newent_df)
```

```
##     Tweet                Avg         
##  Length:356         Min.   :-2.0000  
##  Class :character   1st Qu.:-0.6000  
##  Mode  :character   Median : 0.0000  
##                     Mean   :-0.1983  
##                     3rd Qu.: 0.2000  
##                     Max.   : 2.0000  
##                                      
##                                                                                                                                  Tweet.fctr 
##  C #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO @apple @itune                                                   :  4  
##  FOLLOW @APPLE NOW                                                                                                                    :  3  
##  A #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO @apple @itune                                                   :  2  
##  {#IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO @apple @itune}:  2  
##  @battalalgoos @apple                                                                                                                 :  2  
##  iOS 7 is so fricking smooth & beautiful!! #ThanxApple @Apple                                                                         :  1  
##  (Other)                                                                                                                              :342  
##      .rnorm         
##  Min.   :-2.940898  
##  1st Qu.:-0.513629  
##  Median :-0.004035  
##  Mean   : 0.028237  
##  3rd Qu.: 0.621616  
##  Max.   : 2.519880  
##                     
##      Tweet        Avg Tweet.fctr     .rnorm 
##          0          0          0          0
```

```r
# Histogram of predictor in glb_trnent_df & glb_newent_df
plot_df <- rbind(cbind(glb_trnent_df[, glb_rsp_var_raw, FALSE], data.frame(.data="Training")),
                 cbind(glb_trnent_df[, glb_rsp_var_raw, FALSE], data.frame(.data="New")))
print(myplot_histogram(plot_df, glb_rsp_var_raw) + facet_wrap(~ .data))
```

```
## stat_bin: binwidth defaulted to range/30. Use 'binwidth = x' to adjust this.
## stat_bin: binwidth defaulted to range/30. Use 'binwidth = x' to adjust this.
```

![](Apple_Tweets_files/figure-html/inspectORexplore_data-1.png) 

```r
# used later in encode.retype.data chunk
glb_display_class_dstrb <- function(var) {
    plot_df <- rbind(cbind(glb_trnent_df[, var, FALSE], 
                           data.frame(.data="Training")),
                     cbind(glb_trnent_df[, var, FALSE], 
                           data.frame(.data="New")))
    xtab_df <- mycreate_xtab(plot_df, c(".data", var))
    rownames(xtab_df) <- xtab_df$.data
    xtab_df <- subset(xtab_df, select=-.data)
    print(xtab_df / rowSums(xtab_df))    
}
if (glb_is_classification) glb_display_class_dstrb(glb_rsp_var_raw)
```

```
## Loading required package: reshape2
```

```
##              Avg.-2   Avg.-1.8    Avg.-1.6   Avg.-1.4   Avg.-1.2
## New      0.01454545 0.01818182 0.007272727 0.01939394 0.03272727
## Training 0.01454545 0.01818182 0.007272727 0.01939394 0.03272727
##              Avg.-1   Avg.-0.8   Avg.-0.6   Avg.-0.4  Avg.-0.2     Avg.0
## New      0.06060606 0.06424242 0.06060606 0.07151515 0.1078788 0.2860606
## Training 0.06060606 0.06424242 0.06060606 0.07151515 0.1078788 0.2860606
##            Avg.0.2    Avg.0.4    Avg.0.6    Avg.0.8      Avg.1    Avg.1.2
## New      0.0969697 0.04242424 0.03757576 0.02787879 0.02181818 0.00969697
## Training 0.0969697 0.04242424 0.03757576 0.02787879 0.02181818 0.00969697
##              Avg.1.4     Avg.1.6     Avg.1.8       Avg.2
## New      0.007272727 0.007272727 0.004848485 0.001212121
## Training 0.007272727 0.007272727 0.004848485 0.001212121
```

```r
# Check for duplicates in glb_id_vars
if (length(glb_id_vars) > 0) {
    id_vars_dups_df <- subset(id_vars_df <- 
            mycreate_tbl_df(glb_entity_df[, glb_id_vars, FALSE], glb_id_vars),
                                .freq > 1)
} else {
    tmp_entity_df <- glb_entity_df
    tmp_entity_df$.rownames <- rownames(tmp_entity_df)
    id_vars_dups_df <- subset(id_vars_df <- 
            mycreate_tbl_df(tmp_entity_df[, ".rownames", FALSE], ".rownames"),
                                .freq > 1)
}

if (nrow(id_vars_dups_df) > 0) {
    warning("Duplicates found in glb_id_vars data:", nrow(id_vars_dups_df))
    myprint_df(id_vars_dups_df)
    } else {
        # glb_id_vars are unique across obs in both glb_<>_df
        glb_exclude_vars_as_features <- union(glb_exclude_vars_as_features, glb_id_vars)
}

#pairs(subset(glb_trnent_df, select=-c(col_symbol)))
# Check for glb_newent_df & glb_trnent_df features range mismatches

# Other diagnostics:
# print(subset(glb_trnent_df, <col1_name> == max(glb_trnent_df$<col1_name>, na.rm=TRUE) & 
#                         <col2_name> <= mean(glb_trnent_df$<col1_name>, na.rm=TRUE)))

# print(glb_trnent_df[which.max(glb_trnent_df$<col_name>),])

# print(<col_name>_freq_glb_trnent_df <- mycreate_tbl_df(glb_trnent_df, "<col_name>"))
# print(which.min(table(glb_trnent_df$<col_name>)))
# print(which.max(table(glb_trnent_df$<col_name>)))
# print(which.max(table(glb_trnent_df$<col1_name>, glb_trnent_df$<col2_name>)[, 2]))
# print(table(glb_trnent_df$<col1_name>, glb_trnent_df$<col2_name>))
# print(table(is.na(glb_trnent_df$<col1_name>), glb_trnent_df$<col2_name>))
# print(table(sign(glb_trnent_df$<col1_name>), glb_trnent_df$<col2_name>))
# print(mycreate_xtab(glb_trnent_df, <col1_name>))
# print(mycreate_xtab(glb_trnent_df, c(<col1_name>, <col2_name>)))
# print(<col1_name>_<col2_name>_xtab_glb_trnent_df <- 
#   mycreate_xtab(glb_trnent_df, c("<col1_name>", "<col2_name>")))
# <col1_name>_<col2_name>_xtab_glb_trnent_df[is.na(<col1_name>_<col2_name>_xtab_glb_trnent_df)] <- 0
# print(<col1_name>_<col2_name>_xtab_glb_trnent_df <- 
#   mutate(<col1_name>_<col2_name>_xtab_glb_trnent_df, 
#             <col3_name>=(<col1_name> * 1.0) / (<col1_name> + <col2_name>))) 

# print(<col2_name>_min_entity_arr <- 
#    sort(tapply(glb_trnent_df$<col1_name>, glb_trnent_df$<col2_name>, min, na.rm=TRUE)))
# print(<col1_name>_na_by_<col2_name>_arr <- 
#    sort(tapply(glb_trnent_df$<col1_name>.NA, glb_trnent_df$<col2_name>, mean, na.rm=TRUE)))

# Other plots:
# print(myplot_box(df=glb_trnent_df, ycol_names="<col1_name>"))
# print(myplot_box(df=glb_trnent_df, ycol_names="<col1_name>", xcol_name="<col2_name>"))
# print(myplot_line(subset(glb_trnent_df, Symbol %in% c("KO", "PG")), 
#                   "Date.my", "StockPrice", facet_row_colnames="Symbol") + 
#     geom_vline(xintercept=as.numeric(as.Date("2003-03-01"))) +
#     geom_vline(xintercept=as.numeric(as.Date("1983-01-01")))        
#         )
# print(myplot_scatter(glb_entity_df, "<col1_name>", "<col2_name>", smooth=TRUE))
# print(myplot_scatter(glb_entity_df, "<col1_name>", "<col2_name>", colorcol_name="<Pred.fctr>") + 
#         geom_point(data=subset(glb_entity_df, <condition>), 
#                     mapping=aes(x=<x_var>, y=<y_var>), color="red", shape=4, size=5))

glb_script_df <- rbind(glb_script_df, 
    data.frame(chunk_label="manage_missing_data", 
        chunk_step_major=max(glb_script_df$chunk_step_major), 
        chunk_step_minor=glb_script_df[nrow(glb_script_df), "chunk_step_minor"]+1,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                    chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed2 inspectORexplore.data                2                1   0.538
## elapsed3   manage_missing_data                2                2   1.620
```

### Step `2`.`2`: manage missing data

```r
# print(sapply(names(glb_trnent_df), function(col) sum(is.na(glb_trnent_df[, col]))))
# print(sapply(names(glb_newent_df), function(col) sum(is.na(glb_newent_df[, col]))))
# glb_trnent_df <- na.omit(glb_trnent_df)
# glb_newent_df <- na.omit(glb_newent_df)
# df[is.na(df)] <- 0

# Not refactored into mydsutils.R since glb_*_df might be reassigned
glb_impute_missing_data <- function(entity_df, newent_df) {
    if (!glb_is_separate_newent_dataset) {
        # Combine entity & newent
        union_df <- rbind(mutate(entity_df, .src = "entity"),
                          mutate(newent_df, .src = "newent"))
        union_imputed_df <- union_df[, setdiff(setdiff(names(entity_df), 
                                                       glb_rsp_var), 
                                               glb_exclude_vars_as_features)]
        print(summary(union_imputed_df))
    
        require(mice)
        set.seed(glb_mice_complete.seed)
        union_imputed_df <- complete(mice(union_imputed_df))
        print(summary(union_imputed_df))
    
        union_df[, names(union_imputed_df)] <- union_imputed_df[, names(union_imputed_df)]
        print(summary(union_df))
#         union_df$.rownames <- rownames(union_df)
#         union_df <- orderBy(~.rownames, union_df)
#         
#         imp_entity_df <- myimport_data(
#             url="<imputed_trnng_url>", 
#             comment="imp_entity_df", force_header=TRUE, print_diagn=TRUE)
#         print(all.equal(subset(union_df, select=-c(.src, .rownames, .rnorm)), 
#                         imp_entity_df))
        
        # Partition again
        glb_trnent_df <<- subset(union_df, .src == "entity", select=-c(.src, .rownames))
        comment(glb_trnent_df) <- "entity_df"
        glb_newent_df <<- subset(union_df, .src == "newent", select=-c(.src, .rownames))
        comment(glb_newent_df) <- "newent_df"
        
        # Generate summaries
        print(summary(entity_df))
        print(sapply(names(entity_df), function(col) sum(is.na(entity_df[, col]))))
        print(summary(newent_df))
        print(sapply(names(newent_df), function(col) sum(is.na(newent_df[, col]))))
    
    } else stop("Not implemented yet")
}

if (glb_impute_na_data) {
    if ((sum(sapply(names(glb_trnent_df), 
                    function(col) sum(is.na(glb_trnent_df[, col])))) > 0) | 
        (sum(sapply(names(glb_newent_df), 
                    function(col) sum(is.na(glb_newent_df[, col])))) > 0))
        glb_impute_missing_data(glb_trnent_df, glb_newent_df)
}    

glb_script_df <- rbind(glb_script_df, 
    data.frame(chunk_label="encodeORretype.data", 
        chunk_step_major=max(glb_script_df$chunk_step_major), 
        chunk_step_minor=glb_script_df[nrow(glb_script_df), "chunk_step_minor"]+1,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                  chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed3 manage_missing_data                2                2   1.620
## elapsed4 encodeORretype.data                2                3   2.161
```

### Step `2`.`3`: encode/retype data

```r
# map_<col_name>_df <- myimport_data(
#     url="<map_url>", 
#     comment="map_<col_name>_df", print_diagn=TRUE)
# map_<col_name>_df <- read.csv(paste0(getwd(), "/data/<file_name>.csv"), strip.white=TRUE)

# glb_trnent_df <- mymap_codes(glb_trnent_df, "<from_col_name>", "<to_col_name>", 
#     map_<to_col_name>_df, map_join_col_name="<map_join_col_name>", 
#                           map_tgt_col_name="<to_col_name>")
# glb_newent_df <- mymap_codes(glb_newent_df, "<from_col_name>", "<to_col_name>", 
#     map_<to_col_name>_df, map_join_col_name="<map_join_col_name>", 
#                           map_tgt_col_name="<to_col_name>")
    					
# glb_trnent_df$<col_name>.fctr <- factor(glb_trnent_df$<col_name>, 
#                     as.factor(union(glb_trnent_df$<col_name>, glb_newent_df$<col_name>)))
# glb_newent_df$<col_name>.fctr <- factor(glb_newent_df$<col_name>, 
#                     as.factor(union(glb_trnent_df$<col_name>, glb_newent_df$<col_name>)))

if (!is.null(glb_map_rsp_raw_to_var)) {
    glb_entity_df[, glb_rsp_var] <- 
        glb_map_rsp_raw_to_var(glb_entity_df[, glb_rsp_var_raw])
    mycheck_map_results(mapd_df=glb_entity_df, 
                        from_col_name=glb_rsp_var_raw, to_col_name=glb_rsp_var)
        
    glb_trnent_df[, glb_rsp_var] <- 
        glb_map_rsp_raw_to_var(glb_trnent_df[, glb_rsp_var_raw])
    mycheck_map_results(mapd_df=glb_trnent_df, 
                        from_col_name=glb_rsp_var_raw, to_col_name=glb_rsp_var)
        
    glb_newent_df[, glb_rsp_var] <- 
        glb_map_rsp_raw_to_var(glb_newent_df[, glb_rsp_var_raw])
    mycheck_map_results(mapd_df=glb_newent_df, 
                        from_col_name=glb_rsp_var_raw, to_col_name=glb_rsp_var)
    
    if (glb_is_classification) glb_display_class_dstrb(glb_rsp_var)
}
```

```
## Loading required package: sqldf
## Loading required package: gsubfn
## Loading required package: proto
## Loading required package: RSQLite
## Loading required package: DBI
## Loading required package: tcltk
```

```
##    Avg Negative.fctr  .n
## 1  0.0             N 337
## 2 -0.2             N 127
## 3  0.2             N 115
## 4 -0.4             N  84
## 5 -0.8             N  76
## 6 -1.0             Y  72
##     Avg Negative.fctr  .n
## 1   0.0             N 337
## 5  -0.8             N  76
## 10 -1.2             Y  39
## 12  1.0             N  25
## 13 -1.4             Y  23
## 17 -1.6             Y   9
##     Avg Negative.fctr .n
## 16  1.2             N 11
## 17 -1.6             Y  9
## 18  1.4             N  9
## 19  1.6             N  9
## 20  1.8             N  5
## 21  2.0             N  2
```

![](Apple_Tweets_files/figure-html/encodeORretype.data-1.png) 

```
##    Avg Negative.fctr  .n
## 1  0.0             N 236
## 2 -0.2             N  89
## 3  0.2             N  80
## 4 -0.4             N  59
## 5 -0.8             N  53
## 6 -1.0             Y  50
##     Avg Negative.fctr .n
## 2  -0.2             N 89
## 12  1.0             N 18
## 13 -1.4             Y 16
## 15 -2.0             Y 12
## 18  1.4             N  6
## 21  2.0             N  1
##     Avg Negative.fctr .n
## 16  1.2             N  8
## 17 -1.6             Y  6
## 18  1.4             N  6
## 19  1.6             N  6
## 20  1.8             N  4
## 21  2.0             N  1
```

![](Apple_Tweets_files/figure-html/encodeORretype.data-2.png) 

```
##    Avg Negative.fctr  .n
## 1  0.0             N 101
## 2 -0.2             N  38
## 3  0.2             N  35
## 4 -0.4             N  25
## 5 -0.8             N  23
## 6 -1.0             Y  22
##     Avg Negative.fctr .n
## 2  -0.2             N 38
## 4  -0.4             N 25
## 9   0.6             N 13
## 12 -1.8             Y  7
## 14  1.0             N  7
## 18  1.4             N  3
##     Avg Negative.fctr .n
## 16 -1.6             Y  3
## 17  1.2             N  3
## 18  1.4             N  3
## 19  1.6             N  3
## 20  1.8             N  1
## 21  2.0             N  1
```

![](Apple_Tweets_files/figure-html/encodeORretype.data-3.png) 

```
##          Negative.fctr.N Negative.fctr.Y
## New            0.8472727       0.1527273
## Training       0.8472727       0.1527273
```

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="extract_features", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                  chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed4 encodeORretype.data                2                3   2.161
## elapsed5    extract_features                3                0   6.366
```

## Step `3`: extract features

```r
#```{r extract_features, cache=FALSE, eval=glb_is_textual}
# Create new features that help prediction
# <col_name>.lag.2 <- lag(zoo(glb_trnent_df$<col_name>), -2, na.pad=TRUE)
# glb_trnent_df[, "<col_name>.lag.2"] <- coredata(<col_name>.lag.2)
# <col_name>.lag.2 <- lag(zoo(glb_newent_df$<col_name>), -2, na.pad=TRUE)
# glb_newent_df[, "<col_name>.lag.2"] <- coredata(<col_name>.lag.2)
# 
# glb_newent_df[1, "<col_name>.lag.2"] <- glb_trnent_df[nrow(glb_trnent_df) - 1, 
#                                                    "<col_name>"]
# glb_newent_df[2, "<col_name>.lag.2"] <- glb_trnent_df[nrow(glb_trnent_df), 
#                                                    "<col_name>"]
                                                   
# glb_trnent_df <- mutate(glb_trnent_df,
#     <new_col_name>=
#                     )

# glb_newent_df <- mutate(glb_newent_df,
#     <new_col_name>=
#                     )

glb_txt_var <- c("Tweet")   # or NULL
glb_append_stop_words <- c("apple") # or NULL
if (glb_is_textual) {
    require(tm)
    
    glb_corpus <- Corpus(VectorSource(glb_entity_df[, glb_txt_var]))
    glb_corpus <- tm_map(glb_corpus, tolower)
    glb_corpus <- tm_map(glb_corpus, PlainTextDocument)
    glb_corpus <- tm_map(glb_corpus, removePunctuation)
    glb_corpus <- tm_map(glb_corpus, removeWords, 
                         c(glb_append_stop_words, stopwords("english")))
    glb_corpus <- tm_map(glb_corpus, stemDocument)    
    
    full_freqs_DTM <- DocumentTermMatrix(glb_corpus)
    full_freqs_vctr <- colSums(as.matrix(full_freqs_DTM))
    names(full_freqs_vctr) <- dimnames(full_freqs_DTM)[[2]]
    full_freqs_df <- as.data.frame(full_freqs_vctr)
    names(full_freqs_df) <- "freq.full"
    full_freqs_df$term <- rownames(full_freqs_df)
    full_freqs_df <- orderBy(~ -freq.full, full_freqs_df)
#    ggplot(full_freqs_df, aes(x=freq.full)) + stat_ecdf()
#    print(myplot_hbar(head(full_freqs_df, 10), "term", "freq.full"))

    sprs_threshold <- 0.995
    sprs_freqs_DTM <- removeSparseTerms(full_freqs_DTM, sprs_threshold)
    sprs_freqs_vctr <- colSums(as.matrix(sprs_freqs_DTM))
    names(sprs_freqs_vctr) <- dimnames(sprs_freqs_DTM)[[2]]
    sprs_freqs_df <- as.data.frame(sprs_freqs_vctr)
    names(sprs_freqs_df) <- "freq.sprs"
    sprs_freqs_df$term <- rownames(sprs_freqs_df)
    sprs_freqs_df <- orderBy(~ -freq.sprs, sprs_freqs_df)
#     ggplot(sprs_freqs_df, aes(x=freq.sprs)) + stat_ecdf()
#     print(myplot_hbar(head(sprs_freqs_df, 10), "term", "freq.sprs"))

    terms_freqs_df <- merge(full_freqs_df, sprs_freqs_df, all.x=TRUE)
    melt_freqs_df <- orderBy(~ -value, melt(terms_freqs_df, id.var="term"))
    print(ggplot(melt_freqs_df, aes(value, color=variable)) + stat_ecdf() + 
              geom_hline(yintercept=1-sprs_threshold, linetype = "dotted"))
    print(myplot_hbar(head(melt_freqs_df, 20), "term", "value", colorcol_name="variable"))
    melt_freqs_df <- orderBy(~ -value, melt(subset(terms_freqs_df, is.na(freq.sprs)), id.var="term"))
    print(myplot_hbar(head(melt_freqs_df, 10), "term", "value", colorcol_name="variable"))

    # Create txt features
    txt_X_df <- as.data.frame(as.matrix(sprs_freqs_DTM))
    colnames(txt_X_df) <- make.names(colnames(txt_X_df))
    # Add dependent variable
    glb_entity_df <- cbind(glb_entity_df, txt_X_df)

    # a working copy of this is reqd in manage.missingdata chunk
    union_df <- rbind(mutate(glb_trnent_df, .src = "trnent"),
                      mutate(glb_newent_df, .src = "newent"))
    if (is.null(glb_id_vars)) {
        union_df$.rownames <- rownames(union_df)
        tmp_entity_df <- glb_entity_df
        tmp_entity_df$.rownames <- rownames(tmp_entity_df)
        mrg_entity_df <- merge(tmp_entity_df, union_df[, c(".src", ".rownames")])
    } else stop("not implemented yet")       
    
    # Partition again
    glb_trnent_df <- subset(mrg_entity_df, .src == "trnent", select=-c(.src, .rownames))
    comment(glb_trnent_df) <- "trnent_df"
    glb_newent_df <- subset(mrg_entity_df, .src == "newent", select=-c(.src, .rownames))
    comment(glb_newent_df) <- "newent_df"

    # Generate summaries
    print(summary(glb_entity_df))
    print(sapply(names(glb_entity_df), function(col) sum(is.na(glb_entity_df[, col]))))
    print(summary(glb_trnent_df))
    print(sapply(names(glb_trnent_df), function(col) sum(is.na(glb_trnent_df[, col]))))
    print(summary(glb_newent_df))
    print(sapply(names(glb_newent_df), function(col) sum(is.na(glb_newent_df[, col]))))
}
```

```
## Loading required package: tm
## Loading required package: NLP
## 
## Attaching package: 'NLP'
## 
## The following object is masked from 'package:ggplot2':
## 
##     annotate
```

```
## Warning: Removed 6 rows containing missing values (geom_path).
```

![](Apple_Tweets_files/figure-html/extract_features-1.png) ![](Apple_Tweets_files/figure-html/extract_features-2.png) 

```
## Warning in data.row.names(row.names, rowsi, i): some row.names duplicated:
## 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,474,475,476,477,478,479,480,481,482,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522,523,524,525,526,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,549,550,551,552,553,554,555,556,557,558,559,560,561,562,563,564,565,566,567,568,569,570,571,572,573,574,575,576,577,578,579,580,581,582,583,584,585,586,587,588,589,590,591,592,593,594,595,596,597,598,599,600,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,630,631,632,633,634,635,636,637,638,639,640,641,642,643,644,645,646,647,648,649,650,651,652,653,654,655,656,657,658,659,660,661,662,663,664,665,666,667,668,669,670,671,672,673,674,675,676,677,678,679,680,681,682,683,684,685,686,687,688,689,690,691,692,693,694,695,696,697,698,699,700,701,702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722,723,724,725,726,727,728,729,730,731,732,733,734,735,736,737,738,739,740,741,742,743,744,745,746,747,748,749,750,751,752,753,754,755,756,757,758,759,760,761,762,763,764,765,766,767,768,769,770,771,772,773,774,775,776,777,778,779,780,781,782,783,784,785,786,787,788,789,790,791,792,793,794,795,796,797,798,799,800,801,802,803,804,805,806,807,808,809,810,811,812,813,814,815,816,817,818,819,820,821,822,823,824,825,826,827,828,829,830,831,832,833,834,835,836,837,838,839,840,841,842,843,844,845,846,847,848,849,850,851,852,853,854,855,856,857,858,859,860,861,862,863,864,865,866,867,868,869,870,871,872,873,874,875,876,877,878,879,880,881,882,883,884,885,886,887,888,889,890,891,892,893,894,895,896,897,898,899,900,901,902,903,904,905,906,907,908,909,910,911,912,913,914,915,916,917,918,919,920,921,922,923,924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960,961,962,963,964,965,966,967,968,969,970,971,972,973,974,975,976,977,978,979,980,981,982,983,984,985,986,987,988,989,990,991,992,993,994,995,996,997,998,999,1000,1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039,1040,1041,1042,1043,1044,1045,1046,1047,1048,1049,1050,1051,1052,1053,1054,1055,1056,1057,1058,1059,1060,1061,1062,1063,1064,1065,1066,1067,1068,1069,1070,1071,1072,1073,1074,1075,1076,1077,1078,1079,1080,1081,1082,1083,1084,1085,1086,1087,1088,1089,1090,1091,1092,1093,1094,1095,1096,1097,1098,1099,1100,1101,1102,1103,1104,1105,1106,1107,1108,1109,1110,1111,1112,1113,1114,1115,1116,1117,1118,1119,1120,1121,1122,1123,1124,1125,1126,1127,1128,1129,1130,1131,1132,1133,1134,1135,1136,1137,1138,1139,1140,1141,1142,1143,1144,1145,1146,1147,1148,1149,1150,1151,1152,1153,1154,1155,1156,1157,1158,1159,1160,1161,1162,1163,1164,1165,1166,1167,1168,1169,1170,1171,1172,1173,1174,1175,1176,1177,1178,1179,1180,1181
## --> row.names NOT used
```

![](Apple_Tweets_files/figure-html/extract_features-3.png) 

```
##     Tweet                Avg         
##  Length:1181        Min.   :-2.0000  
##  Class :character   1st Qu.:-0.6000  
##  Mode  :character   Median : 0.0000  
##                     Mean   :-0.1931  
##                     3rd Qu.: 0.2000  
##                     Max.   : 2.0000  
##                                      
##                                                                                                                                   Tweet.fctr  
##  {#IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO  @apple @itune}:   9  
##  {#IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO @apple @itune} :   9  
##  C #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO @apple @itune                                                    :   8  
##  #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO @apple @itune #IPOD GENERATION                                     :   7  
##  FOLLOW @APPLE NOW                                                                                                                     :   5  
##  #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO @apple @itune                                                      :   5  
##  (Other)                                                                                                                               :1138  
##      .rnorm         Negative.fctr X244tsuyoponzu     X7evenstarz      
##  Min.   :-3.04697   N:999         Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:-0.71210   Y:182         1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :-0.02806                 Median :0.00000   Median :0.000000  
##  Mean   :-0.02602                 Mean   :0.00508   Mean   :0.005927  
##  3rd Qu.: 0.65972                 3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   : 3.30433                 Max.   :1.00000   Max.   :1.000000  
##                                                                       
##      actual              add             alreadi        
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.007621   Mean   :0.00508   Mean   :0.009314  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :2.000000   Max.   :1.00000   Max.   :1.000000  
##                                                         
##      alway               amaz              amazon        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.006774   Mean   :0.005927   Mean   :0.009314  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##     android           announc            anyon              app         
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.00000   Median :0.00000   Median :0.00000  
##  Mean   :0.02032   Mean   :0.01609   Mean   :0.01693   Mean   :0.04572  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :2.00000   Max.   :1.00000   Max.   :2.00000   Max.   :2.00000  
##                                                                         
##       appl            appstor             arent         
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.000000   Median :0.000000  
##  Mean   :0.02879   Mean   :0.005927   Mean   :0.005927  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :2.00000   Max.   :1.000000   Max.   :1.000000  
##                                                         
##       ask               avail               away        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0.00000  
##  Mean   :0.006774   Mean   :0.008467   Mean   :0.00508  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.00000  
##                                                         
##      awesom              back            batteri             best        
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.00000  
##  Mean   :0.006774   Mean   :0.01947   Mean   :0.01693   Mean   :0.01016  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :2.00000   Max.   :2.00000   Max.   :1.00000  
##                                                                          
##      better             big                bit               black        
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.000000   Median :0.000000   Median :0.00000  
##  Mean   :0.01693   Mean   :0.008467   Mean   :0.007621   Mean   :0.01016  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :2.00000   Max.   :1.000000   Max.   :2.000000   Max.   :2.00000  
##                                                                           
##    blackberri           break.            bring         
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.007621   Mean   :0.00508   Mean   :0.005927  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :2.000000  
##                                                         
##     burberri             busi               buy         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0.00000  
##  Mean   :0.007621   Mean   :0.007621   Mean   :0.01778  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :2.000000   Max.   :1.000000   Max.   :1.00000  
##                                                         
##       call               can               cant             carbon        
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.000000  
##  Mean   :0.009314   Mean   :0.04064   Mean   :0.01947   Mean   :0.006774  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :2.000000   Max.   :2.00000   Max.   :1.00000   Max.   :1.000000  
##                                                                           
##       card               care              case         
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.008467   Mean   :0.01016   Mean   :0.005927  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :2.000000  
##                                                         
##       cdp               chang              charg         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.006774   Mean   :0.005927   Mean   :0.006774  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :2.000000   Max.   :1.000000  
##                                                          
##     charger            cheap             china              color        
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.00000   Median :0.000000   Median :0.00000  
##  Mean   :0.01185   Mean   :0.01185   Mean   :0.006774   Mean   :0.01439  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :2.00000   Max.   :2.000000   Max.   :2.00000  
##                                                                          
##      colour              come           compani         condescens      
##  Min.   :0.000000   Min.   :0.0000   Min.   :0.0000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.0000   1st Qu.:0.0000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.0000   Median :0.0000   Median :0.000000  
##  Mean   :0.006774   Mean   :0.0271   Mean   :0.0127   Mean   :0.005927  
##  3rd Qu.:0.000000   3rd Qu.:0.0000   3rd Qu.:0.0000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.0000   Max.   :1.0000   Max.   :1.000000  
##                                                                         
##      condom             copi              crack             creat         
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.00508   Mean   :0.005927   Mean   :0.00508   Mean   :0.006774  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :1.00000   Max.   :1.000000  
##                                                                           
##      custom              darn               data         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.009314   Mean   :0.005927   Mean   :0.009314  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##       date              day               dear            design        
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.0000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.0000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.00000   Median :0.0000   Median :0.000000  
##  Mean   :0.00508   Mean   :0.01524   Mean   :0.0127   Mean   :0.007621  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.0000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :2.00000   Max.   :1.0000   Max.   :1.000000  
##                                                                         
##     develop             devic             didnt              die          
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.000000  
##  Mean   :0.006774   Mean   :0.01609   Mean   :0.01016   Mean   :0.008467  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :2.000000   Max.   :1.00000   Max.   :1.00000   Max.   :1.000000  
##                                                                           
##      differ          disappoint         discontinu      
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.000000   Median :0.000000  
##  Mean   :0.01016   Mean   :0.006774   Mean   :0.006774  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :1.000000  
##                                                         
##      divulg             doesnt             done               dont        
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000   Median :0.000000   Median :0.00000  
##  Mean   :0.005927   Mean   :0.01101   Mean   :0.009314   Mean   :0.04234  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :1.000000   Max.   :2.00000  
##                                                                           
##     download            drop              email             emiss         
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.00508   Mean   :0.007621   Mean   :0.00508   Mean   :0.006774  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :1.00000   Max.   :1.000000  
##                                                                           
##      emoji              even             event              ever         
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.00000   Median :0.00000   Median :0.000000  
##  Mean   :0.01016   Mean   :0.01101   Mean   :0.00508   Mean   :0.007621  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :2.00000   Max.   :1.00000   Max.   :1.00000   Max.   :1.000000  
##                                                                          
##      everi           everyth           facebook            fail         
##  Min.   :0.0000   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.0000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.0000   Median :0.00000   Median :0.00000   Median :0.000000  
##  Mean   :0.0127   Mean   :0.00508   Mean   :0.01101   Mean   :0.007621  
##  3rd Qu.:0.0000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :2.0000   Max.   :1.00000   Max.   :1.00000   Max.   :1.000000  
##                                                                         
##      featur             feel              femal             figur        
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.000000   Median :0.00000   Median :0.00000  
##  Mean   :0.01016   Mean   :0.006774   Mean   :0.00508   Mean   :0.00508  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :2.00000   Max.   :1.000000   Max.   :1.00000   Max.   :1.00000  
##                                                                          
##      final              finger          fingerprint           fire        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0.00000   Median :0.00000  
##  Mean   :0.005927   Mean   :0.008467   Mean   :0.02879   Mean   :0.00508  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :2.00000   Max.   :1.00000  
##                                                                           
##      first               fix             follow            freak        
##  Min.   :0.000000   Min.   :0.0000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.0000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.0000   Median :0.00000   Median :0.00000  
##  Mean   :0.005927   Mean   :0.0127   Mean   :0.01016   Mean   :0.04742  
##  3rd Qu.:0.000000   3rd Qu.:0.0000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.0000   Max.   :1.00000   Max.   :2.00000  
##                                                                         
##       free              fun              generat        
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.000000   Median :0.000000  
##  Mean   :0.01185   Mean   :0.006774   Mean   :0.008467  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :1.000000  
##                                                         
##      genius              get               give              gold        
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.00000  
##  Mean   :0.005927   Mean   :0.06351   Mean   :0.01355   Mean   :0.01016  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :2.00000   Max.   :1.00000   Max.   :2.00000  
##                                                                          
##      gonna               good             googl              got         
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.00000  
##  Mean   :0.007621   Mean   :0.01355   Mean   :0.02286   Mean   :0.01101  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :2.00000   Max.   :1.00000  
##                                                                          
##      great              guess               guy         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0.00000  
##  Mean   :0.009314   Mean   :0.006774   Mean   :0.01355  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :2.00000  
##                                                         
##      happen             happi              hate              help        
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.00000  
##  Mean   :0.008467   Mean   :0.00508   Mean   :0.01609   Mean   :0.01524  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :1.00000   Max.   :1.00000  
##                                                                          
##       hey               hope               hour         
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.000000   Median :0.000000  
##  Mean   :0.01355   Mean   :0.009314   Mean   :0.007621  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :1.000000  
##                                                         
##  httpbitly18xc8dk     ibrooklynb            idea         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.007621   Mean   :0.006774   Mean   :0.007621  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##       ill              imessag            impress        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.005927   Mean   :0.005927   Mean   :0.006774  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :2.000000  
##                                                          
##      improv             innov            instead           internet      
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.00000  
##  Mean   :0.005927   Mean   :0.01355   Mean   :0.01016   Mean   :0.00508  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :1.00000   Max.   :1.00000  
##                                                                          
##       ios7              ipad             iphon          iphone4       
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.00000   Median :0.000   Median :0.00000  
##  Mean   :0.02202   Mean   :0.07705   Mean   :0.243   Mean   :0.00508  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :2.00000   Max.   :3.000   Max.   :1.00000  
##                                                                       
##     iphone5           iphone5c           iphoto              ipod        
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.00000   Median :0.000000   Median :0.00000  
##  Mean   :0.03556   Mean   :0.02964   Mean   :0.008467   Mean   :0.06181  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :2.00000   Max.   :2.00000   Max.   :1.000000   Max.   :2.00000  
##                                                                          
##  ipodplayerpromo       isnt               itun             ive         
##  Min.   :0.0000   Min.   :0.000000   Min.   :0.0000   Min.   :0.00000  
##  1st Qu.:0.0000   1st Qu.:0.000000   1st Qu.:0.0000   1st Qu.:0.00000  
##  Median :0.0000   Median :0.000000   Median :0.0000   Median :0.00000  
##  Mean   :0.0508   Mean   :0.009314   Mean   :0.1025   Mean   :0.00508  
##  3rd Qu.:0.0000   3rd Qu.:0.000000   3rd Qu.:0.0000   3rd Qu.:0.00000  
##  Max.   :2.0000   Max.   :1.000000   Max.   :3.0000   Max.   :1.00000  
##                                                                        
##       job                just             keynot              know        
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000   Median :0.000000   Median :0.00000  
##  Mean   :0.006774   Mean   :0.05165   Mean   :0.006774   Mean   :0.01524  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :2.00000   Max.   :1.000000   Max.   :1.00000  
##                                                                           
##       last              launch             let               life         
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.000000  
##  Mean   :0.005927   Mean   :0.00508   Mean   :0.01439   Mean   :0.008467  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :2.000000   Max.   :1.00000   Max.   :1.00000   Max.   :1.000000  
##                                                                           
##       like              line              lmao              lock        
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.00000   Median :0.00000   Median :0.00000  
##  Mean   :0.04488   Mean   :0.00508   Mean   :0.00508   Mean   :0.00508  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :2.00000   Max.   :1.00000   Max.   :1.00000   Max.   :1.00000  
##                                                                         
##       lol               look              los                lost        
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.00000   Median :0.000000   Median :0.00000  
##  Mean   :0.01778   Mean   :0.02625   Mean   :0.005927   Mean   :0.00508  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :2.00000   Max.   :1.00000   Max.   :1.000000   Max.   :1.00000  
##                                                                          
##       love              mac              macbook        
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.000000   Median :0.000000  
##  Mean   :0.02117   Mean   :0.008467   Mean   :0.009314  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :1.000000  
##                                                         
##       made               make              man          
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.008467   Mean   :0.05165   Mean   :0.007621  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :2.00000   Max.   :1.000000  
##                                                         
##       mani              market             mayb              mean         
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.000000  
##  Mean   :0.005927   Mean   :0.01778   Mean   :0.00508   Mean   :0.005927  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :2.00000   Max.   :1.00000   Max.   :1.000000  
##                                                                           
##    microsoft          mishiza              miss              mobil        
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.000000   Median :0.000000   Median :0.00000  
##  Mean   :0.02202   Mean   :0.006774   Mean   :0.007621   Mean   :0.01439  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :2.00000   Max.   :1.000000   Max.   :1.000000   Max.   :2.00000  
##                                                                           
##      money             motorola             move              much        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0.00000   Median :0.00000  
##  Mean   :0.006774   Mean   :0.007621   Mean   :0.00508   Mean   :0.01101  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.00000   Max.   :2.00000  
##                                                                           
##      music            natz0711             need             never         
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.00508   Mean   :0.005927   Mean   :0.03302   Mean   :0.008467  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :1.00000   Max.   :1.000000  
##                                                                           
##       new               news             next.              nfc         
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.00000   Median :0.00000   Median :0.00000  
##  Mean   :0.09568   Mean   :0.01185   Mean   :0.01439   Mean   :0.00508  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :2.00000   Max.   :2.00000   Max.   :1.00000   Max.   :1.00000  
##                                                                         
##      nokia              noth               now               nsa          
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.01355   Mean   :0.007621   Mean   :0.03895   Mean   :0.008467  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :2.00000   Max.   :1.000000  
##                                                                           
##      nuevo             offer               old                one         
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.000000   Median :0.000000   Median :0.00000  
##  Mean   :0.00508   Mean   :0.006774   Mean   :0.008467   Mean   :0.03218  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :2.000000   Max.   :1.000000   Max.   :2.00000  
##                                                                           
##       page               para             peopl            perfect       
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.00000  
##  Mean   :0.005927   Mean   :0.00508   Mean   :0.01355   Mean   :0.00508  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :1.00000   Max.   :1.00000  
##                                                                          
##      person            phone             photog          photographi      
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.00000   Median :0.000000   Median :0.000000  
##  Mean   :0.00508   Mean   :0.07282   Mean   :0.005927   Mean   :0.005927  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :2.00000   Max.   :1.000000   Max.   :1.000000  
##                                                                           
##      pictur            plastic              play             pleas        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0.00000   Median :0.00000  
##  Mean   :0.008467   Mean   :0.008467   Mean   :0.00508   Mean   :0.01863  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.00000   Max.   :1.00000  
##                                                                           
##       ppl              preorder           price             print        
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.00000  
##  Mean   :0.006774   Mean   :0.01524   Mean   :0.01355   Mean   :0.01185  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :3.000000   Max.   :1.00000   Max.   :2.00000   Max.   :1.00000  
##                                                                          
##       pro              problem            product            promo        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0.00000   Median :0.00000  
##  Mean   :0.005927   Mean   :0.005927   Mean   :0.01355   Mean   :0.01693  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.00000   Max.   :1.00000  
##                                                                           
##  promoipodplayerpromo      put                que         
##  Min.   :0.00000      Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.00000      1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.00000      Median :0.000000   Median :0.00000  
##  Mean   :0.03556      Mean   :0.007621   Mean   :0.01016  
##  3rd Qu.:0.00000      3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.00000      Max.   :1.000000   Max.   :2.00000  
##                                                           
##      quiet               read             realli          recommend       
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.000000  
##  Mean   :0.005927   Mean   :0.00508   Mean   :0.02456   Mean   :0.005927  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :2.00000   Max.   :1.000000  
##                                                                           
##      refus              releas            right              said         
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.000000  
##  Mean   :0.006774   Mean   :0.02117   Mean   :0.01101   Mean   :0.005927  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :1.00000   Max.   :1.000000  
##                                                                           
##     samsung          samsungsa             say             scanner        
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.02032   Mean   :0.006774   Mean   :0.01778   Mean   :0.008467  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :1.00000   Max.   :1.000000  
##                                                                           
##      screen            secur               see          
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.000000   Median :0.000000  
##  Mean   :0.01185   Mean   :0.005927   Mean   :0.009314  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :2.00000   Max.   :1.000000   Max.   :1.000000  
##                                                         
##       seem               sell               send         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.008467   Mean   :0.008467   Mean   :0.006774  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##      servic            shame             share              short        
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.00000   Median :0.000000   Median :0.00000  
##  Mean   :0.01355   Mean   :0.00508   Mean   :0.006774   Mean   :0.00508  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :1.00000   Max.   :2.000000   Max.   :1.00000  
##                                                                          
##       show              simpl               sinc         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.005927   Mean   :0.005927   Mean   :0.006774  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :2.000000   Max.   :1.000000  
##                                                          
##       siri             smart           smartphon            someth        
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.00000   Median :0.000000   Median :0.000000  
##  Mean   :0.01016   Mean   :0.00508   Mean   :0.005927   Mean   :0.006774  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.00000   Max.   :1.000000   Max.   :1.000000  
##                                                                           
##       soon             stand              start              steve        
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.000000   Median :0.000000   Median :0.00000  
##  Mean   :0.00508   Mean   :0.006774   Mean   :0.006774   Mean   :0.01016  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :1.000000   Max.   :1.00000  
##                                                                           
##      still              stop             store             stuff        
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.00000   Median :0.00000   Median :0.00000  
##  Mean   :0.01524   Mean   :0.01185   Mean   :0.03387   Mean   :0.01101  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :1.00000   Max.   :2.00000   Max.   :1.00000  
##                                                                         
##      stupid              suck             support        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.005927   Mean   :0.005927   Mean   :0.008467  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :2.000000   Max.   :1.000000   Max.   :2.000000  
##                                                          
##       sure             switch              take             talk        
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.0000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.0000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.000000   Median :0.0000   Median :0.00000  
##  Mean   :0.01355   Mean   :0.007621   Mean   :0.0127   Mean   :0.00508  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.0000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :1.0000   Max.   :1.00000  
##                                                                         
##       team              tech           technolog             tell         
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.00000   Median :0.000000   Median :0.000000  
##  Mean   :0.00508   Mean   :0.01101   Mean   :0.006774   Mean   :0.009314  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.00000   Max.   :1.000000   Max.   :1.000000  
##                                                                           
##       text              thank              that             theyr        
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.00000  
##  Mean   :0.009314   Mean   :0.02794   Mean   :0.01439   Mean   :0.00508  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :2.000000   Max.   :2.00000   Max.   :1.00000   Max.   :1.00000  
##                                                                          
##      thing            think              tho              thought        
##  Min.   :0.0000   Min.   :0.00000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.0000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.0000   Median :0.00000   Median :0.000000   Median :0.000000  
##  Mean   :0.0127   Mean   :0.02456   Mean   :0.005927   Mean   :0.006774  
##  3rd Qu.:0.0000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.0000   Max.   :2.00000   Max.   :1.000000   Max.   :1.000000  
##                                                                          
##       time             today             togeth            touch        
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.00000   Median :0.00000   Median :0.00000  
##  Mean   :0.02879   Mean   :0.01101   Mean   :0.00508   Mean   :0.00508  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :2.00000   Max.   :1.00000   Max.   :1.00000   Max.   :1.00000  
##                                                                         
##     touchid              tri              true              turn        
##  Min.   :0.000000   Min.   :0.0000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.0000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.0000   Median :0.00000   Median :0.00000  
##  Mean   :0.006774   Mean   :0.0127   Mean   :0.00508   Mean   :0.01016  
##  3rd Qu.:0.000000   3rd Qu.:0.0000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :2.0000   Max.   :1.00000   Max.   :2.00000  
##                                                                         
##     twitter             two               updat             upgrad        
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.02032   Mean   :0.005927   Mean   :0.01947   Mean   :0.009314  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :1.00000   Max.   :2.000000  
##                                                                           
##       use               user              via              video        
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.00000   Median :0.00000   Median :0.00000  
##  Mean   :0.03472   Mean   :0.00508   Mean   :0.01693   Mean   :0.01016  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :1.00000   Max.   :1.00000   Max.   :2.00000  
##                                                                         
##       wait              want            watch              way         
##  Min.   :0.00000   Min.   :0.0000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.0000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.0000   Median :0.00000   Median :0.00000  
##  Mean   :0.01609   Mean   :0.0254   Mean   :0.01016   Mean   :0.01185  
##  3rd Qu.:0.00000   3rd Qu.:0.0000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :1.0000   Max.   :1.00000   Max.   :2.00000  
##                                                                        
##       week               well              what         
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.008467   Mean   :0.01863   Mean   :0.005927  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :2.000000   Max.   :2.00000   Max.   :1.000000  
##                                                         
##      white               will          windowsphon            wish        
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000   Median :0.000000   Median :0.00000  
##  Mean   :0.005927   Mean   :0.04657   Mean   :0.007621   Mean   :0.01016  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :2.000000   Max.   :2.00000   Max.   :1.000000   Max.   :1.00000  
##                                                                           
##     without            wonder             wont              work        
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.00000   Median :0.00000   Median :0.00000  
##  Mean   :0.00508   Mean   :0.00508   Mean   :0.01439   Mean   :0.02117  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :1.00000   Max.   :2.00000   Max.   :2.00000  
##                                                                         
##      world             worst               wow                wtf         
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.000000   Median :0.000000   Median :0.00000  
##  Mean   :0.00508   Mean   :0.006774   Mean   :0.007621   Mean   :0.01185  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :1.000000   Max.   :1.00000  
##                                                                           
##       yall              year              yes                yet         
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.00000   Median :0.000000   Median :0.00000  
##  Mean   :0.01355   Mean   :0.01609   Mean   :0.005927   Mean   :0.01101  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :2.00000   Max.   :2.00000   Max.   :1.000000   Max.   :1.00000  
##                                                                          
##       yooo               your        
##  Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000  
##  Mean   :0.005927   Mean   :0.01101  
##  3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :2.00000  
##                                      
##                Tweet                  Avg           Tweet.fctr 
##                    0                    0                    0 
##               .rnorm        Negative.fctr       X244tsuyoponzu 
##                    0                    0                    0 
##          X7evenstarz               actual                  add 
##                    0                    0                    0 
##              alreadi                alway                 amaz 
##                    0                    0                    0 
##               amazon              android              announc 
##                    0                    0                    0 
##                anyon                  app                 appl 
##                    0                    0                    0 
##              appstor                arent                  ask 
##                    0                    0                    0 
##                avail                 away               awesom 
##                    0                    0                    0 
##                 back              batteri                 best 
##                    0                    0                    0 
##               better                  big                  bit 
##                    0                    0                    0 
##                black           blackberri               break. 
##                    0                    0                    0 
##                bring             burberri                 busi 
##                    0                    0                    0 
##                  buy                 call                  can 
##                    0                    0                    0 
##                 cant               carbon                 card 
##                    0                    0                    0 
##                 care                 case                  cdp 
##                    0                    0                    0 
##                chang                charg              charger 
##                    0                    0                    0 
##                cheap                china                color 
##                    0                    0                    0 
##               colour                 come              compani 
##                    0                    0                    0 
##           condescens               condom                 copi 
##                    0                    0                    0 
##                crack                creat               custom 
##                    0                    0                    0 
##                 darn                 data                 date 
##                    0                    0                    0 
##                  day                 dear               design 
##                    0                    0                    0 
##              develop                devic                didnt 
##                    0                    0                    0 
##                  die               differ           disappoint 
##                    0                    0                    0 
##           discontinu               divulg               doesnt 
##                    0                    0                    0 
##                 done                 dont             download 
##                    0                    0                    0 
##                 drop                email                emiss 
##                    0                    0                    0 
##                emoji                 even                event 
##                    0                    0                    0 
##                 ever                everi              everyth 
##                    0                    0                    0 
##             facebook                 fail               featur 
##                    0                    0                    0 
##                 feel                femal                figur 
##                    0                    0                    0 
##                final               finger          fingerprint 
##                    0                    0                    0 
##                 fire                first                  fix 
##                    0                    0                    0 
##               follow                freak                 free 
##                    0                    0                    0 
##                  fun              generat               genius 
##                    0                    0                    0 
##                  get                 give                 gold 
##                    0                    0                    0 
##                gonna                 good                googl 
##                    0                    0                    0 
##                  got                great                guess 
##                    0                    0                    0 
##                  guy               happen                happi 
##                    0                    0                    0 
##                 hate                 help                  hey 
##                    0                    0                    0 
##                 hope                 hour     httpbitly18xc8dk 
##                    0                    0                    0 
##           ibrooklynb                 idea                  ill 
##                    0                    0                    0 
##              imessag              impress               improv 
##                    0                    0                    0 
##                innov              instead             internet 
##                    0                    0                    0 
##                 ios7                 ipad                iphon 
##                    0                    0                    0 
##              iphone4              iphone5             iphone5c 
##                    0                    0                    0 
##               iphoto                 ipod      ipodplayerpromo 
##                    0                    0                    0 
##                 isnt                 itun                  ive 
##                    0                    0                    0 
##                  job                 just               keynot 
##                    0                    0                    0 
##                 know                 last               launch 
##                    0                    0                    0 
##                  let                 life                 like 
##                    0                    0                    0 
##                 line                 lmao                 lock 
##                    0                    0                    0 
##                  lol                 look                  los 
##                    0                    0                    0 
##                 lost                 love                  mac 
##                    0                    0                    0 
##              macbook                 made                 make 
##                    0                    0                    0 
##                  man                 mani               market 
##                    0                    0                    0 
##                 mayb                 mean            microsoft 
##                    0                    0                    0 
##              mishiza                 miss                mobil 
##                    0                    0                    0 
##                money             motorola                 move 
##                    0                    0                    0 
##                 much                music             natz0711 
##                    0                    0                    0 
##                 need                never                  new 
##                    0                    0                    0 
##                 news                next.                  nfc 
##                    0                    0                    0 
##                nokia                 noth                  now 
##                    0                    0                    0 
##                  nsa                nuevo                offer 
##                    0                    0                    0 
##                  old                  one                 page 
##                    0                    0                    0 
##                 para                peopl              perfect 
##                    0                    0                    0 
##               person                phone               photog 
##                    0                    0                    0 
##          photographi               pictur              plastic 
##                    0                    0                    0 
##                 play                pleas                  ppl 
##                    0                    0                    0 
##             preorder                price                print 
##                    0                    0                    0 
##                  pro              problem              product 
##                    0                    0                    0 
##                promo promoipodplayerpromo                  put 
##                    0                    0                    0 
##                  que                quiet                 read 
##                    0                    0                    0 
##               realli            recommend                refus 
##                    0                    0                    0 
##               releas                right                 said 
##                    0                    0                    0 
##              samsung            samsungsa                  say 
##                    0                    0                    0 
##              scanner               screen                secur 
##                    0                    0                    0 
##                  see                 seem                 sell 
##                    0                    0                    0 
##                 send               servic                shame 
##                    0                    0                    0 
##                share                short                 show 
##                    0                    0                    0 
##                simpl                 sinc                 siri 
##                    0                    0                    0 
##                smart            smartphon               someth 
##                    0                    0                    0 
##                 soon                stand                start 
##                    0                    0                    0 
##                steve                still                 stop 
##                    0                    0                    0 
##                store                stuff               stupid 
##                    0                    0                    0 
##                 suck              support                 sure 
##                    0                    0                    0 
##               switch                 take                 talk 
##                    0                    0                    0 
##                 team                 tech            technolog 
##                    0                    0                    0 
##                 tell                 text                thank 
##                    0                    0                    0 
##                 that                theyr                thing 
##                    0                    0                    0 
##                think                  tho              thought 
##                    0                    0                    0 
##                 time                today               togeth 
##                    0                    0                    0 
##                touch              touchid                  tri 
##                    0                    0                    0 
##                 true                 turn              twitter 
##                    0                    0                    0 
##                  two                updat               upgrad 
##                    0                    0                    0 
##                  use                 user                  via 
##                    0                    0                    0 
##                video                 wait                 want 
##                    0                    0                    0 
##                watch                  way                 week 
##                    0                    0                    0 
##                 well                 what                white 
##                    0                    0                    0 
##                 will          windowsphon                 wish 
##                    0                    0                    0 
##              without               wonder                 wont 
##                    0                    0                    0 
##                 work                world                worst 
##                    0                    0                    0 
##                  wow                  wtf                 yall 
##                    0                    0                    0 
##                 year                  yes                  yet 
##                    0                    0                    0 
##                 yooo                 your 
##                    0                    0 
##     Tweet                Avg         
##  Length:825         Min.   :-2.0000  
##  Class :character   1st Qu.:-0.6000  
##  Mode  :character   Median : 0.0000  
##                     Mean   :-0.1908  
##                     3rd Qu.: 0.2000  
##                     Max.   : 2.0000  
##                                      
##                                                                                                                                   Tweet.fctr 
##  {#IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO  @apple @itune}:  8  
##  {#IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO @apple @itune} :  7  
##  #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO @apple @itune #IPOD GENERATION                                     :  6  
##  #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO @apple @itune                                                      :  4  
##  C #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO @apple @itune                                                    :  4  
##  @apple                                                                                                                                :  3  
##  (Other)                                                                                                                               :793  
##      .rnorm          Negative.fctr X244tsuyoponzu      X7evenstarz      
##  Min.   :-2.693482   N:699         Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:-0.682172   Y:126         1st Qu.:0.000000   1st Qu.:0.000000  
##  Median : 0.024742                 Median :0.000000   Median :0.000000  
##  Mean   :-0.003363                 Mean   :0.006061   Mean   :0.004849  
##  3rd Qu.: 0.662820                 3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   : 3.304330                 Max.   :1.000000   Max.   :1.000000  
##                                                                         
##      actual              add              alreadi        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.008485   Mean   :0.003636   Mean   :0.003636  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :2.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##      alway               amaz              amazon           android       
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0.00000   Median :0.00000  
##  Mean   :0.004849   Mean   :0.006061   Mean   :0.01091   Mean   :0.02545  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.00000   Max.   :2.00000  
##                                                                           
##     announc            anyon              app              appl        
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.0000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.0000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.00000   Median :0.0000   Median :0.00000  
##  Mean   :0.01697   Mean   :0.01818   Mean   :0.0497   Mean   :0.03273  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.0000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :2.00000   Max.   :2.0000   Max.   :2.00000  
##                                                                        
##     appstor             arent               ask          
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.006061   Mean   :0.006061   Mean   :0.006061  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##      avail               away              awesom        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.009697   Mean   :0.006061   Mean   :0.007273  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##       back            batteri             best              better       
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.00000   Median :0.000000   Median :0.00000  
##  Mean   :0.02303   Mean   :0.01333   Mean   :0.009697   Mean   :0.01818  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :2.00000   Max.   :2.00000   Max.   :1.000000   Max.   :2.00000  
##                                                                          
##       big                bit               black        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0.00000  
##  Mean   :0.008485   Mean   :0.007273   Mean   :0.01212  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :2.000000   Max.   :2.00000  
##                                                         
##    blackberri           break.             bring         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.009697   Mean   :0.007273   Mean   :0.008485  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :2.000000  
##                                                          
##     burberri             busi               buy         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0.00000  
##  Mean   :0.009697   Mean   :0.009697   Mean   :0.01939  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :2.000000   Max.   :1.000000   Max.   :1.00000  
##                                                         
##       call               can               cant             carbon        
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.000000  
##  Mean   :0.008485   Mean   :0.03879   Mean   :0.02061   Mean   :0.008485  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :2.00000   Max.   :1.00000   Max.   :1.000000  
##                                                                           
##       card               care              case         
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.007273   Mean   :0.01091   Mean   :0.006061  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :2.000000  
##                                                         
##       cdp               chang              charg         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.008485   Mean   :0.006061   Mean   :0.006061  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :2.000000   Max.   :1.000000  
##                                                          
##     charger            cheap             china              color        
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.00000   Median :0.000000   Median :0.00000  
##  Mean   :0.01212   Mean   :0.01091   Mean   :0.008485   Mean   :0.01333  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :2.00000   Max.   :2.000000   Max.   :1.00000  
##                                                                          
##      colour              come            compani          condescens      
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.000000  
##  Mean   :0.006061   Mean   :0.02182   Mean   :0.01333   Mean   :0.004849  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :1.00000   Max.   :1.000000  
##                                                                           
##      condom              copi              crack         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.004849   Mean   :0.003636   Mean   :0.004849  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##      creat              custom              darn         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.006061   Mean   :0.007273   Mean   :0.008485  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##       data              date               day               dear        
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.000000   Median :0.00000   Median :0.00000  
##  Mean   :0.01212   Mean   :0.006061   Mean   :0.01455   Mean   :0.01333  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :1.00000   Max.   :1.00000  
##                                                                          
##      design            develop             devic        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0.00000  
##  Mean   :0.006061   Mean   :0.009697   Mean   :0.01818  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :2.000000   Max.   :1.00000  
##                                                         
##      didnt               die               differ        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.008485   Mean   :0.007273   Mean   :0.009697  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##    disappoint         discontinu           divulg        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.006061   Mean   :0.004849   Mean   :0.006061  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##      doesnt              done               dont         download       
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.00   Median :0.000000  
##  Mean   :0.008485   Mean   :0.007273   Mean   :0.04   Mean   :0.004849  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :2.00   Max.   :1.000000  
##                                                                         
##       drop              email              emiss         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.008485   Mean   :0.004849   Mean   :0.008485  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##      emoji               even              event         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.008485   Mean   :0.004849   Mean   :0.007273  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##       ever              everi            everyth            facebook      
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000   Median :0.000000   Median :0.00000  
##  Mean   :0.008485   Mean   :0.01455   Mean   :0.002424   Mean   :0.01333  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :2.00000   Max.   :1.000000   Max.   :1.00000  
##                                                                           
##       fail              featur              feel         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.008485   Mean   :0.007273   Mean   :0.008485  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##      femal              figur              final         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.004849   Mean   :0.004849   Mean   :0.003636  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##      finger         fingerprint           fire              first         
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.00000   Median :0.000000   Median :0.000000  
##  Mean   :0.01091   Mean   :0.03394   Mean   :0.007273   Mean   :0.008485  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :2.00000   Max.   :1.000000   Max.   :1.000000  
##                                                                           
##       fix               follow             freak              free        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0.00000   Median :0.00000  
##  Mean   :0.008485   Mean   :0.009697   Mean   :0.04485   Mean   :0.01212  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :2.00000   Max.   :1.00000  
##                                                                           
##       fun              generat             genius        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.008485   Mean   :0.009697   Mean   :0.003636  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##       get               give              gold             gonna         
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.00000   Median :0.00000   Median :0.000000  
##  Mean   :0.06667   Mean   :0.01333   Mean   :0.01212   Mean   :0.006061  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :2.00000   Max.   :1.00000   Max.   :2.00000   Max.   :1.000000  
##                                                                          
##       good             googl              got               great         
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.00000   Median :0.000000   Median :0.000000  
##  Mean   :0.01333   Mean   :0.02303   Mean   :0.008485   Mean   :0.007273  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :2.00000   Max.   :1.000000   Max.   :1.000000  
##                                                                           
##      guess               guy              happen        
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.003636   Mean   :0.01091   Mean   :0.006061  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :2.00000   Max.   :1.000000  
##                                                         
##      happi               hate              help              hey         
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.00000  
##  Mean   :0.006061   Mean   :0.01333   Mean   :0.01212   Mean   :0.01333  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :1.00000   Max.   :1.00000  
##                                                                          
##       hope               hour          httpbitly18xc8dk  
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.009697   Mean   :0.008485   Mean   :0.008485  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##    ibrooklynb            idea               ill          
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.006061   Mean   :0.004849   Mean   :0.006061  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##     imessag            impress             improv        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.007273   Mean   :0.008485   Mean   :0.008485  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :2.000000   Max.   :1.000000  
##                                                          
##      innov            instead           internet             ios7        
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.00000   Median :0.000000   Median :0.00000  
##  Mean   :0.01212   Mean   :0.01091   Mean   :0.004849   Mean   :0.01576  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :1.00000   Max.   :1.000000   Max.   :1.00000  
##                                                                          
##       ipad             iphon           iphone4            iphone5       
##  Min.   :0.00000   Min.   :0.0000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.0000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.0000   Median :0.000000   Median :0.00000  
##  Mean   :0.08606   Mean   :0.2473   Mean   :0.004849   Mean   :0.03758  
##  3rd Qu.:0.00000   3rd Qu.:0.0000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :2.00000   Max.   :3.0000   Max.   :1.000000   Max.   :2.00000  
##                                                                         
##     iphone5c           iphoto              ipod         ipodplayerpromo  
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.000000   Median :0.00000   Median :0.00000  
##  Mean   :0.02667   Mean   :0.008485   Mean   :0.06788   Mean   :0.05576  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :2.00000   Max.   :1.000000   Max.   :2.00000   Max.   :2.00000  
##                                                                          
##       isnt               itun             ive                job          
##  Min.   :0.000000   Min.   :0.0000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.0000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.0000   Median :0.000000   Median :0.000000  
##  Mean   :0.008485   Mean   :0.1079   Mean   :0.006061   Mean   :0.004849  
##  3rd Qu.:0.000000   3rd Qu.:0.0000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :3.0000   Max.   :1.000000   Max.   :1.000000  
##                                                                           
##       just             keynot              know              last         
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.05212   Mean   :0.007273   Mean   :0.01333   Mean   :0.004849  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :1.00000   Max.   :2.000000  
##                                                                           
##      launch              let               life               like        
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000   Median :0.000000   Median :0.00000  
##  Mean   :0.006061   Mean   :0.01333   Mean   :0.009697   Mean   :0.05091  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :1.000000   Max.   :2.00000  
##                                                                           
##       line               lmao               lock         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.004849   Mean   :0.004849   Mean   :0.006061  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##       lol               look             los                lost         
##  Min.   :0.00000   Min.   :0.0000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.0000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.0000   Median :0.000000   Median :0.000000  
##  Mean   :0.01576   Mean   :0.0303   Mean   :0.004849   Mean   :0.006061  
##  3rd Qu.:0.00000   3rd Qu.:0.0000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.0000   Max.   :1.000000   Max.   :1.000000  
##                                                                          
##       love              mac              macbook        
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.000000   Median :0.000000  
##  Mean   :0.01818   Mean   :0.009697   Mean   :0.008485  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :1.000000  
##                                                         
##       made               make              man          
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.008485   Mean   :0.05455   Mean   :0.006061  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :2.00000   Max.   :1.000000  
##                                                         
##       mani              market             mayb         
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.004849   Mean   :0.01818   Mean   :0.007273  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :2.00000   Max.   :1.000000  
##                                                         
##       mean            microsoft          mishiza        
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.008485   Mean   :0.02545   Mean   :0.007273  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :2.00000   Max.   :1.000000  
##                                                         
##       miss              mobil             money         
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.008485   Mean   :0.01212   Mean   :0.004849  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :2.00000   Max.   :1.000000  
##                                                         
##     motorola             move               much        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0.00000  
##  Mean   :0.008485   Mean   :0.004849   Mean   :0.01091  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :2.00000  
##                                                         
##      music             natz0711             need            never         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.0000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.0000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.0000   Median :0.000000  
##  Mean   :0.002424   Mean   :0.007273   Mean   :0.0303   Mean   :0.009697  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.0000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.0000   Max.   :1.000000  
##                                                                           
##       new               news             next.              nfc          
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.00000   Median :0.00000   Median :0.000000  
##  Mean   :0.09455   Mean   :0.01333   Mean   :0.01697   Mean   :0.003636  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :2.00000   Max.   :2.00000   Max.   :1.00000   Max.   :1.000000  
##                                                                          
##      nokia              noth               now               nsa          
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.01455   Mean   :0.006061   Mean   :0.04121   Mean   :0.009697  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :2.00000   Max.   :1.000000  
##                                                                           
##      nuevo              offer               old               one         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0.00000   Median :0.00000  
##  Mean   :0.004849   Mean   :0.007273   Mean   :0.01091   Mean   :0.03636  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.00000   Max.   :2.00000  
##                                                                           
##       page               para              peopl        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0.00000  
##  Mean   :0.006061   Mean   :0.007273   Mean   :0.01576  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.00000  
##                                                         
##     perfect             person             phone            photog        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.0000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.0000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.0000   Median :0.000000  
##  Mean   :0.004849   Mean   :0.003636   Mean   :0.0703   Mean   :0.004849  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.0000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :2.0000   Max.   :1.000000  
##                                                                           
##   photographi           pictur            plastic        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.004849   Mean   :0.009697   Mean   :0.009697  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##       play              pleas              ppl              preorder      
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000   Median :0.000000   Median :0.00000  
##  Mean   :0.006061   Mean   :0.01939   Mean   :0.006061   Mean   :0.01455  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :3.000000   Max.   :1.00000  
##                                                                           
##      price             print              pro              problem        
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.00000   Median :0.000000   Median :0.000000  
##  Mean   :0.01455   Mean   :0.01212   Mean   :0.003636   Mean   :0.007273  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :2.00000   Max.   :1.00000   Max.   :1.000000   Max.   :1.000000  
##                                                                           
##     product             promo         promoipodplayerpromo
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000     
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000     
##  Median :0.000000   Median :0.00000   Median :0.00000     
##  Mean   :0.009697   Mean   :0.02061   Mean   :0.03758     
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000     
##  Max.   :1.000000   Max.   :1.00000   Max.   :1.00000     
##                                                           
##       put                que              quiet         
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.008485   Mean   :0.01212   Mean   :0.004849  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :2.00000   Max.   :1.000000  
##                                                         
##       read              realli          recommend       
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.002424   Mean   :0.02303   Mean   :0.004849  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :2.00000   Max.   :1.000000  
##                                                         
##      refus              releas            right         
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.007273   Mean   :0.01576   Mean   :0.009697  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :1.000000  
##                                                         
##       said             samsung          samsungsa             say         
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000   Median :0.000000   Median :0.00000  
##  Mean   :0.007273   Mean   :0.02061   Mean   :0.007273   Mean   :0.01576  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :1.000000   Max.   :1.00000  
##                                                                           
##     scanner            screen            secur               see         
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.00000   Median :0.000000   Median :0.00000  
##  Mean   :0.01091   Mean   :0.01333   Mean   :0.007273   Mean   :0.01091  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :1.00000   Max.   :1.000000   Max.   :1.00000  
##                                                                          
##       seem               sell               send         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.007273   Mean   :0.009697   Mean   :0.004849  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##      servic            shame              share         
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.000000   Median :0.000000  
##  Mean   :0.01212   Mean   :0.006061   Mean   :0.008485  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :2.000000  
##                                                         
##      short               show              simpl         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.006061   Mean   :0.007273   Mean   :0.004849  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##       sinc               siri              smart         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.009697   Mean   :0.008485   Mean   :0.003636  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##    smartphon            someth              soon         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.007273   Mean   :0.006061   Mean   :0.004849  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##      stand              start              steve         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.007273   Mean   :0.008485   Mean   :0.009697  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##      still              stop             store             stuff        
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.00000   Median :0.00000   Median :0.00000  
##  Mean   :0.01697   Mean   :0.01333   Mean   :0.03636   Mean   :0.01212  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :1.00000   Max.   :2.00000   Max.   :1.00000  
##                                                                         
##      stupid              suck             support             sure        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0.00000   Median :0.00000  
##  Mean   :0.006061   Mean   :0.004849   Mean   :0.01091   Mean   :0.01576  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :2.000000   Max.   :1.000000   Max.   :2.00000   Max.   :1.00000  
##                                                                           
##      switch              take              talk         
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.009697   Mean   :0.01212   Mean   :0.006061  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :1.000000  
##                                                         
##       team               tech           technolog             tell        
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000   Median :0.000000   Median :0.00000  
##  Mean   :0.004849   Mean   :0.01212   Mean   :0.004849   Mean   :0.01212  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :1.000000   Max.   :1.00000  
##                                                                           
##       text             thank              that             theyr         
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.00000   Median :0.00000   Median :0.000000  
##  Mean   :0.01091   Mean   :0.03394   Mean   :0.01212   Mean   :0.004849  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :2.00000   Max.   :2.00000   Max.   :1.00000   Max.   :1.000000  
##                                                                          
##      thing             think              tho              thought        
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.00000   Median :0.000000   Median :0.000000  
##  Mean   :0.01333   Mean   :0.02667   Mean   :0.004849   Mean   :0.007273  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :2.00000   Max.   :1.000000   Max.   :1.000000  
##                                                                           
##       time             today             togeth             touch         
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.00000   Median :0.000000   Median :0.000000  
##  Mean   :0.03273   Mean   :0.01333   Mean   :0.007273   Mean   :0.006061  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :2.00000   Max.   :1.00000   Max.   :1.000000   Max.   :1.000000  
##                                                                           
##     touchid              tri               true         
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.008485   Mean   :0.01333   Mean   :0.004849  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :2.00000   Max.   :1.000000  
##                                                         
##       turn             twitter             two               updat        
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000   Median :0.000000   Median :0.00000  
##  Mean   :0.008485   Mean   :0.02424   Mean   :0.007273   Mean   :0.02182  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :2.000000   Max.   :1.00000   Max.   :1.000000   Max.   :1.00000  
##                                                                           
##      upgrad              use               user               via         
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000   Median :0.000000   Median :0.00000  
##  Mean   :0.008485   Mean   :0.03758   Mean   :0.004849   Mean   :0.01212  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :1.000000   Max.   :1.00000  
##                                                                           
##      video               wait              want             watch         
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.000000  
##  Mean   :0.009697   Mean   :0.01091   Mean   :0.02061   Mean   :0.007273  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :2.000000   Max.   :1.00000   Max.   :1.00000   Max.   :1.000000  
##                                                                           
##       way               week               well              what         
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.01455   Mean   :0.009697   Mean   :0.02061   Mean   :0.007273  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :2.00000   Max.   :2.000000   Max.   :2.00000   Max.   :1.000000  
##                                                                           
##      white               will          windowsphon      
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.008485   Mean   :0.04848   Mean   :0.007273  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :2.000000   Max.   :2.00000   Max.   :1.000000  
##                                                         
##       wish             without             wonder        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.009697   Mean   :0.002424   Mean   :0.006061  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##       wont              work             world              worst         
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.00000   Median :0.000000   Median :0.000000  
##  Mean   :0.01576   Mean   :0.02182   Mean   :0.007273   Mean   :0.006061  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :2.00000   Max.   :1.00000   Max.   :1.000000   Max.   :1.000000  
##                                                                           
##       wow                wtf                yall              year        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0.00000   Median :0.00000  
##  Mean   :0.004849   Mean   :0.009697   Mean   :0.01091   Mean   :0.01091  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.00000   Max.   :1.00000  
##                                                                           
##       yes                yet                yooo         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.003636   Mean   :0.008485   Mean   :0.004849  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##       your         
##  Min.   :0.000000  
##  1st Qu.:0.000000  
##  Median :0.000000  
##  Mean   :0.009697  
##  3rd Qu.:0.000000  
##  Max.   :2.000000  
##                    
##                Tweet                  Avg           Tweet.fctr 
##                    0                    0                    0 
##               .rnorm        Negative.fctr       X244tsuyoponzu 
##                    0                    0                    0 
##          X7evenstarz               actual                  add 
##                    0                    0                    0 
##              alreadi                alway                 amaz 
##                    0                    0                    0 
##               amazon              android              announc 
##                    0                    0                    0 
##                anyon                  app                 appl 
##                    0                    0                    0 
##              appstor                arent                  ask 
##                    0                    0                    0 
##                avail                 away               awesom 
##                    0                    0                    0 
##                 back              batteri                 best 
##                    0                    0                    0 
##               better                  big                  bit 
##                    0                    0                    0 
##                black           blackberri               break. 
##                    0                    0                    0 
##                bring             burberri                 busi 
##                    0                    0                    0 
##                  buy                 call                  can 
##                    0                    0                    0 
##                 cant               carbon                 card 
##                    0                    0                    0 
##                 care                 case                  cdp 
##                    0                    0                    0 
##                chang                charg              charger 
##                    0                    0                    0 
##                cheap                china                color 
##                    0                    0                    0 
##               colour                 come              compani 
##                    0                    0                    0 
##           condescens               condom                 copi 
##                    0                    0                    0 
##                crack                creat               custom 
##                    0                    0                    0 
##                 darn                 data                 date 
##                    0                    0                    0 
##                  day                 dear               design 
##                    0                    0                    0 
##              develop                devic                didnt 
##                    0                    0                    0 
##                  die               differ           disappoint 
##                    0                    0                    0 
##           discontinu               divulg               doesnt 
##                    0                    0                    0 
##                 done                 dont             download 
##                    0                    0                    0 
##                 drop                email                emiss 
##                    0                    0                    0 
##                emoji                 even                event 
##                    0                    0                    0 
##                 ever                everi              everyth 
##                    0                    0                    0 
##             facebook                 fail               featur 
##                    0                    0                    0 
##                 feel                femal                figur 
##                    0                    0                    0 
##                final               finger          fingerprint 
##                    0                    0                    0 
##                 fire                first                  fix 
##                    0                    0                    0 
##               follow                freak                 free 
##                    0                    0                    0 
##                  fun              generat               genius 
##                    0                    0                    0 
##                  get                 give                 gold 
##                    0                    0                    0 
##                gonna                 good                googl 
##                    0                    0                    0 
##                  got                great                guess 
##                    0                    0                    0 
##                  guy               happen                happi 
##                    0                    0                    0 
##                 hate                 help                  hey 
##                    0                    0                    0 
##                 hope                 hour     httpbitly18xc8dk 
##                    0                    0                    0 
##           ibrooklynb                 idea                  ill 
##                    0                    0                    0 
##              imessag              impress               improv 
##                    0                    0                    0 
##                innov              instead             internet 
##                    0                    0                    0 
##                 ios7                 ipad                iphon 
##                    0                    0                    0 
##              iphone4              iphone5             iphone5c 
##                    0                    0                    0 
##               iphoto                 ipod      ipodplayerpromo 
##                    0                    0                    0 
##                 isnt                 itun                  ive 
##                    0                    0                    0 
##                  job                 just               keynot 
##                    0                    0                    0 
##                 know                 last               launch 
##                    0                    0                    0 
##                  let                 life                 like 
##                    0                    0                    0 
##                 line                 lmao                 lock 
##                    0                    0                    0 
##                  lol                 look                  los 
##                    0                    0                    0 
##                 lost                 love                  mac 
##                    0                    0                    0 
##              macbook                 made                 make 
##                    0                    0                    0 
##                  man                 mani               market 
##                    0                    0                    0 
##                 mayb                 mean            microsoft 
##                    0                    0                    0 
##              mishiza                 miss                mobil 
##                    0                    0                    0 
##                money             motorola                 move 
##                    0                    0                    0 
##                 much                music             natz0711 
##                    0                    0                    0 
##                 need                never                  new 
##                    0                    0                    0 
##                 news                next.                  nfc 
##                    0                    0                    0 
##                nokia                 noth                  now 
##                    0                    0                    0 
##                  nsa                nuevo                offer 
##                    0                    0                    0 
##                  old                  one                 page 
##                    0                    0                    0 
##                 para                peopl              perfect 
##                    0                    0                    0 
##               person                phone               photog 
##                    0                    0                    0 
##          photographi               pictur              plastic 
##                    0                    0                    0 
##                 play                pleas                  ppl 
##                    0                    0                    0 
##             preorder                price                print 
##                    0                    0                    0 
##                  pro              problem              product 
##                    0                    0                    0 
##                promo promoipodplayerpromo                  put 
##                    0                    0                    0 
##                  que                quiet                 read 
##                    0                    0                    0 
##               realli            recommend                refus 
##                    0                    0                    0 
##               releas                right                 said 
##                    0                    0                    0 
##              samsung            samsungsa                  say 
##                    0                    0                    0 
##              scanner               screen                secur 
##                    0                    0                    0 
##                  see                 seem                 sell 
##                    0                    0                    0 
##                 send               servic                shame 
##                    0                    0                    0 
##                share                short                 show 
##                    0                    0                    0 
##                simpl                 sinc                 siri 
##                    0                    0                    0 
##                smart            smartphon               someth 
##                    0                    0                    0 
##                 soon                stand                start 
##                    0                    0                    0 
##                steve                still                 stop 
##                    0                    0                    0 
##                store                stuff               stupid 
##                    0                    0                    0 
##                 suck              support                 sure 
##                    0                    0                    0 
##               switch                 take                 talk 
##                    0                    0                    0 
##                 team                 tech            technolog 
##                    0                    0                    0 
##                 tell                 text                thank 
##                    0                    0                    0 
##                 that                theyr                thing 
##                    0                    0                    0 
##                think                  tho              thought 
##                    0                    0                    0 
##                 time                today               togeth 
##                    0                    0                    0 
##                touch              touchid                  tri 
##                    0                    0                    0 
##                 true                 turn              twitter 
##                    0                    0                    0 
##                  two                updat               upgrad 
##                    0                    0                    0 
##                  use                 user                  via 
##                    0                    0                    0 
##                video                 wait                 want 
##                    0                    0                    0 
##                watch                  way                 week 
##                    0                    0                    0 
##                 well                 what                white 
##                    0                    0                    0 
##                 will          windowsphon                 wish 
##                    0                    0                    0 
##              without               wonder                 wont 
##                    0                    0                    0 
##                 work                world                worst 
##                    0                    0                    0 
##                  wow                  wtf                 yall 
##                    0                    0                    0 
##                 year                  yes                  yet 
##                    0                    0                    0 
##                 yooo                 your 
##                    0                    0 
##     Tweet                Avg         
##  Length:356         Min.   :-2.0000  
##  Class :character   1st Qu.:-0.6000  
##  Mode  :character   Median : 0.0000  
##                     Mean   :-0.1983  
##                     3rd Qu.: 0.2000  
##                     Max.   : 2.0000  
##                                      
##                                                                                                                                  Tweet.fctr 
##  C #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO @apple @itune                                                   :  4  
##  FOLLOW @APPLE NOW                                                                                                                    :  3  
##  A #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO @apple @itune                                                   :  2  
##  {#IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO #IPODPLAYERPROMO #IPOD #IPAD #ITUNES #APPLE #PROMO#IPODPLAYERPROMO @apple @itune}:  2  
##  @battalalgoos @apple                                                                                                                 :  2  
##  iOS 7 is so fricking smooth & beautiful!! #ThanxApple @Apple                                                                         :  1  
##  (Other)                                                                                                                              :342  
##      .rnorm         Negative.fctr X244tsuyoponzu      X7evenstarz      
##  Min.   :-3.04697   N:300         Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:-0.80307   Y: 56         1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :-0.08948                 Median :0.000000   Median :0.000000  
##  Mean   :-0.07853                 Mean   :0.002809   Mean   :0.008427  
##  3rd Qu.: 0.61123                 3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   : 2.27908                 Max.   :1.000000   Max.   :1.000000  
##                                                                        
##      actual              add              alreadi            alway        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0.00000   Median :0.00000  
##  Mean   :0.005618   Mean   :0.008427   Mean   :0.02247   Mean   :0.01124  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.00000   Max.   :1.00000  
##                                                                           
##       amaz              amazon            android        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.005618   Mean   :0.005618   Mean   :0.008427  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##     announc            anyon              app               appl        
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.00000   Median :0.00000   Median :0.00000  
##  Mean   :0.01404   Mean   :0.01404   Mean   :0.03652   Mean   :0.01966  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :1.00000   Max.   :2.00000   Max.   :1.00000  
##                                                                         
##     appstor             arent               ask          
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.005618   Mean   :0.005618   Mean   :0.008427  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##      avail               away              awesom        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.005618   Mean   :0.002809   Mean   :0.005618  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##       back            batteri             best             better       
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.00000   Median :0.00000   Median :0.00000  
##  Mean   :0.01124   Mean   :0.02528   Mean   :0.01124   Mean   :0.01404  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :2.00000   Max.   :2.00000   Max.   :1.00000   Max.   :1.00000  
##                                                                         
##       big                bit               black         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.008427   Mean   :0.008427   Mean   :0.005618  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :2.000000   Max.   :1.000000  
##                                                          
##    blackberri           break.      bring      burberri       
##  Min.   :0.000000   Min.   :0   Min.   :0   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0   1st Qu.:0   1st Qu.:0.000000  
##  Median :0.000000   Median :0   Median :0   Median :0.000000  
##  Mean   :0.002809   Mean   :0   Mean   :0   Mean   :0.002809  
##  3rd Qu.:0.000000   3rd Qu.:0   3rd Qu.:0   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :0   Max.   :0   Max.   :1.000000  
##                                                               
##       busi               buy               call              can         
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.00000  
##  Mean   :0.002809   Mean   :0.01404   Mean   :0.01124   Mean   :0.04494  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :2.00000   Max.   :2.00000  
##                                                                          
##       cant             carbon              card              care         
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.01685   Mean   :0.002809   Mean   :0.01124   Mean   :0.008427  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :1.00000   Max.   :1.000000  
##                                                                           
##       case               cdp               chang         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.005618   Mean   :0.002809   Mean   :0.005618  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##      charg             charger            cheap             china         
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.000000  
##  Mean   :0.008427   Mean   :0.01124   Mean   :0.01404   Mean   :0.002809  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :1.00000   Max.   :1.000000  
##                                                                           
##      color             colour              come            compani       
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.000000   Median :0.00000   Median :0.00000  
##  Mean   :0.01685   Mean   :0.008427   Mean   :0.03933   Mean   :0.01124  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :2.00000   Max.   :1.000000   Max.   :1.00000   Max.   :1.00000  
##                                                                          
##    condescens           condom              copi        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0.00000  
##  Mean   :0.008427   Mean   :0.005618   Mean   :0.01124  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.00000  
##                                                         
##      crack              creat              custom             darn  
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000   Min.   :0  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0  
##  Median :0.000000   Median :0.000000   Median :0.00000   Median :0  
##  Mean   :0.005618   Mean   :0.008427   Mean   :0.01404   Mean   :0  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.00000   Max.   :0  
##                                                                     
##       data               date               day               dear        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0.00000   Median :0.00000  
##  Mean   :0.002809   Mean   :0.002809   Mean   :0.01685   Mean   :0.01124  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :2.00000   Max.   :1.00000  
##                                                                           
##      design           develop      devic             didnt        
##  Min.   :0.00000   Min.   :0   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.00000   Median :0   Median :0.00000   Median :0.00000  
##  Mean   :0.01124   Mean   :0   Mean   :0.01124   Mean   :0.01404  
##  3rd Qu.:0.00000   3rd Qu.:0   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :0   Max.   :1.00000   Max.   :1.00000  
##                                                                   
##       die              differ          disappoint         discontinu     
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.00000   Median :0.000000   Median :0.00000  
##  Mean   :0.01124   Mean   :0.01124   Mean   :0.008427   Mean   :0.01124  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :1.00000   Max.   :1.000000   Max.   :1.00000  
##                                                                          
##      divulg             doesnt             done              dont        
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.00000  
##  Mean   :0.005618   Mean   :0.01685   Mean   :0.01404   Mean   :0.04775  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :1.00000   Max.   :1.00000  
##                                                                          
##     download             drop              email         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.005618   Mean   :0.005618   Mean   :0.005618  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##      emiss              emoji              even             event  
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0  
##  Mean   :0.002809   Mean   :0.01404   Mean   :0.02528   Mean   :0  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0  
##  Max.   :1.000000   Max.   :2.00000   Max.   :1.00000   Max.   :0  
##                                                                    
##       ever              everi             everyth       
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0.00000  
##  Mean   :0.005618   Mean   :0.008427   Mean   :0.01124  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :2.000000   Max.   :1.00000  
##                                                         
##     facebook             fail              featur       
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0.00000  
##  Mean   :0.005618   Mean   :0.005618   Mean   :0.01685  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :2.00000  
##                                                         
##       feel              femal              figur         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.002809   Mean   :0.005618   Mean   :0.005618  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##      final             finger          fingerprint           fire  
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.00000   Min.   :0  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0  
##  Median :0.00000   Median :0.000000   Median :0.00000   Median :0  
##  Mean   :0.01124   Mean   :0.002809   Mean   :0.01685   Mean   :0  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0  
##  Max.   :1.00000   Max.   :1.000000   Max.   :1.00000   Max.   :0  
##                                                                    
##      first        fix              follow            freak        
##  Min.   :0   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0   Median :0.00000   Median :0.00000   Median :0.00000  
##  Mean   :0   Mean   :0.02247   Mean   :0.01124   Mean   :0.05337  
##  3rd Qu.:0   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :0   Max.   :1.00000   Max.   :1.00000   Max.   :2.00000  
##                                                                   
##       free              fun              generat             genius       
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.000000   Median :0.000000   Median :0.00000  
##  Mean   :0.01124   Mean   :0.002809   Mean   :0.005618   Mean   :0.01124  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :1.000000   Max.   :1.00000  
##                                                                           
##       get               give              gold              gonna        
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.00000   Median :0.000000   Median :0.00000  
##  Mean   :0.05618   Mean   :0.01404   Mean   :0.005618   Mean   :0.01124  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :2.00000   Max.   :1.00000   Max.   :1.000000   Max.   :1.00000  
##                                                                          
##       good             googl              got              great        
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.00000   Median :0.00000   Median :0.00000  
##  Mean   :0.01404   Mean   :0.02247   Mean   :0.01685   Mean   :0.01404  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :1.00000   Max.   :1.00000   Max.   :1.00000  
##                                                                         
##      guess              guy              happen            happi         
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.00000   Median :0.00000   Median :0.000000  
##  Mean   :0.01404   Mean   :0.01966   Mean   :0.01404   Mean   :0.002809  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.00000   Max.   :1.00000   Max.   :1.000000  
##                                                                          
##       hate              help              hey               hope         
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.00000   Median :0.00000   Median :0.000000  
##  Mean   :0.02247   Mean   :0.02247   Mean   :0.01404   Mean   :0.008427  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.00000   Max.   :1.00000   Max.   :1.000000  
##                                                                          
##       hour          httpbitly18xc8dk     ibrooklynb      
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.005618   Mean   :0.005618   Mean   :0.008427  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##       idea              ill              imessag        
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.000000   Median :0.000000  
##  Mean   :0.01404   Mean   :0.005618   Mean   :0.002809  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :1.000000  
##                                                         
##     impress             improv      innov            instead        
##  Min.   :0.000000   Min.   :0   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0   Median :0.00000   Median :0.000000  
##  Mean   :0.002809   Mean   :0   Mean   :0.01685   Mean   :0.008427  
##  3rd Qu.:0.000000   3rd Qu.:0   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :0   Max.   :1.00000   Max.   :1.000000  
##                                                                     
##     internet             ios7              ipad             iphon       
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.0000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.0000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.0000  
##  Mean   :0.005618   Mean   :0.03652   Mean   :0.05618   Mean   :0.2331  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.0000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :2.00000   Max.   :3.0000  
##                                                                         
##     iphone4            iphone5          iphone5c           iphoto        
##  Min.   :0.000000   Min.   :0.0000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.0000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.0000   Median :0.00000   Median :0.000000  
##  Mean   :0.005618   Mean   :0.0309   Mean   :0.03652   Mean   :0.008427  
##  3rd Qu.:0.000000   3rd Qu.:0.0000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.0000   Max.   :1.00000   Max.   :1.000000  
##                                                                          
##       ipod         ipodplayerpromo        isnt              itun        
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.00000   Median :0.00000   Median :0.00000  
##  Mean   :0.04775   Mean   :0.03933   Mean   :0.01124   Mean   :0.08989  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :2.00000   Max.   :2.00000   Max.   :1.00000   Max.   :3.00000  
##                                                                         
##       ive                job               just             keynot        
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.000000  
##  Mean   :0.002809   Mean   :0.01124   Mean   :0.05056   Mean   :0.005618  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :2.00000   Max.   :1.000000  
##                                                                           
##       know              last              launch              let         
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.000000   Median :0.000000   Median :0.00000  
##  Mean   :0.01966   Mean   :0.008427   Mean   :0.002809   Mean   :0.01685  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :1.000000   Max.   :1.00000  
##                                                                           
##       life               like             line               lmao         
##  Min.   :0.000000   Min.   :0.0000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.0000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.0000   Median :0.000000   Median :0.000000  
##  Mean   :0.005618   Mean   :0.0309   Mean   :0.005618   Mean   :0.005618  
##  3rd Qu.:0.000000   3rd Qu.:0.0000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :2.0000   Max.   :1.000000   Max.   :1.000000  
##                                                                           
##       lock               lol               look              los          
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.000000  
##  Mean   :0.002809   Mean   :0.02247   Mean   :0.01685   Mean   :0.008427  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :2.00000   Max.   :1.00000   Max.   :1.000000  
##                                                                           
##       lost               love              mac              macbook       
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000   Median :0.000000   Median :0.00000  
##  Mean   :0.002809   Mean   :0.02809   Mean   :0.005618   Mean   :0.01124  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :1.000000   Max.   :1.00000  
##                                                                           
##       made               make              man               mani         
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.000000  
##  Mean   :0.008427   Mean   :0.04494   Mean   :0.01124   Mean   :0.008427  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :2.00000   Max.   :1.00000   Max.   :1.000000  
##                                                                           
##      market             mayb        mean     microsoft      
##  Min.   :0.00000   Min.   :0   Min.   :0   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0   1st Qu.:0   1st Qu.:0.00000  
##  Median :0.00000   Median :0   Median :0   Median :0.00000  
##  Mean   :0.01685   Mean   :0   Mean   :0   Mean   :0.01404  
##  3rd Qu.:0.00000   3rd Qu.:0   3rd Qu.:0   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :0   Max.   :0   Max.   :1.00000  
##                                                             
##     mishiza              miss              mobil             money        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0.00000   Median :0.00000  
##  Mean   :0.005618   Mean   :0.005618   Mean   :0.01966   Mean   :0.01124  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.00000   Max.   :1.00000  
##                                                                           
##     motorola             move               much             music        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0.00000   Median :0.00000  
##  Mean   :0.005618   Mean   :0.005618   Mean   :0.01124   Mean   :0.01124  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.00000   Max.   :1.00000  
##                                                                           
##     natz0711             need             never               new         
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000   Median :0.000000   Median :0.00000  
##  Mean   :0.002809   Mean   :0.03933   Mean   :0.005618   Mean   :0.09831  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :1.000000   Max.   :2.00000  
##                                                                           
##       news              next.               nfc          
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.008427   Mean   :0.008427   Mean   :0.008427  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##      nokia              noth              now               nsa          
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.00000   Median :0.00000   Median :0.000000  
##  Mean   :0.01124   Mean   :0.01124   Mean   :0.03371   Mean   :0.005618  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.00000   Max.   :1.00000   Max.   :1.000000  
##                                                                          
##      nuevo              offer               old          
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.005618   Mean   :0.005618   Mean   :0.002809  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :2.000000   Max.   :1.000000  
##                                                          
##       one               page               para       peopl         
##  Min.   :0.00000   Min.   :0.000000   Min.   :0   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0   1st Qu.:0.000000  
##  Median :0.00000   Median :0.000000   Median :0   Median :0.000000  
##  Mean   :0.02247   Mean   :0.005618   Mean   :0   Mean   :0.008427  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0   3rd Qu.:0.000000  
##  Max.   :2.00000   Max.   :1.000000   Max.   :0   Max.   :1.000000  
##                                                                     
##     perfect             person             phone        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0.00000  
##  Mean   :0.005618   Mean   :0.008427   Mean   :0.07865  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :2.00000  
##                                                         
##      photog          photographi           pictur        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.008427   Mean   :0.008427   Mean   :0.005618  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##     plastic              play              pleas        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0.00000  
##  Mean   :0.005618   Mean   :0.002809   Mean   :0.01685  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.00000  
##                                                         
##       ppl              preorder           price             print        
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.00000  
##  Mean   :0.008427   Mean   :0.01685   Mean   :0.01124   Mean   :0.01124  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :1.00000   Max.   :1.00000  
##                                                                          
##       pro             problem            product            promo         
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.01124   Mean   :0.002809   Mean   :0.02247   Mean   :0.008427  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :1.00000   Max.   :1.000000  
##                                                                           
##  promoipodplayerpromo      put                que          
##  Min.   :0.0000       Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.0000       1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.0000       Median :0.000000   Median :0.000000  
##  Mean   :0.0309       Mean   :0.005618   Mean   :0.005618  
##  3rd Qu.:0.0000       3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.0000       Max.   :1.000000   Max.   :1.000000  
##                                                            
##      quiet               read             realli          recommend       
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.000000  
##  Mean   :0.008427   Mean   :0.01124   Mean   :0.02809   Mean   :0.008427  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :2.00000   Max.   :1.000000  
##                                                                           
##      refus              releas            right              said         
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.000000  
##  Mean   :0.005618   Mean   :0.03371   Mean   :0.01404   Mean   :0.002809  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :1.00000   Max.   :1.000000  
##                                                                           
##     samsung          samsungsa             say             scanner        
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.01966   Mean   :0.005618   Mean   :0.02247   Mean   :0.002809  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :1.00000   Max.   :1.000000  
##                                                                           
##      screen             secur               see          
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.008427   Mean   :0.002809   Mean   :0.005618  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :2.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##       seem              sell               send             servic       
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.000000   Median :0.00000   Median :0.00000  
##  Mean   :0.01124   Mean   :0.005618   Mean   :0.01124   Mean   :0.01685  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :1.00000   Max.   :1.00000  
##                                                                          
##      shame              share              short         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.002809   Mean   :0.002809   Mean   :0.002809  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##       show              simpl               sinc        siri        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0   1st Qu.:0.00000  
##  Median :0.000000   Median :0.000000   Median :0   Median :0.00000  
##  Mean   :0.002809   Mean   :0.008427   Mean   :0   Mean   :0.01404  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :2.000000   Max.   :0   Max.   :1.00000  
##                                                                     
##      smart            smartphon            someth        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.008427   Mean   :0.002809   Mean   :0.008427  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##       soon              stand              start         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.005618   Mean   :0.005618   Mean   :0.002809  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##      steve             still              stop              store        
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.00000   Median :0.000000   Median :0.00000  
##  Mean   :0.01124   Mean   :0.01124   Mean   :0.008427   Mean   :0.02809  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :1.00000   Max.   :1.000000   Max.   :1.00000  
##                                                                          
##      stuff              stupid              suck         
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.008427   Mean   :0.005618   Mean   :0.008427  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##     support              sure              switch        
##  Min.   :0.000000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.000000   Median :0.000000  
##  Mean   :0.002809   Mean   :0.008427   Mean   :0.002809  
##  3rd Qu.:0.000000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.000000   Max.   :1.000000  
##                                                          
##       take              talk               team         
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.000000   Median :0.000000  
##  Mean   :0.01404   Mean   :0.002809   Mean   :0.005618  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :1.000000  
##                                                         
##       tech            technolog            tell         
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.000000  
##  Mean   :0.008427   Mean   :0.01124   Mean   :0.002809  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :1.000000  
##                                                         
##       text              thank              that             theyr         
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.000000  
##  Mean   :0.005618   Mean   :0.01404   Mean   :0.01966   Mean   :0.005618  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :1.00000   Max.   :1.000000  
##                                                                           
##      thing             think              tho              thought        
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.000000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.00000   Median :0.000000   Median :0.000000  
##  Mean   :0.01124   Mean   :0.01966   Mean   :0.008427   Mean   :0.005618  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.00000   Max.   :1.000000   Max.   :1.000000  
##                                                                           
##       time             today              togeth      touch         
##  Min.   :0.00000   Min.   :0.000000   Min.   :0   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0   1st Qu.:0.000000  
##  Median :0.00000   Median :0.000000   Median :0   Median :0.000000  
##  Mean   :0.01966   Mean   :0.005618   Mean   :0   Mean   :0.002809  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :0   Max.   :1.000000  
##                                                                     
##     touchid              tri               true               turn        
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000   Median :0.000000   Median :0.00000  
##  Mean   :0.002809   Mean   :0.01124   Mean   :0.005618   Mean   :0.01404  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :1.000000   Max.   :1.00000  
##                                                                           
##     twitter             two               updat             upgrad       
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.000000   Median :0.00000   Median :0.00000  
##  Mean   :0.01124   Mean   :0.002809   Mean   :0.01404   Mean   :0.01124  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :1.00000   Max.   :2.00000  
##                                                                          
##       use               user               via              video        
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.000000   Median :0.00000   Median :0.00000  
##  Mean   :0.02809   Mean   :0.005618   Mean   :0.02809   Mean   :0.01124  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.00000   Max.   :1.000000   Max.   :1.00000   Max.   :1.00000  
##                                                                          
##       wait              want             watch              way          
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.00000   Median :0.00000   Median :0.000000  
##  Mean   :0.02809   Mean   :0.03652   Mean   :0.01685   Mean   :0.005618  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :1.00000   Max.   :1.00000   Max.   :1.00000   Max.   :1.000000  
##                                                                          
##       week               well              what              white  
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.000000   Min.   :0  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0  
##  Median :0.000000   Median :0.00000   Median :0.000000   Median :0  
##  Mean   :0.005618   Mean   :0.01404   Mean   :0.002809   Mean   :0  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0  
##  Max.   :1.000000   Max.   :1.00000   Max.   :1.000000   Max.   :0  
##                                                                     
##       will          windowsphon            wish            without       
##  Min.   :0.00000   Min.   :0.000000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.00000   1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.00000   Median :0.000000   Median :0.00000   Median :0.00000  
##  Mean   :0.04213   Mean   :0.008427   Mean   :0.01124   Mean   :0.01124  
##  3rd Qu.:0.00000   3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :2.00000   Max.   :1.000000   Max.   :1.00000   Max.   :1.00000  
##                                                                          
##      wonder              wont              work             world  
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0  
##  Mean   :0.002809   Mean   :0.01124   Mean   :0.01966   Mean   :0  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0  
##  Max.   :1.000000   Max.   :1.00000   Max.   :2.00000   Max.   :0  
##                                                                    
##      worst               wow               wtf               yall        
##  Min.   :0.000000   Min.   :0.00000   Min.   :0.00000   Min.   :0.00000  
##  1st Qu.:0.000000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000  
##  Median :0.000000   Median :0.00000   Median :0.00000   Median :0.00000  
##  Mean   :0.008427   Mean   :0.01404   Mean   :0.01685   Mean   :0.01966  
##  3rd Qu.:0.000000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000  
##  Max.   :1.000000   Max.   :1.00000   Max.   :1.00000   Max.   :2.00000  
##                                                                          
##       year              yes               yet               yooo         
##  Min.   :0.00000   Min.   :0.00000   Min.   :0.00000   Min.   :0.000000  
##  1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.00000   1st Qu.:0.000000  
##  Median :0.00000   Median :0.00000   Median :0.00000   Median :0.000000  
##  Mean   :0.02809   Mean   :0.01124   Mean   :0.01685   Mean   :0.008427  
##  3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000   3rd Qu.:0.000000  
##  Max.   :2.00000   Max.   :1.00000   Max.   :1.00000   Max.   :1.000000  
##                                                                          
##       your        
##  Min.   :0.00000  
##  1st Qu.:0.00000  
##  Median :0.00000  
##  Mean   :0.01404  
##  3rd Qu.:0.00000  
##  Max.   :1.00000  
##                   
##                Tweet                  Avg           Tweet.fctr 
##                    0                    0                    0 
##               .rnorm        Negative.fctr       X244tsuyoponzu 
##                    0                    0                    0 
##          X7evenstarz               actual                  add 
##                    0                    0                    0 
##              alreadi                alway                 amaz 
##                    0                    0                    0 
##               amazon              android              announc 
##                    0                    0                    0 
##                anyon                  app                 appl 
##                    0                    0                    0 
##              appstor                arent                  ask 
##                    0                    0                    0 
##                avail                 away               awesom 
##                    0                    0                    0 
##                 back              batteri                 best 
##                    0                    0                    0 
##               better                  big                  bit 
##                    0                    0                    0 
##                black           blackberri               break. 
##                    0                    0                    0 
##                bring             burberri                 busi 
##                    0                    0                    0 
##                  buy                 call                  can 
##                    0                    0                    0 
##                 cant               carbon                 card 
##                    0                    0                    0 
##                 care                 case                  cdp 
##                    0                    0                    0 
##                chang                charg              charger 
##                    0                    0                    0 
##                cheap                china                color 
##                    0                    0                    0 
##               colour                 come              compani 
##                    0                    0                    0 
##           condescens               condom                 copi 
##                    0                    0                    0 
##                crack                creat               custom 
##                    0                    0                    0 
##                 darn                 data                 date 
##                    0                    0                    0 
##                  day                 dear               design 
##                    0                    0                    0 
##              develop                devic                didnt 
##                    0                    0                    0 
##                  die               differ           disappoint 
##                    0                    0                    0 
##           discontinu               divulg               doesnt 
##                    0                    0                    0 
##                 done                 dont             download 
##                    0                    0                    0 
##                 drop                email                emiss 
##                    0                    0                    0 
##                emoji                 even                event 
##                    0                    0                    0 
##                 ever                everi              everyth 
##                    0                    0                    0 
##             facebook                 fail               featur 
##                    0                    0                    0 
##                 feel                femal                figur 
##                    0                    0                    0 
##                final               finger          fingerprint 
##                    0                    0                    0 
##                 fire                first                  fix 
##                    0                    0                    0 
##               follow                freak                 free 
##                    0                    0                    0 
##                  fun              generat               genius 
##                    0                    0                    0 
##                  get                 give                 gold 
##                    0                    0                    0 
##                gonna                 good                googl 
##                    0                    0                    0 
##                  got                great                guess 
##                    0                    0                    0 
##                  guy               happen                happi 
##                    0                    0                    0 
##                 hate                 help                  hey 
##                    0                    0                    0 
##                 hope                 hour     httpbitly18xc8dk 
##                    0                    0                    0 
##           ibrooklynb                 idea                  ill 
##                    0                    0                    0 
##              imessag              impress               improv 
##                    0                    0                    0 
##                innov              instead             internet 
##                    0                    0                    0 
##                 ios7                 ipad                iphon 
##                    0                    0                    0 
##              iphone4              iphone5             iphone5c 
##                    0                    0                    0 
##               iphoto                 ipod      ipodplayerpromo 
##                    0                    0                    0 
##                 isnt                 itun                  ive 
##                    0                    0                    0 
##                  job                 just               keynot 
##                    0                    0                    0 
##                 know                 last               launch 
##                    0                    0                    0 
##                  let                 life                 like 
##                    0                    0                    0 
##                 line                 lmao                 lock 
##                    0                    0                    0 
##                  lol                 look                  los 
##                    0                    0                    0 
##                 lost                 love                  mac 
##                    0                    0                    0 
##              macbook                 made                 make 
##                    0                    0                    0 
##                  man                 mani               market 
##                    0                    0                    0 
##                 mayb                 mean            microsoft 
##                    0                    0                    0 
##              mishiza                 miss                mobil 
##                    0                    0                    0 
##                money             motorola                 move 
##                    0                    0                    0 
##                 much                music             natz0711 
##                    0                    0                    0 
##                 need                never                  new 
##                    0                    0                    0 
##                 news                next.                  nfc 
##                    0                    0                    0 
##                nokia                 noth                  now 
##                    0                    0                    0 
##                  nsa                nuevo                offer 
##                    0                    0                    0 
##                  old                  one                 page 
##                    0                    0                    0 
##                 para                peopl              perfect 
##                    0                    0                    0 
##               person                phone               photog 
##                    0                    0                    0 
##          photographi               pictur              plastic 
##                    0                    0                    0 
##                 play                pleas                  ppl 
##                    0                    0                    0 
##             preorder                price                print 
##                    0                    0                    0 
##                  pro              problem              product 
##                    0                    0                    0 
##                promo promoipodplayerpromo                  put 
##                    0                    0                    0 
##                  que                quiet                 read 
##                    0                    0                    0 
##               realli            recommend                refus 
##                    0                    0                    0 
##               releas                right                 said 
##                    0                    0                    0 
##              samsung            samsungsa                  say 
##                    0                    0                    0 
##              scanner               screen                secur 
##                    0                    0                    0 
##                  see                 seem                 sell 
##                    0                    0                    0 
##                 send               servic                shame 
##                    0                    0                    0 
##                share                short                 show 
##                    0                    0                    0 
##                simpl                 sinc                 siri 
##                    0                    0                    0 
##                smart            smartphon               someth 
##                    0                    0                    0 
##                 soon                stand                start 
##                    0                    0                    0 
##                steve                still                 stop 
##                    0                    0                    0 
##                store                stuff               stupid 
##                    0                    0                    0 
##                 suck              support                 sure 
##                    0                    0                    0 
##               switch                 take                 talk 
##                    0                    0                    0 
##                 team                 tech            technolog 
##                    0                    0                    0 
##                 tell                 text                thank 
##                    0                    0                    0 
##                 that                theyr                thing 
##                    0                    0                    0 
##                think                  tho              thought 
##                    0                    0                    0 
##                 time                today               togeth 
##                    0                    0                    0 
##                touch              touchid                  tri 
##                    0                    0                    0 
##                 true                 turn              twitter 
##                    0                    0                    0 
##                  two                updat               upgrad 
##                    0                    0                    0 
##                  use                 user                  via 
##                    0                    0                    0 
##                video                 wait                 want 
##                    0                    0                    0 
##                watch                  way                 week 
##                    0                    0                    0 
##                 well                 what                white 
##                    0                    0                    0 
##                 will          windowsphon                 wish 
##                    0                    0                    0 
##              without               wonder                 wont 
##                    0                    0                    0 
##                 work                world                worst 
##                    0                    0                    0 
##                  wow                  wtf                 yall 
##                    0                    0                    0 
##                 year                  yes                  yet 
##                    0                    0                    0 
##                 yooo                 your 
##                    0                    0
```

```r
# print(sapply(names(glb_trnent_df), function(col) sum(is.na(glb_trnent_df[, col]))))
# print(sapply(names(glb_newent_df), function(col) sum(is.na(glb_newent_df[, col]))))

# print(myplot_scatter(glb_trnent_df, "<col1_name>", "<col2_name>", smooth=TRUE))

replay.petrisim(pn=glb_analytics_pn, 
    replay.trans=(glb_analytics_avl_objs <- c(glb_analytics_avl_objs, 
        "data.training.all","data.new")), flip_coord=TRUE)
```

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0
```

![](Apple_Tweets_files/figure-html/extract_features-4.png) 

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="select_features", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##               chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed5 extract_features                3                0   6.366
## elapsed6  select_features                4                0  14.222
```

## Step `4`: select features

```r
print(glb_feats_df <- myselect_features(entity_df=glb_trnent_df, 
                       exclude_vars_as_features=glb_exclude_vars_as_features, 
                       rsp_var=glb_rsp_var))
```

```
##                                        id         cor.y exclude.as.feat
## Avg                                   Avg -0.7061583707               1
## Tweet.fctr                     Tweet.fctr  0.6354986940               1
## freak                               freak  0.4220126658               0
## hate                                 hate  0.2150466216               0
## stuff                               stuff  0.1685214122               0
## suck                                 suck  0.1644037549               0
## pictur                             pictur  0.1642996876               0
## wtf                                   wtf  0.1642996876               0
## cant                                 cant  0.1518891001               0
## shame                               shame  0.1405061949               0
## stupid                             stupid  0.1186461031               0
## even                                 even  0.1158940633               0
## line                                 line  0.1158940633               0
## yooo                                 yooo  0.1158940633               0
## better                             better  0.1114365401               0
## ever                                 ever  0.1076729878               0
## fix                                   fix  0.1076729878               0
## charger                           charger  0.1069355141               0
## still                               still  0.1007505594               0
## charg                               charg  0.0970913594               0
## disappoint                     disappoint  0.0970913594               0
## short                               short  0.0970913594               0
## like                                 like  0.0962377709               0
## ipod                                 ipod -0.0852874403               0
## amazon                             amazon  0.0851663937               0
## yall                                 yall  0.0851663937               0
## promoipodplayerpromo promoipodplayerpromo -0.0838913688               0
## break.                             break.  0.0826293697               0
## imessag                           imessag  0.0826293697               0
## stand                               stand  0.0826293697               0
## togeth                             togeth  0.0826293697               0
## ipodplayerpromo           ipodplayerpromo -0.0793460925               0
## cheap                               cheap  0.0769587757               0
## wont                                 wont  0.0758516866               0
## make                                 make  0.0727362480               0
## carbon                             carbon  0.0709359262               0
## darn                                 darn  0.0709359262               0
## httpbitly18xc8dk         httpbitly18xc8dk  0.0709359262               0
## dear                                 dear  0.0681568527               0
## facebook                         facebook  0.0681568527               0
## X7evenstarz                   X7evenstarz  0.0673843716               0
## condom                             condom  0.0673843716               0
## femal                               femal  0.0673843716               0
## money                               money  0.0673843716               0
## theyr                               theyr  0.0673843716               0
## ipad                                 ipad -0.0646730918               0
## iphone5c                         iphone5c -0.0645077026               0
## batteri                           batteri  0.0626301156               0
## china                               china  0.0625002078               0
## turn                                 turn  0.0625002078               0
## iphone5                           iphone5 -0.0621379328               0
## promo                               promo -0.0615836556               0
## samsung                           samsung -0.0615836556               0
## hope                                 hope  0.0611434911               0
## life                                 life  0.0611434911               0
## sinc                                 sinc  0.0611434911               0
## steve                               steve  0.0611434911               0
## switch                             switch  0.0611434911               0
## love                                 love -0.0577763105               0
## everi                               everi  0.0564127109               0
## announc                           announc -0.0557827956               0
## last                                 last  0.0549744869               0
## your                                 your  0.0546349290               0
## lol                                   lol -0.0537205371               0
## amaz                                 amaz  0.0536765239               0
## arent                               arent  0.0536765239               0
## date                                 date  0.0536765239               0
## divulg                             divulg  0.0536765239               0
## ill                                   ill  0.0536765239               0
## ive                                   ive  0.0536765239               0
## lost                                 lost  0.0536765239               0
## noth                                 noth  0.0536765239               0
## worst                               worst  0.0536765239               0
## new                                   new -0.0536221452               0
## phone                               phone  0.0536044876               0
## appl                                 appl -0.0533458207               0
## care                                 care  0.0527276703               0
## way                                   way  0.0527276703               0
## year                                 year  0.0527276703               0
## updat                               updat  0.0519176496               0
## app                                   app  0.0506802204               0
## well                                 well -0.0506130885               0
## iphon                               iphon  0.0473646206               0
## help                                 help -0.0470292313               0
## print                               print -0.0470292313               0
## servic                             servic -0.0470292313               0
## tech                                 tech -0.0470292313               0
## tell                                 tell -0.0470292313               0
## will                                 will -0.0463979478               0
## news                                 news -0.0453528423               0
## data                                 data  0.0453496159               0
## take                                 take  0.0453496159               0
## microsoft                       microsoft -0.0450714475               0
## itun                                 itun -0.0449613971               0
## finger                             finger -0.0445884997               0
## instead                           instead -0.0445884997               0
## wait                                 wait -0.0445884997               0
## card                                 card  0.0429730405               0
## custom                             custom  0.0429730405               0
## die                                   die  0.0429730405               0
## event                               event  0.0429730405               0
## problem                           problem  0.0429730405               0
## refus                               refus  0.0429730405               0
## two                                   two  0.0429730405               0
## gold                                 gold -0.0428877879               0
## mobil                               mobil -0.0428877879               0
## que                                   que -0.0428877879               0
## avail                               avail -0.0420127055               0
## busi                                 busi -0.0420127055               0
## follow                             follow -0.0420127055               0
## generat                           generat -0.0420127055               0
## wish                                 wish -0.0420127055               0
## back                                 back -0.0405914861               0
## googl                               googl -0.0405914861               0
## use                                   use  0.0401415689               0
## big                                   big -0.0392752586               0
## emoji                               emoji -0.0392752586               0
## iphoto                             iphoto -0.0392752586               0
## motorola                         motorola -0.0392752586               0
## touchid                           touchid -0.0392752586               0
## screen                             screen  0.0387788990               0
## thing                               thing  0.0387788990               0
## buy                                   buy  0.0380283000               0
## burberri                         burberri -0.0375405647               0
## develop                           develop -0.0375405647               0
## video                               video -0.0375405647               0
## week                                 week -0.0375405647               0
## fire                                 fire -0.0363396181               0
## great                               great -0.0363396181               0
## keynot                             keynot -0.0363396181               0
## mayb                                 mayb -0.0363396181               0
## mishiza                           mishiza -0.0363396181               0
## natz0711                         natz0711 -0.0363396181               0
## offer                               offer -0.0363396181               0
## para                                 para -0.0363396181               0
## samsungsa                       samsungsa -0.0363396181               0
## show                                 show -0.0363396181               0
## watch                               watch -0.0363396181               0
## world                               world -0.0363396181               0
## need                                 need -0.0357398233               0
## tri                                   tri  0.0356343761               0
## actual                             actual -0.0346046348               0
## bring                               bring -0.0346046348               0
## share                               share -0.0346046348               0
## white                               white -0.0346046348               0
## cdp                                   cdp  0.0341988646               0
## doesnt                             doesnt  0.0341988646               0
## emiss                               emiss  0.0341988646               0
## feel                                 feel  0.0341988646               0
## fun                                   fun  0.0341988646               0
## got                                   got  0.0341988646               0
## hour                                 hour  0.0341988646               0
## macbook                           macbook  0.0341988646               0
## miss                                 miss  0.0341988646               0
## siri                                 siri  0.0341988646               0
## start                               start  0.0341988646               0
## upgrad                             upgrad  0.0341988646               0
## get                                   get  0.0332332539               0
## X244tsuyoponzu             X244tsuyoponzu -0.0331531471               0
## appstor                           appstor -0.0331531471               0
## away                                 away -0.0331531471               0
## creat                               creat -0.0331531471               0
## design                             design -0.0331531471               0
## happi                               happi -0.0331531471               0
## ibrooklynb                     ibrooklynb -0.0331531471               0
## launch                             launch -0.0331531471               0
## lock                                 lock -0.0331531471               0
## page                                 page -0.0331531471               0
## play                                 play -0.0331531471               0
## someth                             someth -0.0331531471               0
## talk                                 talk -0.0331531471               0
## touch                               touch -0.0331531471               0
## wonder                             wonder -0.0331531471               0
## devic                               devic -0.0325564924               0
## think                               think  0.0314859024               0
## anyon                               anyon -0.0305482407               0
## copi                                 copi  0.0303310278               0
## guess                               guess  0.0303310278               0
## person                             person  0.0303310278               0
## smart                               smart  0.0303310278               0
## alway                               alway -0.0296350116               0
## condescens                     condescens -0.0296350116               0
## crack                               crack -0.0296350116               0
## discontinu                     discontinu -0.0296350116               0
## email                               email -0.0296350116               0
## figur                               figur -0.0296350116               0
## internet                         internet -0.0296350116               0
## iphone4                           iphone4 -0.0296350116               0
## lmao                                 lmao -0.0296350116               0
## los                                   los -0.0296350116               0
## move                                 move -0.0296350116               0
## nuevo                               nuevo -0.0296350116               0
## perfect                           perfect -0.0296350116               0
## photog                             photog -0.0296350116               0
## photographi                   photographi -0.0296350116               0
## quiet                               quiet -0.0296350116               0
## recommend                       recommend -0.0296350116               0
## send                                 send -0.0296350116               0
## tho                                   tho -0.0296350116               0
## true                                 true -0.0296350116               0
## user                                 user -0.0296350116               0
## best                                 best  0.0267580922               0
## differ                             differ  0.0267580922               0
## nsa                                   nsa  0.0267580922               0
## product                           product  0.0267580922               0
## ios7                                 ios7 -0.0266634900               0
## peopl                               peopl -0.0266634900               0
## releas                             releas -0.0266634900               0
## say                                   say -0.0266634900               0
## store                               store -0.0259113719               0
## add                                   add -0.0256490570               0
## alreadi                           alreadi -0.0256490570               0
## final                               final -0.0256490570               0
## genius                             genius -0.0256490570               0
## nfc                                   nfc -0.0256490570               0
## pro                                   pro -0.0256490570               0
## yes                                   yes -0.0256490570               0
## just                                 just -0.0237594870               0
## ppl                                   ppl -0.0223147692               0
## price                               price -0.0216753537               0
## everyth                           everyth -0.0209296403               0
## music                               music -0.0209296403               0
## read                                 read -0.0209296403               0
## without                           without -0.0209296403               0
## twitter                           twitter  0.0207137058               0
## old                                   old  0.0202889470               0
## see                                   see  0.0202889470               0
## color                               color -0.0199770086               0
## compani                           compani -0.0199770086               0
## know                                 know -0.0199770086               0
## good                                 good -0.0199770086               0
## let                                   let -0.0199770086               0
## stop                                 stop -0.0199770086               0
## download                         download  0.0188746800               0
## idea                                 idea  0.0188746800               0
## job                                   job  0.0188746800               0
## mani                                 mani  0.0188746800               0
## simpl                               simpl  0.0188746800               0
## soon                                 soon  0.0188746800               0
## team                                 team  0.0188746800               0
## technolog                       technolog  0.0188746800               0
## wow                                   wow  0.0188746800               0
## much                                 much  0.0183336696               0
## support                           support  0.0183336696               0
## text                                 text  0.0183336696               0
## come                                 come -0.0172779254               0
## free                                 free -0.0162362822               0
## innov                               innov -0.0162362822               0
## via                                   via -0.0162362822               0
## time                                 time  0.0159958708               0
## android                           android  0.0149137629               0
## black                               black -0.0148064982               0
## that                                 that  0.0145566668               0
## want                                 want -0.0141452654               0
## fingerprint                   fingerprint  0.0129942941               0
## now                                   now  0.0129148701               0
## .rnorm                             .rnorm -0.0127552181               0
## scanner                           scanner -0.0121497764               0
## guy                                   guy -0.0109788835               0
## pleas                               pleas -0.0108398425               0
## ask                                   ask  0.0102616884               0
## colour                             colour  0.0102616884               0
## gonna                               gonna  0.0102616884               0
## happen                             happen  0.0102616884               0
## man                                   man  0.0102616884               0
## give                                 give  0.0094009452               0
## hey                                   hey  0.0094009452               0
## today                               today  0.0094009452               0
## case                                 case  0.0086651648               0
## chang                               chang  0.0086651648               0
## blackberri                     blackberri -0.0076273067               0
## mac                                   mac -0.0076273067               0
## never                               never -0.0076273067               0
## plastic                           plastic -0.0076273067               0
## right                               right -0.0076273067               0
## sell                                 sell -0.0076273067               0
## one                                   one  0.0070552161               0
## market                             market -0.0068841106               0
## work                                 work  0.0057872663               0
## thank                               thank -0.0049626450               0
## day                                   day  0.0047078144               0
## nokia                               nokia  0.0047078144               0
## preorder                         preorder  0.0047078144               0
## next.                               next. -0.0036050106               0
## look                                 look  0.0035739823               0
## awesom                             awesom  0.0033167112               0
## done                                 done  0.0033167112               0
## featur                             featur  0.0033167112               0
## said                                 said  0.0033167112               0
## secur                               secur  0.0033167112               0
## seem                                 seem  0.0033167112               0
## smartphon                       smartphon  0.0033167112               0
## thought                           thought  0.0033167112               0
## what                                 what  0.0033167112               0
## windowsphon                   windowsphon  0.0033167112               0
## bit                                   bit  0.0028697294               0
## call                                 call -0.0025381970               0
## didnt                               didnt -0.0025381970               0
## drop                                 drop -0.0025381970               0
## fail                                 fail -0.0025381970               0
## first                               first -0.0025381970               0
## improv                             improv -0.0025381970               0
## isnt                                 isnt -0.0025381970               0
## made                                 made -0.0025381970               0
## mean                                 mean -0.0025381970               0
## put                                   put -0.0025381970               0
## yet                                   yet -0.0025381970               0
## impress                           impress -0.0022363540               0
## realli                             realli  0.0020955452               0
## can                                   can  0.0017524556               0
## dont                                 dont -0.0006481129               0
## sure                                 sure  0.0003935570               0
##                         cor.y.abs
## Avg                  0.7061583707
## Tweet.fctr           0.6354986940
## freak                0.4220126658
## hate                 0.2150466216
## stuff                0.1685214122
## suck                 0.1644037549
## pictur               0.1642996876
## wtf                  0.1642996876
## cant                 0.1518891001
## shame                0.1405061949
## stupid               0.1186461031
## even                 0.1158940633
## line                 0.1158940633
## yooo                 0.1158940633
## better               0.1114365401
## ever                 0.1076729878
## fix                  0.1076729878
## charger              0.1069355141
## still                0.1007505594
## charg                0.0970913594
## disappoint           0.0970913594
## short                0.0970913594
## like                 0.0962377709
## ipod                 0.0852874403
## amazon               0.0851663937
## yall                 0.0851663937
## promoipodplayerpromo 0.0838913688
## break.               0.0826293697
## imessag              0.0826293697
## stand                0.0826293697
## togeth               0.0826293697
## ipodplayerpromo      0.0793460925
## cheap                0.0769587757
## wont                 0.0758516866
## make                 0.0727362480
## carbon               0.0709359262
## darn                 0.0709359262
## httpbitly18xc8dk     0.0709359262
## dear                 0.0681568527
## facebook             0.0681568527
## X7evenstarz          0.0673843716
## condom               0.0673843716
## femal                0.0673843716
## money                0.0673843716
## theyr                0.0673843716
## ipad                 0.0646730918
## iphone5c             0.0645077026
## batteri              0.0626301156
## china                0.0625002078
## turn                 0.0625002078
## iphone5              0.0621379328
## promo                0.0615836556
## samsung              0.0615836556
## hope                 0.0611434911
## life                 0.0611434911
## sinc                 0.0611434911
## steve                0.0611434911
## switch               0.0611434911
## love                 0.0577763105
## everi                0.0564127109
## announc              0.0557827956
## last                 0.0549744869
## your                 0.0546349290
## lol                  0.0537205371
## amaz                 0.0536765239
## arent                0.0536765239
## date                 0.0536765239
## divulg               0.0536765239
## ill                  0.0536765239
## ive                  0.0536765239
## lost                 0.0536765239
## noth                 0.0536765239
## worst                0.0536765239
## new                  0.0536221452
## phone                0.0536044876
## appl                 0.0533458207
## care                 0.0527276703
## way                  0.0527276703
## year                 0.0527276703
## updat                0.0519176496
## app                  0.0506802204
## well                 0.0506130885
## iphon                0.0473646206
## help                 0.0470292313
## print                0.0470292313
## servic               0.0470292313
## tech                 0.0470292313
## tell                 0.0470292313
## will                 0.0463979478
## news                 0.0453528423
## data                 0.0453496159
## take                 0.0453496159
## microsoft            0.0450714475
## itun                 0.0449613971
## finger               0.0445884997
## instead              0.0445884997
## wait                 0.0445884997
## card                 0.0429730405
## custom               0.0429730405
## die                  0.0429730405
## event                0.0429730405
## problem              0.0429730405
## refus                0.0429730405
## two                  0.0429730405
## gold                 0.0428877879
## mobil                0.0428877879
## que                  0.0428877879
## avail                0.0420127055
## busi                 0.0420127055
## follow               0.0420127055
## generat              0.0420127055
## wish                 0.0420127055
## back                 0.0405914861
## googl                0.0405914861
## use                  0.0401415689
## big                  0.0392752586
## emoji                0.0392752586
## iphoto               0.0392752586
## motorola             0.0392752586
## touchid              0.0392752586
## screen               0.0387788990
## thing                0.0387788990
## buy                  0.0380283000
## burberri             0.0375405647
## develop              0.0375405647
## video                0.0375405647
## week                 0.0375405647
## fire                 0.0363396181
## great                0.0363396181
## keynot               0.0363396181
## mayb                 0.0363396181
## mishiza              0.0363396181
## natz0711             0.0363396181
## offer                0.0363396181
## para                 0.0363396181
## samsungsa            0.0363396181
## show                 0.0363396181
## watch                0.0363396181
## world                0.0363396181
## need                 0.0357398233
## tri                  0.0356343761
## actual               0.0346046348
## bring                0.0346046348
## share                0.0346046348
## white                0.0346046348
## cdp                  0.0341988646
## doesnt               0.0341988646
## emiss                0.0341988646
## feel                 0.0341988646
## fun                  0.0341988646
## got                  0.0341988646
## hour                 0.0341988646
## macbook              0.0341988646
## miss                 0.0341988646
## siri                 0.0341988646
## start                0.0341988646
## upgrad               0.0341988646
## get                  0.0332332539
## X244tsuyoponzu       0.0331531471
## appstor              0.0331531471
## away                 0.0331531471
## creat                0.0331531471
## design               0.0331531471
## happi                0.0331531471
## ibrooklynb           0.0331531471
## launch               0.0331531471
## lock                 0.0331531471
## page                 0.0331531471
## play                 0.0331531471
## someth               0.0331531471
## talk                 0.0331531471
## touch                0.0331531471
## wonder               0.0331531471
## devic                0.0325564924
## think                0.0314859024
## anyon                0.0305482407
## copi                 0.0303310278
## guess                0.0303310278
## person               0.0303310278
## smart                0.0303310278
## alway                0.0296350116
## condescens           0.0296350116
## crack                0.0296350116
## discontinu           0.0296350116
## email                0.0296350116
## figur                0.0296350116
## internet             0.0296350116
## iphone4              0.0296350116
## lmao                 0.0296350116
## los                  0.0296350116
## move                 0.0296350116
## nuevo                0.0296350116
## perfect              0.0296350116
## photog               0.0296350116
## photographi          0.0296350116
## quiet                0.0296350116
## recommend            0.0296350116
## send                 0.0296350116
## tho                  0.0296350116
## true                 0.0296350116
## user                 0.0296350116
## best                 0.0267580922
## differ               0.0267580922
## nsa                  0.0267580922
## product              0.0267580922
## ios7                 0.0266634900
## peopl                0.0266634900
## releas               0.0266634900
## say                  0.0266634900
## store                0.0259113719
## add                  0.0256490570
## alreadi              0.0256490570
## final                0.0256490570
## genius               0.0256490570
## nfc                  0.0256490570
## pro                  0.0256490570
## yes                  0.0256490570
## just                 0.0237594870
## ppl                  0.0223147692
## price                0.0216753537
## everyth              0.0209296403
## music                0.0209296403
## read                 0.0209296403
## without              0.0209296403
## twitter              0.0207137058
## old                  0.0202889470
## see                  0.0202889470
## color                0.0199770086
## compani              0.0199770086
## know                 0.0199770086
## good                 0.0199770086
## let                  0.0199770086
## stop                 0.0199770086
## download             0.0188746800
## idea                 0.0188746800
## job                  0.0188746800
## mani                 0.0188746800
## simpl                0.0188746800
## soon                 0.0188746800
## team                 0.0188746800
## technolog            0.0188746800
## wow                  0.0188746800
## much                 0.0183336696
## support              0.0183336696
## text                 0.0183336696
## come                 0.0172779254
## free                 0.0162362822
## innov                0.0162362822
## via                  0.0162362822
## time                 0.0159958708
## android              0.0149137629
## black                0.0148064982
## that                 0.0145566668
## want                 0.0141452654
## fingerprint          0.0129942941
## now                  0.0129148701
## .rnorm               0.0127552181
## scanner              0.0121497764
## guy                  0.0109788835
## pleas                0.0108398425
## ask                  0.0102616884
## colour               0.0102616884
## gonna                0.0102616884
## happen               0.0102616884
## man                  0.0102616884
## give                 0.0094009452
## hey                  0.0094009452
## today                0.0094009452
## case                 0.0086651648
## chang                0.0086651648
## blackberri           0.0076273067
## mac                  0.0076273067
## never                0.0076273067
## plastic              0.0076273067
## right                0.0076273067
## sell                 0.0076273067
## one                  0.0070552161
## market               0.0068841106
## work                 0.0057872663
## thank                0.0049626450
## day                  0.0047078144
## nokia                0.0047078144
## preorder             0.0047078144
## next.                0.0036050106
## look                 0.0035739823
## awesom               0.0033167112
## done                 0.0033167112
## featur               0.0033167112
## said                 0.0033167112
## secur                0.0033167112
## seem                 0.0033167112
## smartphon            0.0033167112
## thought              0.0033167112
## what                 0.0033167112
## windowsphon          0.0033167112
## bit                  0.0028697294
## call                 0.0025381970
## didnt                0.0025381970
## drop                 0.0025381970
## fail                 0.0025381970
## first                0.0025381970
## improv               0.0025381970
## isnt                 0.0025381970
## made                 0.0025381970
## mean                 0.0025381970
## put                  0.0025381970
## yet                  0.0025381970
## impress              0.0022363540
## realli               0.0020955452
## can                  0.0017524556
## dont                 0.0006481129
## sure                 0.0003935570
```

```r
glb_script_df <- rbind(glb_script_df, 
    data.frame(chunk_label="remove_correlated_features", 
        chunk_step_major=max(glb_script_df$chunk_step_major),
        chunk_step_minor=glb_script_df[nrow(glb_script_df), "chunk_step_minor"]+1,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))        
print(tail(glb_script_df, 2))
```

```
##                         chunk_label chunk_step_major chunk_step_minor
## elapsed6            select_features                4                0
## elapsed7 remove_correlated_features                4                1
##          elapsed
## elapsed6  14.222
## elapsed7  15.261
```

### Step `4`.`1`: remove correlated features

```r
print(glb_feats_df <- orderBy(~-cor.y, 
          myfind_cor_features(feats_df=glb_feats_df, entity_df=glb_trnent_df, 
                              rsp_var=glb_rsp_var, 
                            checkConditionalX=(glb_is_classification && glb_is_binomial))))
```

```
## Loading required package: caret
## Loading required package: lattice
## 
## Attaching package: 'caret'
## 
## The following object is masked from 'package:survival':
## 
##     cluster
```

```
## [1] "cor(condom, femal)=1.0000"
```

![](Apple_Tweets_files/figure-html/remove_correlated_features-1.png) 

```
## [1] "cor(Negative.fctr, condom)=0.0674"
## [1] "cor(Negative.fctr, femal)=0.0674"
```

```
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
```

```
## Warning in myfind_cor_features(feats_df = glb_feats_df, entity_df =
## glb_trnent_df, : Identified femal as highly correlated with condom
```

![](Apple_Tweets_files/figure-html/remove_correlated_features-2.png) 

```
## [1] "checking correlations for features:"
##   [1] ".rnorm"           "amaz"             "amazon"          
##   [4] "android"          "anyon"            "app"             
##   [7] "appl"             "arent"            "back"            
##  [10] "batteri"          "best"             "better"          
##  [13] "black"            "break."           "buy"             
##  [16] "cant"             "carbon"           "card"            
##  [19] "care"             "cdp"              "charg"           
##  [22] "charger"          "cheap"            "china"           
##  [25] "color"            "come"             "compani"         
##  [28] "condom"           "copi"             "custom"          
##  [31] "darn"             "data"             "date"            
##  [34] "dear"             "devic"            "die"             
##  [37] "differ"           "disappoint"       "divulg"          
##  [40] "doesnt"           "download"         "emiss"           
##  [43] "even"             "event"            "ever"            
##  [46] "everi"            "facebook"         "feel"            
##  [49] "fingerprint"      "fix"              "freak"           
##  [52] "free"             "fun"              "get"             
##  [55] "good"             "googl"            "got"             
##  [58] "guess"            "hate"             "hope"            
##  [61] "hour"             "httpbitly18xc8dk" "idea"            
##  [64] "ill"              "imessag"          "innov"           
##  [67] "ios7"             "ipad"             "iphon"           
##  [70] "iphone5"          "itun"             "ive"             
##  [73] "job"              "just"             "know"            
##  [76] "last"             "let"              "life"            
##  [79] "like"             "line"             "lost"            
##  [82] "macbook"          "make"             "mani"            
##  [85] "microsoft"        "miss"             "money"           
##  [88] "much"             "need"             "new"             
##  [91] "noth"             "now"              "nsa"             
##  [94] "old"              "peopl"            "person"          
##  [97] "phone"            "pictur"           "price"           
## [100] "problem"          "product"          "refus"           
## [103] "releas"           "say"              "screen"          
## [106] "see"              "shame"            "short"           
## [109] "simpl"            "sinc"             "siri"            
## [112] "smart"            "soon"             "stand"           
## [115] "start"            "steve"            "still"           
## [118] "stop"             "store"            "stuff"           
## [121] "stupid"           "support"          "switch"          
## [124] "take"             "team"             "technolog"       
## [127] "text"             "that"             "theyr"           
## [130] "thing"            "think"            "time"            
## [133] "togeth"           "tri"              "turn"            
## [136] "twitter"          "two"              "updat"           
## [139] "upgrad"           "use"              "via"             
## [142] "want"             "way"              "will"            
## [145] "wont"             "worst"            "wow"             
## [148] "wtf"              "X7evenstarz"      "yall"            
## [151] "year"             "yooo"             "your"            
## [1] "cor(emiss, refus)=0.9253"
```

![](Apple_Tweets_files/figure-html/remove_correlated_features-3.png) 

```
## [1] "cor(Negative.fctr, emiss)=0.0342"
## [1] "cor(Negative.fctr, refus)=0.0430"
```

```
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
```

```
## Warning in myfind_cor_features(feats_df = glb_feats_df, entity_df =
## glb_trnent_df, : Identified emiss as highly correlated with refus
```

![](Apple_Tweets_files/figure-html/remove_correlated_features-4.png) 

```
## [1] "checking correlations for features:"
##   [1] ".rnorm"           "amaz"             "amazon"          
##   [4] "android"          "anyon"            "app"             
##   [7] "appl"             "arent"            "back"            
##  [10] "batteri"          "best"             "better"          
##  [13] "black"            "break."           "buy"             
##  [16] "cant"             "carbon"           "card"            
##  [19] "care"             "cdp"              "charg"           
##  [22] "charger"          "cheap"            "china"           
##  [25] "color"            "come"             "compani"         
##  [28] "condom"           "copi"             "custom"          
##  [31] "darn"             "data"             "date"            
##  [34] "dear"             "devic"            "die"             
##  [37] "differ"           "disappoint"       "divulg"          
##  [40] "doesnt"           "download"         "even"            
##  [43] "event"            "ever"             "everi"           
##  [46] "facebook"         "feel"             "fingerprint"     
##  [49] "fix"              "freak"            "free"            
##  [52] "fun"              "get"              "good"            
##  [55] "googl"            "got"              "guess"           
##  [58] "hate"             "hope"             "hour"            
##  [61] "httpbitly18xc8dk" "idea"             "ill"             
##  [64] "imessag"          "innov"            "ios7"            
##  [67] "ipad"             "iphon"            "iphone5"         
##  [70] "itun"             "ive"              "job"             
##  [73] "just"             "know"             "last"            
##  [76] "let"              "life"             "like"            
##  [79] "line"             "lost"             "macbook"         
##  [82] "make"             "mani"             "microsoft"       
##  [85] "miss"             "money"            "much"            
##  [88] "need"             "new"              "noth"            
##  [91] "now"              "nsa"              "old"             
##  [94] "peopl"            "person"           "phone"           
##  [97] "pictur"           "price"            "problem"         
## [100] "product"          "refus"            "releas"          
## [103] "say"              "screen"           "see"             
## [106] "shame"            "short"            "simpl"           
## [109] "sinc"             "siri"             "smart"           
## [112] "soon"             "stand"            "start"           
## [115] "steve"            "still"            "stop"            
## [118] "store"            "stuff"            "stupid"          
## [121] "support"          "switch"           "take"            
## [124] "team"             "technolog"        "text"            
## [127] "that"             "theyr"            "thing"           
## [130] "think"            "time"             "togeth"          
## [133] "tri"              "turn"             "twitter"         
## [136] "two"              "updat"            "upgrad"          
## [139] "use"              "via"              "want"            
## [142] "way"              "will"             "wont"            
## [145] "worst"            "wow"              "wtf"             
## [148] "X7evenstarz"      "yall"             "year"            
## [151] "yooo"             "your"            
## [1] "cor(divulg, refus)=0.9123"
```

![](Apple_Tweets_files/figure-html/remove_correlated_features-5.png) 

```
## [1] "cor(Negative.fctr, divulg)=0.0537"
## [1] "cor(Negative.fctr, refus)=0.0430"
```

```
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
```

```
## Warning in myfind_cor_features(feats_df = glb_feats_df, entity_df =
## glb_trnent_df, : Identified refus as highly correlated with divulg
```

![](Apple_Tweets_files/figure-html/remove_correlated_features-6.png) 

```
## [1] "checking correlations for features:"
##   [1] ".rnorm"           "amaz"             "amazon"          
##   [4] "android"          "anyon"            "app"             
##   [7] "appl"             "arent"            "back"            
##  [10] "batteri"          "best"             "better"          
##  [13] "black"            "break."           "buy"             
##  [16] "cant"             "carbon"           "card"            
##  [19] "care"             "cdp"              "charg"           
##  [22] "charger"          "cheap"            "china"           
##  [25] "color"            "come"             "compani"         
##  [28] "condom"           "copi"             "custom"          
##  [31] "darn"             "data"             "date"            
##  [34] "dear"             "devic"            "die"             
##  [37] "differ"           "disappoint"       "divulg"          
##  [40] "doesnt"           "download"         "even"            
##  [43] "event"            "ever"             "everi"           
##  [46] "facebook"         "feel"             "fingerprint"     
##  [49] "fix"              "freak"            "free"            
##  [52] "fun"              "get"              "good"            
##  [55] "googl"            "got"              "guess"           
##  [58] "hate"             "hope"             "hour"            
##  [61] "httpbitly18xc8dk" "idea"             "ill"             
##  [64] "imessag"          "innov"            "ios7"            
##  [67] "ipad"             "iphon"            "iphone5"         
##  [70] "itun"             "ive"              "job"             
##  [73] "just"             "know"             "last"            
##  [76] "let"              "life"             "like"            
##  [79] "line"             "lost"             "macbook"         
##  [82] "make"             "mani"             "microsoft"       
##  [85] "miss"             "money"            "much"            
##  [88] "need"             "new"              "noth"            
##  [91] "now"              "nsa"              "old"             
##  [94] "peopl"            "person"           "phone"           
##  [97] "pictur"           "price"            "problem"         
## [100] "product"          "releas"           "say"             
## [103] "screen"           "see"              "shame"           
## [106] "short"            "simpl"            "sinc"            
## [109] "siri"             "smart"            "soon"            
## [112] "stand"            "start"            "steve"           
## [115] "still"            "stop"             "store"           
## [118] "stuff"            "stupid"           "support"         
## [121] "switch"           "take"             "team"            
## [124] "technolog"        "text"             "that"            
## [127] "theyr"            "thing"            "think"           
## [130] "time"             "togeth"           "tri"             
## [133] "turn"             "twitter"          "two"             
## [136] "updat"            "upgrad"           "use"             
## [139] "via"              "want"             "way"             
## [142] "will"             "wont"             "worst"           
## [145] "wow"              "wtf"              "X7evenstarz"     
## [148] "yall"             "year"             "yooo"            
## [151] "your"            
## [1] "cor(amazon, facebook)=0.9034"
```

![](Apple_Tweets_files/figure-html/remove_correlated_features-7.png) 

```
## [1] "cor(Negative.fctr, amazon)=0.0852"
## [1] "cor(Negative.fctr, facebook)=0.0682"
```

```
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
```

```
## Warning in myfind_cor_features(feats_df = glb_feats_df, entity_df =
## glb_trnent_df, : Identified facebook as highly correlated with amazon
```

![](Apple_Tweets_files/figure-html/remove_correlated_features-8.png) 

```
## [1] "checking correlations for features:"
##   [1] ".rnorm"           "amaz"             "amazon"          
##   [4] "android"          "anyon"            "app"             
##   [7] "appl"             "arent"            "back"            
##  [10] "batteri"          "best"             "better"          
##  [13] "black"            "break."           "buy"             
##  [16] "cant"             "carbon"           "card"            
##  [19] "care"             "cdp"              "charg"           
##  [22] "charger"          "cheap"            "china"           
##  [25] "color"            "come"             "compani"         
##  [28] "condom"           "copi"             "custom"          
##  [31] "darn"             "data"             "date"            
##  [34] "dear"             "devic"            "die"             
##  [37] "differ"           "disappoint"       "divulg"          
##  [40] "doesnt"           "download"         "even"            
##  [43] "event"            "ever"             "everi"           
##  [46] "feel"             "fingerprint"      "fix"             
##  [49] "freak"            "free"             "fun"             
##  [52] "get"              "good"             "googl"           
##  [55] "got"              "guess"            "hate"            
##  [58] "hope"             "hour"             "httpbitly18xc8dk"
##  [61] "idea"             "ill"              "imessag"         
##  [64] "innov"            "ios7"             "ipad"            
##  [67] "iphon"            "iphone5"          "itun"            
##  [70] "ive"              "job"              "just"            
##  [73] "know"             "last"             "let"             
##  [76] "life"             "like"             "line"            
##  [79] "lost"             "macbook"          "make"            
##  [82] "mani"             "microsoft"        "miss"            
##  [85] "money"            "much"             "need"            
##  [88] "new"              "noth"             "now"             
##  [91] "nsa"              "old"              "peopl"           
##  [94] "person"           "phone"            "pictur"          
##  [97] "price"            "problem"          "product"         
## [100] "releas"           "say"              "screen"          
## [103] "see"              "shame"            "short"           
## [106] "simpl"            "sinc"             "siri"            
## [109] "smart"            "soon"             "stand"           
## [112] "start"            "steve"            "still"           
## [115] "stop"             "store"            "stuff"           
## [118] "stupid"           "support"          "switch"          
## [121] "take"             "team"             "technolog"       
## [124] "text"             "that"             "theyr"           
## [127] "thing"            "think"            "time"            
## [130] "togeth"           "tri"              "turn"            
## [133] "twitter"          "two"              "updat"           
## [136] "upgrad"           "use"              "via"             
## [139] "want"             "way"              "will"            
## [142] "wont"             "worst"            "wow"             
## [145] "wtf"              "X7evenstarz"      "yall"            
## [148] "year"             "yooo"             "your"            
## [1] "cor(carbon, divulg)=0.8441"
```

![](Apple_Tweets_files/figure-html/remove_correlated_features-9.png) 

```
## [1] "cor(Negative.fctr, carbon)=0.0709"
## [1] "cor(Negative.fctr, divulg)=0.0537"
```

```
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
```

```
## Warning in myfind_cor_features(feats_df = glb_feats_df, entity_df =
## glb_trnent_df, : Identified divulg as highly correlated with carbon
```

![](Apple_Tweets_files/figure-html/remove_correlated_features-10.png) 

```
## [1] "checking correlations for features:"
##   [1] ".rnorm"           "amaz"             "amazon"          
##   [4] "android"          "anyon"            "app"             
##   [7] "appl"             "arent"            "back"            
##  [10] "batteri"          "best"             "better"          
##  [13] "black"            "break."           "buy"             
##  [16] "cant"             "carbon"           "card"            
##  [19] "care"             "cdp"              "charg"           
##  [22] "charger"          "cheap"            "china"           
##  [25] "color"            "come"             "compani"         
##  [28] "condom"           "copi"             "custom"          
##  [31] "darn"             "data"             "date"            
##  [34] "dear"             "devic"            "die"             
##  [37] "differ"           "disappoint"       "doesnt"          
##  [40] "download"         "even"             "event"           
##  [43] "ever"             "everi"            "feel"            
##  [46] "fingerprint"      "fix"              "freak"           
##  [49] "free"             "fun"              "get"             
##  [52] "good"             "googl"            "got"             
##  [55] "guess"            "hate"             "hope"            
##  [58] "hour"             "httpbitly18xc8dk" "idea"            
##  [61] "ill"              "imessag"          "innov"           
##  [64] "ios7"             "ipad"             "iphon"           
##  [67] "iphone5"          "itun"             "ive"             
##  [70] "job"              "just"             "know"            
##  [73] "last"             "let"              "life"            
##  [76] "like"             "line"             "lost"            
##  [79] "macbook"          "make"             "mani"            
##  [82] "microsoft"        "miss"             "money"           
##  [85] "much"             "need"             "new"             
##  [88] "noth"             "now"              "nsa"             
##  [91] "old"              "peopl"            "person"          
##  [94] "phone"            "pictur"           "price"           
##  [97] "problem"          "product"          "releas"          
## [100] "say"              "screen"           "see"             
## [103] "shame"            "short"            "simpl"           
## [106] "sinc"             "siri"             "smart"           
## [109] "soon"             "stand"            "start"           
## [112] "steve"            "still"            "stop"            
## [115] "store"            "stuff"            "stupid"          
## [118] "support"          "switch"           "take"            
## [121] "team"             "technolog"        "text"            
## [124] "that"             "theyr"            "thing"           
## [127] "think"            "time"             "togeth"          
## [130] "tri"              "turn"             "twitter"         
## [133] "two"              "updat"            "upgrad"          
## [136] "use"              "via"              "want"            
## [139] "way"              "will"             "wont"            
## [142] "worst"            "wow"              "wtf"             
## [145] "X7evenstarz"      "yall"             "year"            
## [148] "yooo"             "your"            
## [1] "cor(ipad, itun)=0.7867"
```

![](Apple_Tweets_files/figure-html/remove_correlated_features-11.png) 

```
## [1] "cor(Negative.fctr, ipad)=-0.0647"
## [1] "cor(Negative.fctr, itun)=-0.0450"
```

```
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
```

```
## Warning in myfind_cor_features(feats_df = glb_feats_df, entity_df =
## glb_trnent_df, : Identified itun as highly correlated with ipad
```

![](Apple_Tweets_files/figure-html/remove_correlated_features-12.png) 

```
## [1] "checking correlations for features:"
##   [1] ".rnorm"           "amaz"             "amazon"          
##   [4] "android"          "anyon"            "app"             
##   [7] "appl"             "arent"            "back"            
##  [10] "batteri"          "best"             "better"          
##  [13] "black"            "break."           "buy"             
##  [16] "cant"             "carbon"           "card"            
##  [19] "care"             "cdp"              "charg"           
##  [22] "charger"          "cheap"            "china"           
##  [25] "color"            "come"             "compani"         
##  [28] "condom"           "copi"             "custom"          
##  [31] "darn"             "data"             "date"            
##  [34] "dear"             "devic"            "die"             
##  [37] "differ"           "disappoint"       "doesnt"          
##  [40] "download"         "even"             "event"           
##  [43] "ever"             "everi"            "feel"            
##  [46] "fingerprint"      "fix"              "freak"           
##  [49] "free"             "fun"              "get"             
##  [52] "good"             "googl"            "got"             
##  [55] "guess"            "hate"             "hope"            
##  [58] "hour"             "httpbitly18xc8dk" "idea"            
##  [61] "ill"              "imessag"          "innov"           
##  [64] "ios7"             "ipad"             "iphon"           
##  [67] "iphone5"          "ive"              "job"             
##  [70] "just"             "know"             "last"            
##  [73] "let"              "life"             "like"            
##  [76] "line"             "lost"             "macbook"         
##  [79] "make"             "mani"             "microsoft"       
##  [82] "miss"             "money"            "much"            
##  [85] "need"             "new"              "noth"            
##  [88] "now"              "nsa"              "old"             
##  [91] "peopl"            "person"           "phone"           
##  [94] "pictur"           "price"            "problem"         
##  [97] "product"          "releas"           "say"             
## [100] "screen"           "see"              "shame"           
## [103] "short"            "simpl"            "sinc"            
## [106] "siri"             "smart"            "soon"            
## [109] "stand"            "start"            "steve"           
## [112] "still"            "stop"             "store"           
## [115] "stuff"            "stupid"           "support"         
## [118] "switch"           "take"             "team"            
## [121] "technolog"        "text"             "that"            
## [124] "theyr"            "thing"            "think"           
## [127] "time"             "togeth"           "tri"             
## [130] "turn"             "twitter"          "two"             
## [133] "updat"            "upgrad"           "use"             
## [136] "via"              "want"             "way"             
## [139] "will"             "wont"             "worst"           
## [142] "wow"              "wtf"              "X7evenstarz"     
## [145] "yall"             "year"             "yooo"            
## [148] "your"            
## [1] "cor(amazon, cdp)=0.7536"
```

![](Apple_Tweets_files/figure-html/remove_correlated_features-13.png) 

```
## [1] "cor(Negative.fctr, amazon)=0.0852"
## [1] "cor(Negative.fctr, cdp)=0.0342"
```

```
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
```

```
## Warning in myfind_cor_features(feats_df = glb_feats_df, entity_df =
## glb_trnent_df, : Identified cdp as highly correlated with amazon
```

![](Apple_Tweets_files/figure-html/remove_correlated_features-14.png) 

```
## [1] "checking correlations for features:"
##   [1] ".rnorm"           "amaz"             "amazon"          
##   [4] "android"          "anyon"            "app"             
##   [7] "appl"             "arent"            "back"            
##  [10] "batteri"          "best"             "better"          
##  [13] "black"            "break."           "buy"             
##  [16] "cant"             "carbon"           "card"            
##  [19] "care"             "charg"            "charger"         
##  [22] "cheap"            "china"            "color"           
##  [25] "come"             "compani"          "condom"          
##  [28] "copi"             "custom"           "darn"            
##  [31] "data"             "date"             "dear"            
##  [34] "devic"            "die"              "differ"          
##  [37] "disappoint"       "doesnt"           "download"        
##  [40] "even"             "event"            "ever"            
##  [43] "everi"            "feel"             "fingerprint"     
##  [46] "fix"              "freak"            "free"            
##  [49] "fun"              "get"              "good"            
##  [52] "googl"            "got"              "guess"           
##  [55] "hate"             "hope"             "hour"            
##  [58] "httpbitly18xc8dk" "idea"             "ill"             
##  [61] "imessag"          "innov"            "ios7"            
##  [64] "ipad"             "iphon"            "iphone5"         
##  [67] "ive"              "job"              "just"            
##  [70] "know"             "last"             "let"             
##  [73] "life"             "like"             "line"            
##  [76] "lost"             "macbook"          "make"            
##  [79] "mani"             "microsoft"        "miss"            
##  [82] "money"            "much"             "need"            
##  [85] "new"              "noth"             "now"             
##  [88] "nsa"              "old"              "peopl"           
##  [91] "person"           "phone"            "pictur"          
##  [94] "price"            "problem"          "product"         
##  [97] "releas"           "say"              "screen"          
## [100] "see"              "shame"            "short"           
## [103] "simpl"            "sinc"             "siri"            
## [106] "smart"            "soon"             "stand"           
## [109] "start"            "steve"            "still"           
## [112] "stop"             "store"            "stuff"           
## [115] "stupid"           "support"          "switch"          
## [118] "take"             "team"             "technolog"       
## [121] "text"             "that"             "theyr"           
## [124] "thing"            "think"            "time"            
## [127] "togeth"           "tri"              "turn"            
## [130] "twitter"          "two"              "updat"           
## [133] "upgrad"           "use"              "via"             
## [136] "want"             "way"              "will"            
## [139] "wont"             "worst"            "wow"             
## [142] "wtf"              "X7evenstarz"      "yall"            
## [145] "year"             "yooo"             "your"            
## [1] "cor(carbon, httpbitly18xc8dk)=0.7118"
```

![](Apple_Tweets_files/figure-html/remove_correlated_features-15.png) 

```
## [1] "cor(Negative.fctr, carbon)=0.0709"
## [1] "cor(Negative.fctr, httpbitly18xc8dk)=0.0709"
```

```
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
## geom_smooth: Only one unique x value each group.Maybe you want aes(group = 1)?
```

```
## Warning in myfind_cor_features(feats_df = glb_feats_df, entity_df =
## glb_trnent_df, : Identified httpbitly18xc8dk as highly correlated with
## carbon
```

![](Apple_Tweets_files/figure-html/remove_correlated_features-16.png) 

```
## [1] "checking correlations for features:"
##   [1] ".rnorm"      "amaz"        "amazon"      "android"     "anyon"      
##   [6] "app"         "appl"        "arent"       "back"        "batteri"    
##  [11] "best"        "better"      "black"       "break."      "buy"        
##  [16] "cant"        "carbon"      "card"        "care"        "charg"      
##  [21] "charger"     "cheap"       "china"       "color"       "come"       
##  [26] "compani"     "condom"      "copi"        "custom"      "darn"       
##  [31] "data"        "date"        "dear"        "devic"       "die"        
##  [36] "differ"      "disappoint"  "doesnt"      "download"    "even"       
##  [41] "event"       "ever"        "everi"       "feel"        "fingerprint"
##  [46] "fix"         "freak"       "free"        "fun"         "get"        
##  [51] "good"        "googl"       "got"         "guess"       "hate"       
##  [56] "hope"        "hour"        "idea"        "ill"         "imessag"    
##  [61] "innov"       "ios7"        "ipad"        "iphon"       "iphone5"    
##  [66] "ive"         "job"         "just"        "know"        "last"       
##  [71] "let"         "life"        "like"        "line"        "lost"       
##  [76] "macbook"     "make"        "mani"        "microsoft"   "miss"       
##  [81] "money"       "much"        "need"        "new"         "noth"       
##  [86] "now"         "nsa"         "old"         "peopl"       "person"     
##  [91] "phone"       "pictur"      "price"       "problem"     "product"    
##  [96] "releas"      "say"         "screen"      "see"         "shame"      
## [101] "short"       "simpl"       "sinc"        "siri"        "smart"      
## [106] "soon"        "stand"       "start"       "steve"       "still"      
## [111] "stop"        "store"       "stuff"       "stupid"      "support"    
## [116] "switch"      "take"        "team"        "technolog"   "text"       
## [121] "that"        "theyr"       "thing"       "think"       "time"       
## [126] "togeth"      "tri"         "turn"        "twitter"     "two"        
## [131] "updat"       "upgrad"      "use"         "via"         "want"       
## [136] "way"         "will"        "wont"        "worst"       "wow"        
## [141] "wtf"         "X7evenstarz" "yall"        "year"        "yooo"       
## [146] "your"       
##                                        id         cor.y exclude.as.feat
## Tweet.fctr                     Tweet.fctr  0.6354986940               1
## freak                               freak  0.4220126658               0
## hate                                 hate  0.2150466216               0
## stuff                               stuff  0.1685214122               0
## suck                                 suck  0.1644037549               0
## pictur                             pictur  0.1642996876               0
## wtf                                   wtf  0.1642996876               0
## cant                                 cant  0.1518891001               0
## shame                               shame  0.1405061949               0
## stupid                             stupid  0.1186461031               0
## even                                 even  0.1158940633               0
## line                                 line  0.1158940633               0
## yooo                                 yooo  0.1158940633               0
## better                             better  0.1114365401               0
## ever                                 ever  0.1076729878               0
## fix                                   fix  0.1076729878               0
## charger                           charger  0.1069355141               0
## still                               still  0.1007505594               0
## charg                               charg  0.0970913594               0
## disappoint                     disappoint  0.0970913594               0
## short                               short  0.0970913594               0
## like                                 like  0.0962377709               0
## amazon                             amazon  0.0851663937               0
## yall                                 yall  0.0851663937               0
## break.                             break.  0.0826293697               0
## imessag                           imessag  0.0826293697               0
## stand                               stand  0.0826293697               0
## togeth                             togeth  0.0826293697               0
## cheap                               cheap  0.0769587757               0
## wont                                 wont  0.0758516866               0
## make                                 make  0.0727362480               0
## carbon                             carbon  0.0709359262               0
## darn                                 darn  0.0709359262               0
## httpbitly18xc8dk         httpbitly18xc8dk  0.0709359262               0
## dear                                 dear  0.0681568527               0
## facebook                         facebook  0.0681568527               0
## X7evenstarz                   X7evenstarz  0.0673843716               0
## condom                             condom  0.0673843716               0
## femal                               femal  0.0673843716               0
## money                               money  0.0673843716               0
## theyr                               theyr  0.0673843716               0
## batteri                           batteri  0.0626301156               0
## china                               china  0.0625002078               0
## turn                                 turn  0.0625002078               0
## hope                                 hope  0.0611434911               0
## life                                 life  0.0611434911               0
## sinc                                 sinc  0.0611434911               0
## steve                               steve  0.0611434911               0
## switch                             switch  0.0611434911               0
## everi                               everi  0.0564127109               0
## last                                 last  0.0549744869               0
## your                                 your  0.0546349290               0
## amaz                                 amaz  0.0536765239               0
## arent                               arent  0.0536765239               0
## date                                 date  0.0536765239               0
## divulg                             divulg  0.0536765239               0
## ill                                   ill  0.0536765239               0
## ive                                   ive  0.0536765239               0
## lost                                 lost  0.0536765239               0
## noth                                 noth  0.0536765239               0
## worst                               worst  0.0536765239               0
## phone                               phone  0.0536044876               0
## care                                 care  0.0527276703               0
## way                                   way  0.0527276703               0
## year                                 year  0.0527276703               0
## updat                               updat  0.0519176496               0
## app                                   app  0.0506802204               0
## iphon                               iphon  0.0473646206               0
## data                                 data  0.0453496159               0
## take                                 take  0.0453496159               0
## card                                 card  0.0429730405               0
## custom                             custom  0.0429730405               0
## die                                   die  0.0429730405               0
## event                               event  0.0429730405               0
## problem                           problem  0.0429730405               0
## refus                               refus  0.0429730405               0
## two                                   two  0.0429730405               0
## use                                   use  0.0401415689               0
## screen                             screen  0.0387788990               0
## thing                               thing  0.0387788990               0
## buy                                   buy  0.0380283000               0
## tri                                   tri  0.0356343761               0
## cdp                                   cdp  0.0341988646               0
## doesnt                             doesnt  0.0341988646               0
## emiss                               emiss  0.0341988646               0
## feel                                 feel  0.0341988646               0
## fun                                   fun  0.0341988646               0
## got                                   got  0.0341988646               0
## hour                                 hour  0.0341988646               0
## macbook                           macbook  0.0341988646               0
## miss                                 miss  0.0341988646               0
## siri                                 siri  0.0341988646               0
## start                               start  0.0341988646               0
## upgrad                             upgrad  0.0341988646               0
## get                                   get  0.0332332539               0
## think                               think  0.0314859024               0
## copi                                 copi  0.0303310278               0
## guess                               guess  0.0303310278               0
## person                             person  0.0303310278               0
## smart                               smart  0.0303310278               0
## best                                 best  0.0267580922               0
## differ                             differ  0.0267580922               0
## nsa                                   nsa  0.0267580922               0
## product                           product  0.0267580922               0
## twitter                           twitter  0.0207137058               0
## old                                   old  0.0202889470               0
## see                                   see  0.0202889470               0
## download                         download  0.0188746800               0
## idea                                 idea  0.0188746800               0
## job                                   job  0.0188746800               0
## mani                                 mani  0.0188746800               0
## simpl                               simpl  0.0188746800               0
## soon                                 soon  0.0188746800               0
## team                                 team  0.0188746800               0
## technolog                       technolog  0.0188746800               0
## wow                                   wow  0.0188746800               0
## much                                 much  0.0183336696               0
## support                           support  0.0183336696               0
## text                                 text  0.0183336696               0
## time                                 time  0.0159958708               0
## android                           android  0.0149137629               0
## that                                 that  0.0145566668               0
## fingerprint                   fingerprint  0.0129942941               0
## now                                   now  0.0129148701               0
## ask                                   ask  0.0102616884               0
## colour                             colour  0.0102616884               0
## gonna                               gonna  0.0102616884               0
## happen                             happen  0.0102616884               0
## man                                   man  0.0102616884               0
## give                                 give  0.0094009452               0
## hey                                   hey  0.0094009452               0
## today                               today  0.0094009452               0
## case                                 case  0.0086651648               0
## chang                               chang  0.0086651648               0
## one                                   one  0.0070552161               0
## work                                 work  0.0057872663               0
## day                                   day  0.0047078144               0
## nokia                               nokia  0.0047078144               0
## preorder                         preorder  0.0047078144               0
## look                                 look  0.0035739823               0
## awesom                             awesom  0.0033167112               0
## done                                 done  0.0033167112               0
## featur                             featur  0.0033167112               0
## said                                 said  0.0033167112               0
## secur                               secur  0.0033167112               0
## seem                                 seem  0.0033167112               0
## smartphon                       smartphon  0.0033167112               0
## thought                           thought  0.0033167112               0
## what                                 what  0.0033167112               0
## windowsphon                   windowsphon  0.0033167112               0
## bit                                   bit  0.0028697294               0
## realli                             realli  0.0020955452               0
## can                                   can  0.0017524556               0
## sure                                 sure  0.0003935570               0
## dont                                 dont -0.0006481129               0
## impress                           impress -0.0022363540               0
## call                                 call -0.0025381970               0
## didnt                               didnt -0.0025381970               0
## drop                                 drop -0.0025381970               0
## fail                                 fail -0.0025381970               0
## first                               first -0.0025381970               0
## improv                             improv -0.0025381970               0
## isnt                                 isnt -0.0025381970               0
## made                                 made -0.0025381970               0
## mean                                 mean -0.0025381970               0
## put                                   put -0.0025381970               0
## yet                                   yet -0.0025381970               0
## next.                               next. -0.0036050106               0
## thank                               thank -0.0049626450               0
## market                             market -0.0068841106               0
## blackberri                     blackberri -0.0076273067               0
## mac                                   mac -0.0076273067               0
## never                               never -0.0076273067               0
## plastic                           plastic -0.0076273067               0
## right                               right -0.0076273067               0
## sell                                 sell -0.0076273067               0
## pleas                               pleas -0.0108398425               0
## guy                                   guy -0.0109788835               0
## scanner                           scanner -0.0121497764               0
## .rnorm                             .rnorm -0.0127552181               0
## want                                 want -0.0141452654               0
## black                               black -0.0148064982               0
## free                                 free -0.0162362822               0
## innov                               innov -0.0162362822               0
## via                                   via -0.0162362822               0
## come                                 come -0.0172779254               0
## good                                 good -0.0199770086               0
## let                                   let -0.0199770086               0
## stop                                 stop -0.0199770086               0
## color                               color -0.0199770086               0
## compani                           compani -0.0199770086               0
## know                                 know -0.0199770086               0
## everyth                           everyth -0.0209296403               0
## music                               music -0.0209296403               0
## read                                 read -0.0209296403               0
## without                           without -0.0209296403               0
## price                               price -0.0216753537               0
## ppl                                   ppl -0.0223147692               0
## just                                 just -0.0237594870               0
## add                                   add -0.0256490570               0
## alreadi                           alreadi -0.0256490570               0
## final                               final -0.0256490570               0
## genius                             genius -0.0256490570               0
## nfc                                   nfc -0.0256490570               0
## pro                                   pro -0.0256490570               0
## yes                                   yes -0.0256490570               0
## store                               store -0.0259113719               0
## ios7                                 ios7 -0.0266634900               0
## peopl                               peopl -0.0266634900               0
## releas                             releas -0.0266634900               0
## say                                   say -0.0266634900               0
## alway                               alway -0.0296350116               0
## condescens                     condescens -0.0296350116               0
## crack                               crack -0.0296350116               0
## discontinu                     discontinu -0.0296350116               0
## email                               email -0.0296350116               0
## figur                               figur -0.0296350116               0
## internet                         internet -0.0296350116               0
## iphone4                           iphone4 -0.0296350116               0
## lmao                                 lmao -0.0296350116               0
## los                                   los -0.0296350116               0
## move                                 move -0.0296350116               0
## nuevo                               nuevo -0.0296350116               0
## perfect                           perfect -0.0296350116               0
## photog                             photog -0.0296350116               0
## photographi                   photographi -0.0296350116               0
## quiet                               quiet -0.0296350116               0
## recommend                       recommend -0.0296350116               0
## send                                 send -0.0296350116               0
## tho                                   tho -0.0296350116               0
## true                                 true -0.0296350116               0
## user                                 user -0.0296350116               0
## anyon                               anyon -0.0305482407               0
## devic                               devic -0.0325564924               0
## X244tsuyoponzu             X244tsuyoponzu -0.0331531471               0
## appstor                           appstor -0.0331531471               0
## away                                 away -0.0331531471               0
## creat                               creat -0.0331531471               0
## design                             design -0.0331531471               0
## happi                               happi -0.0331531471               0
## ibrooklynb                     ibrooklynb -0.0331531471               0
## launch                             launch -0.0331531471               0
## lock                                 lock -0.0331531471               0
## page                                 page -0.0331531471               0
## play                                 play -0.0331531471               0
## someth                             someth -0.0331531471               0
## talk                                 talk -0.0331531471               0
## touch                               touch -0.0331531471               0
## wonder                             wonder -0.0331531471               0
## actual                             actual -0.0346046348               0
## bring                               bring -0.0346046348               0
## share                               share -0.0346046348               0
## white                               white -0.0346046348               0
## need                                 need -0.0357398233               0
## fire                                 fire -0.0363396181               0
## great                               great -0.0363396181               0
## keynot                             keynot -0.0363396181               0
## mayb                                 mayb -0.0363396181               0
## mishiza                           mishiza -0.0363396181               0
## natz0711                         natz0711 -0.0363396181               0
## offer                               offer -0.0363396181               0
## para                                 para -0.0363396181               0
## samsungsa                       samsungsa -0.0363396181               0
## show                                 show -0.0363396181               0
## watch                               watch -0.0363396181               0
## world                               world -0.0363396181               0
## burberri                         burberri -0.0375405647               0
## develop                           develop -0.0375405647               0
## video                               video -0.0375405647               0
## week                                 week -0.0375405647               0
## big                                   big -0.0392752586               0
## emoji                               emoji -0.0392752586               0
## iphoto                             iphoto -0.0392752586               0
## motorola                         motorola -0.0392752586               0
## touchid                           touchid -0.0392752586               0
## back                                 back -0.0405914861               0
## googl                               googl -0.0405914861               0
## avail                               avail -0.0420127055               0
## busi                                 busi -0.0420127055               0
## follow                             follow -0.0420127055               0
## generat                           generat -0.0420127055               0
## wish                                 wish -0.0420127055               0
## gold                                 gold -0.0428877879               0
## mobil                               mobil -0.0428877879               0
## que                                   que -0.0428877879               0
## finger                             finger -0.0445884997               0
## instead                           instead -0.0445884997               0
## wait                                 wait -0.0445884997               0
## itun                                 itun -0.0449613971               0
## microsoft                       microsoft -0.0450714475               0
## news                                 news -0.0453528423               0
## will                                 will -0.0463979478               0
## help                                 help -0.0470292313               0
## print                               print -0.0470292313               0
## servic                             servic -0.0470292313               0
## tech                                 tech -0.0470292313               0
## tell                                 tell -0.0470292313               0
## well                                 well -0.0506130885               0
## appl                                 appl -0.0533458207               0
## new                                   new -0.0536221452               0
## lol                                   lol -0.0537205371               0
## announc                           announc -0.0557827956               0
## love                                 love -0.0577763105               0
## promo                               promo -0.0615836556               0
## samsung                           samsung -0.0615836556               0
## iphone5                           iphone5 -0.0621379328               0
## iphone5c                         iphone5c -0.0645077026               0
## ipad                                 ipad -0.0646730918               0
## ipodplayerpromo           ipodplayerpromo -0.0793460925               0
## promoipodplayerpromo promoipodplayerpromo -0.0838913688               0
## ipod                                 ipod -0.0852874403               0
## Avg                                   Avg -0.7061583707               1
##                         cor.y.abs       cor.high.X is.ConditionalX.y
## Tweet.fctr           0.6354986940             <NA>                NA
## freak                0.4220126658             <NA>              TRUE
## hate                 0.2150466216             <NA>              TRUE
## stuff                0.1685214122             <NA>              TRUE
## suck                 0.1644037549             <NA>             FALSE
## pictur               0.1642996876             <NA>              TRUE
## wtf                  0.1642996876             <NA>              TRUE
## cant                 0.1518891001             <NA>              TRUE
## shame                0.1405061949             <NA>              TRUE
## stupid               0.1186461031             <NA>              TRUE
## even                 0.1158940633             <NA>              TRUE
## line                 0.1158940633             <NA>              TRUE
## yooo                 0.1158940633             <NA>              TRUE
## better               0.1114365401             <NA>              TRUE
## ever                 0.1076729878             <NA>              TRUE
## fix                  0.1076729878             <NA>              TRUE
## charger              0.1069355141             <NA>              TRUE
## still                0.1007505594             <NA>              TRUE
## charg                0.0970913594             <NA>              TRUE
## disappoint           0.0970913594             <NA>              TRUE
## short                0.0970913594             <NA>              TRUE
## like                 0.0962377709             <NA>              TRUE
## amazon               0.0851663937              cdp              TRUE
## yall                 0.0851663937             <NA>              TRUE
## break.               0.0826293697             <NA>              TRUE
## imessag              0.0826293697             <NA>              TRUE
## stand                0.0826293697             <NA>              TRUE
## togeth               0.0826293697             <NA>              TRUE
## cheap                0.0769587757             <NA>              TRUE
## wont                 0.0758516866             <NA>              TRUE
## make                 0.0727362480             <NA>              TRUE
## carbon               0.0709359262 httpbitly18xc8dk              TRUE
## darn                 0.0709359262             <NA>              TRUE
## httpbitly18xc8dk     0.0709359262             <NA>              TRUE
## dear                 0.0681568527             <NA>              TRUE
## facebook             0.0681568527             <NA>              TRUE
## X7evenstarz          0.0673843716             <NA>              TRUE
## condom               0.0673843716            femal              TRUE
## femal                0.0673843716             <NA>              TRUE
## money                0.0673843716             <NA>              TRUE
## theyr                0.0673843716             <NA>              TRUE
## batteri              0.0626301156             <NA>              TRUE
## china                0.0625002078             <NA>              TRUE
## turn                 0.0625002078             <NA>              TRUE
## hope                 0.0611434911             <NA>              TRUE
## life                 0.0611434911             <NA>              TRUE
## sinc                 0.0611434911             <NA>              TRUE
## steve                0.0611434911             <NA>              TRUE
## switch               0.0611434911             <NA>              TRUE
## everi                0.0564127109             <NA>              TRUE
## last                 0.0549744869             <NA>              TRUE
## your                 0.0546349290             <NA>              TRUE
## amaz                 0.0536765239             <NA>              TRUE
## arent                0.0536765239             <NA>              TRUE
## date                 0.0536765239             <NA>              TRUE
## divulg               0.0536765239            refus              TRUE
## ill                  0.0536765239             <NA>              TRUE
## ive                  0.0536765239             <NA>              TRUE
## lost                 0.0536765239             <NA>              TRUE
## noth                 0.0536765239             <NA>              TRUE
## worst                0.0536765239             <NA>              TRUE
## phone                0.0536044876             <NA>              TRUE
## care                 0.0527276703             <NA>              TRUE
## way                  0.0527276703             <NA>              TRUE
## year                 0.0527276703             <NA>              TRUE
## updat                0.0519176496             <NA>              TRUE
## app                  0.0506802204             <NA>              TRUE
## iphon                0.0473646206             <NA>              TRUE
## data                 0.0453496159             <NA>              TRUE
## take                 0.0453496159             <NA>              TRUE
## card                 0.0429730405             <NA>              TRUE
## custom               0.0429730405             <NA>              TRUE
## die                  0.0429730405             <NA>              TRUE
## event                0.0429730405             <NA>              TRUE
## problem              0.0429730405             <NA>              TRUE
## refus                0.0429730405            emiss              TRUE
## two                  0.0429730405             <NA>              TRUE
## use                  0.0401415689             <NA>              TRUE
## screen               0.0387788990             <NA>              TRUE
## thing                0.0387788990             <NA>              TRUE
## buy                  0.0380283000             <NA>              TRUE
## tri                  0.0356343761             <NA>              TRUE
## cdp                  0.0341988646             <NA>              TRUE
## doesnt               0.0341988646             <NA>              TRUE
## emiss                0.0341988646             <NA>              TRUE
## feel                 0.0341988646             <NA>              TRUE
## fun                  0.0341988646             <NA>              TRUE
## got                  0.0341988646             <NA>              TRUE
## hour                 0.0341988646             <NA>              TRUE
## macbook              0.0341988646             <NA>              TRUE
## miss                 0.0341988646             <NA>              TRUE
## siri                 0.0341988646             <NA>              TRUE
## start                0.0341988646             <NA>              TRUE
## upgrad               0.0341988646             <NA>              TRUE
## get                  0.0332332539             <NA>              TRUE
## think                0.0314859024             <NA>              TRUE
## copi                 0.0303310278             <NA>              TRUE
## guess                0.0303310278             <NA>              TRUE
## person               0.0303310278             <NA>              TRUE
## smart                0.0303310278             <NA>              TRUE
## best                 0.0267580922             <NA>              TRUE
## differ               0.0267580922             <NA>              TRUE
## nsa                  0.0267580922             <NA>              TRUE
## product              0.0267580922             <NA>              TRUE
## twitter              0.0207137058             <NA>              TRUE
## old                  0.0202889470             <NA>              TRUE
## see                  0.0202889470             <NA>              TRUE
## download             0.0188746800             <NA>              TRUE
## idea                 0.0188746800             <NA>              TRUE
## job                  0.0188746800             <NA>              TRUE
## mani                 0.0188746800             <NA>              TRUE
## simpl                0.0188746800             <NA>              TRUE
## soon                 0.0188746800             <NA>              TRUE
## team                 0.0188746800             <NA>              TRUE
## technolog            0.0188746800             <NA>              TRUE
## wow                  0.0188746800             <NA>              TRUE
## much                 0.0183336696             <NA>              TRUE
## support              0.0183336696             <NA>              TRUE
## text                 0.0183336696             <NA>              TRUE
## time                 0.0159958708             <NA>              TRUE
## android              0.0149137629             <NA>              TRUE
## that                 0.0145566668             <NA>              TRUE
## fingerprint          0.0129942941             <NA>              TRUE
## now                  0.0129148701             <NA>              TRUE
## ask                  0.0102616884             <NA>              TRUE
## colour               0.0102616884             <NA>              TRUE
## gonna                0.0102616884             <NA>              TRUE
## happen               0.0102616884             <NA>              TRUE
## man                  0.0102616884             <NA>              TRUE
## give                 0.0094009452             <NA>              TRUE
## hey                  0.0094009452             <NA>              TRUE
## today                0.0094009452             <NA>              TRUE
## case                 0.0086651648             <NA>              TRUE
## chang                0.0086651648             <NA>              TRUE
## one                  0.0070552161             <NA>              TRUE
## work                 0.0057872663             <NA>              TRUE
## day                  0.0047078144             <NA>              TRUE
## nokia                0.0047078144             <NA>              TRUE
## preorder             0.0047078144             <NA>              TRUE
## look                 0.0035739823             <NA>              TRUE
## awesom               0.0033167112             <NA>              TRUE
## done                 0.0033167112             <NA>              TRUE
## featur               0.0033167112             <NA>              TRUE
## said                 0.0033167112             <NA>              TRUE
## secur                0.0033167112             <NA>              TRUE
## seem                 0.0033167112             <NA>              TRUE
## smartphon            0.0033167112             <NA>              TRUE
## thought              0.0033167112             <NA>              TRUE
## what                 0.0033167112             <NA>              TRUE
## windowsphon          0.0033167112             <NA>              TRUE
## bit                  0.0028697294             <NA>              TRUE
## realli               0.0020955452             <NA>              TRUE
## can                  0.0017524556             <NA>              TRUE
## sure                 0.0003935570             <NA>              TRUE
## dont                 0.0006481129             <NA>              TRUE
## impress              0.0022363540             <NA>              TRUE
## call                 0.0025381970             <NA>              TRUE
## didnt                0.0025381970             <NA>              TRUE
## drop                 0.0025381970             <NA>              TRUE
## fail                 0.0025381970             <NA>              TRUE
## first                0.0025381970             <NA>              TRUE
## improv               0.0025381970             <NA>              TRUE
## isnt                 0.0025381970             <NA>              TRUE
## made                 0.0025381970             <NA>              TRUE
## mean                 0.0025381970             <NA>              TRUE
## put                  0.0025381970             <NA>              TRUE
## yet                  0.0025381970             <NA>              TRUE
## next.                0.0036050106             <NA>              TRUE
## thank                0.0049626450             <NA>              TRUE
## market               0.0068841106             <NA>              TRUE
## blackberri           0.0076273067             <NA>              TRUE
## mac                  0.0076273067             <NA>              TRUE
## never                0.0076273067             <NA>              TRUE
## plastic              0.0076273067             <NA>              TRUE
## right                0.0076273067             <NA>              TRUE
## sell                 0.0076273067             <NA>              TRUE
## pleas                0.0108398425             <NA>              TRUE
## guy                  0.0109788835             <NA>              TRUE
## scanner              0.0121497764             <NA>              TRUE
## .rnorm               0.0127552181             <NA>              TRUE
## want                 0.0141452654             <NA>              TRUE
## black                0.0148064982             <NA>              TRUE
## free                 0.0162362822             <NA>              TRUE
## innov                0.0162362822             <NA>              TRUE
## via                  0.0162362822             <NA>              TRUE
## come                 0.0172779254             <NA>              TRUE
## good                 0.0199770086             <NA>              TRUE
## let                  0.0199770086             <NA>              TRUE
## stop                 0.0199770086             <NA>              TRUE
## color                0.0199770086             <NA>              TRUE
## compani              0.0199770086             <NA>              TRUE
## know                 0.0199770086             <NA>              TRUE
## everyth              0.0209296403             <NA>             FALSE
## music                0.0209296403             <NA>             FALSE
## read                 0.0209296403             <NA>             FALSE
## without              0.0209296403             <NA>             FALSE
## price                0.0216753537             <NA>              TRUE
## ppl                  0.0223147692             <NA>             FALSE
## just                 0.0237594870             <NA>              TRUE
## add                  0.0256490570             <NA>             FALSE
## alreadi              0.0256490570             <NA>             FALSE
## final                0.0256490570             <NA>             FALSE
## genius               0.0256490570             <NA>             FALSE
## nfc                  0.0256490570             <NA>             FALSE
## pro                  0.0256490570             <NA>             FALSE
## yes                  0.0256490570             <NA>             FALSE
## store                0.0259113719             <NA>              TRUE
## ios7                 0.0266634900             <NA>              TRUE
## peopl                0.0266634900             <NA>              TRUE
## releas               0.0266634900             <NA>              TRUE
## say                  0.0266634900             <NA>              TRUE
## alway                0.0296350116             <NA>             FALSE
## condescens           0.0296350116             <NA>             FALSE
## crack                0.0296350116             <NA>             FALSE
## discontinu           0.0296350116             <NA>             FALSE
## email                0.0296350116             <NA>             FALSE
## figur                0.0296350116             <NA>             FALSE
## internet             0.0296350116             <NA>             FALSE
## iphone4              0.0296350116             <NA>             FALSE
## lmao                 0.0296350116             <NA>             FALSE
## los                  0.0296350116             <NA>             FALSE
## move                 0.0296350116             <NA>             FALSE
## nuevo                0.0296350116             <NA>             FALSE
## perfect              0.0296350116             <NA>             FALSE
## photog               0.0296350116             <NA>             FALSE
## photographi          0.0296350116             <NA>             FALSE
## quiet                0.0296350116             <NA>             FALSE
## recommend            0.0296350116             <NA>             FALSE
## send                 0.0296350116             <NA>             FALSE
## tho                  0.0296350116             <NA>             FALSE
## true                 0.0296350116             <NA>             FALSE
## user                 0.0296350116             <NA>             FALSE
## anyon                0.0305482407             <NA>              TRUE
## devic                0.0325564924             <NA>              TRUE
## X244tsuyoponzu       0.0331531471             <NA>             FALSE
## appstor              0.0331531471             <NA>             FALSE
## away                 0.0331531471             <NA>             FALSE
## creat                0.0331531471             <NA>             FALSE
## design               0.0331531471             <NA>             FALSE
## happi                0.0331531471             <NA>             FALSE
## ibrooklynb           0.0331531471             <NA>             FALSE
## launch               0.0331531471             <NA>             FALSE
## lock                 0.0331531471             <NA>             FALSE
## page                 0.0331531471             <NA>             FALSE
## play                 0.0331531471             <NA>             FALSE
## someth               0.0331531471             <NA>             FALSE
## talk                 0.0331531471             <NA>             FALSE
## touch                0.0331531471             <NA>             FALSE
## wonder               0.0331531471             <NA>             FALSE
## actual               0.0346046348             <NA>             FALSE
## bring                0.0346046348             <NA>             FALSE
## share                0.0346046348             <NA>             FALSE
## white                0.0346046348             <NA>             FALSE
## need                 0.0357398233             <NA>              TRUE
## fire                 0.0363396181             <NA>             FALSE
## great                0.0363396181             <NA>             FALSE
## keynot               0.0363396181             <NA>             FALSE
## mayb                 0.0363396181             <NA>             FALSE
## mishiza              0.0363396181             <NA>             FALSE
## natz0711             0.0363396181             <NA>             FALSE
## offer                0.0363396181             <NA>             FALSE
## para                 0.0363396181             <NA>             FALSE
## samsungsa            0.0363396181             <NA>             FALSE
## show                 0.0363396181             <NA>             FALSE
## watch                0.0363396181             <NA>             FALSE
## world                0.0363396181             <NA>             FALSE
## burberri             0.0375405647             <NA>             FALSE
## develop              0.0375405647             <NA>             FALSE
## video                0.0375405647             <NA>             FALSE
## week                 0.0375405647             <NA>             FALSE
## big                  0.0392752586             <NA>             FALSE
## emoji                0.0392752586             <NA>             FALSE
## iphoto               0.0392752586             <NA>             FALSE
## motorola             0.0392752586             <NA>             FALSE
## touchid              0.0392752586             <NA>             FALSE
## back                 0.0405914861             <NA>              TRUE
## googl                0.0405914861             <NA>              TRUE
## avail                0.0420127055             <NA>             FALSE
## busi                 0.0420127055             <NA>             FALSE
## follow               0.0420127055             <NA>             FALSE
## generat              0.0420127055             <NA>             FALSE
## wish                 0.0420127055             <NA>             FALSE
## gold                 0.0428877879             <NA>             FALSE
## mobil                0.0428877879             <NA>             FALSE
## que                  0.0428877879             <NA>             FALSE
## finger               0.0445884997             <NA>             FALSE
## instead              0.0445884997             <NA>             FALSE
## wait                 0.0445884997             <NA>             FALSE
## itun                 0.0449613971             <NA>              TRUE
## microsoft            0.0450714475             <NA>              TRUE
## news                 0.0453528423             <NA>             FALSE
## will                 0.0463979478             <NA>              TRUE
## help                 0.0470292313             <NA>             FALSE
## print                0.0470292313             <NA>             FALSE
## servic               0.0470292313             <NA>             FALSE
## tech                 0.0470292313             <NA>             FALSE
## tell                 0.0470292313             <NA>             FALSE
## well                 0.0506130885             <NA>             FALSE
## appl                 0.0533458207             <NA>              TRUE
## new                  0.0536221452             <NA>              TRUE
## lol                  0.0537205371             <NA>             FALSE
## announc              0.0557827956             <NA>             FALSE
## love                 0.0577763105             <NA>             FALSE
## promo                0.0615836556             <NA>             FALSE
## samsung              0.0615836556             <NA>             FALSE
## iphone5              0.0621379328             <NA>              TRUE
## iphone5c             0.0645077026             <NA>             FALSE
## ipad                 0.0646730918             itun              TRUE
## ipodplayerpromo      0.0793460925             <NA>             FALSE
## promoipodplayerpromo 0.0838913688             <NA>             FALSE
## ipod                 0.0852874403             <NA>             FALSE
## Avg                  0.7061583707             <NA>                NA
##                      is.cor.y.abs.low
## Tweet.fctr                      FALSE
## freak                           FALSE
## hate                            FALSE
## stuff                           FALSE
## suck                            FALSE
## pictur                          FALSE
## wtf                             FALSE
## cant                            FALSE
## shame                           FALSE
## stupid                          FALSE
## even                            FALSE
## line                            FALSE
## yooo                            FALSE
## better                          FALSE
## ever                            FALSE
## fix                             FALSE
## charger                         FALSE
## still                           FALSE
## charg                           FALSE
## disappoint                      FALSE
## short                           FALSE
## like                            FALSE
## amazon                          FALSE
## yall                            FALSE
## break.                          FALSE
## imessag                         FALSE
## stand                           FALSE
## togeth                          FALSE
## cheap                           FALSE
## wont                            FALSE
## make                            FALSE
## carbon                          FALSE
## darn                            FALSE
## httpbitly18xc8dk                FALSE
## dear                            FALSE
## facebook                        FALSE
## X7evenstarz                     FALSE
## condom                          FALSE
## femal                           FALSE
## money                           FALSE
## theyr                           FALSE
## batteri                         FALSE
## china                           FALSE
## turn                            FALSE
## hope                            FALSE
## life                            FALSE
## sinc                            FALSE
## steve                           FALSE
## switch                          FALSE
## everi                           FALSE
## last                            FALSE
## your                            FALSE
## amaz                            FALSE
## arent                           FALSE
## date                            FALSE
## divulg                          FALSE
## ill                             FALSE
## ive                             FALSE
## lost                            FALSE
## noth                            FALSE
## worst                           FALSE
## phone                           FALSE
## care                            FALSE
## way                             FALSE
## year                            FALSE
## updat                           FALSE
## app                             FALSE
## iphon                           FALSE
## data                            FALSE
## take                            FALSE
## card                            FALSE
## custom                          FALSE
## die                             FALSE
## event                           FALSE
## problem                         FALSE
## refus                           FALSE
## two                             FALSE
## use                             FALSE
## screen                          FALSE
## thing                           FALSE
## buy                             FALSE
## tri                             FALSE
## cdp                             FALSE
## doesnt                          FALSE
## emiss                           FALSE
## feel                            FALSE
## fun                             FALSE
## got                             FALSE
## hour                            FALSE
## macbook                         FALSE
## miss                            FALSE
## siri                            FALSE
## start                           FALSE
## upgrad                          FALSE
## get                             FALSE
## think                           FALSE
## copi                            FALSE
## guess                           FALSE
## person                          FALSE
## smart                           FALSE
## best                            FALSE
## differ                          FALSE
## nsa                             FALSE
## product                         FALSE
## twitter                         FALSE
## old                             FALSE
## see                             FALSE
## download                        FALSE
## idea                            FALSE
## job                             FALSE
## mani                            FALSE
## simpl                           FALSE
## soon                            FALSE
## team                            FALSE
## technolog                       FALSE
## wow                             FALSE
## much                            FALSE
## support                         FALSE
## text                            FALSE
## time                            FALSE
## android                         FALSE
## that                            FALSE
## fingerprint                     FALSE
## now                             FALSE
## ask                              TRUE
## colour                           TRUE
## gonna                            TRUE
## happen                           TRUE
## man                              TRUE
## give                             TRUE
## hey                              TRUE
## today                            TRUE
## case                             TRUE
## chang                            TRUE
## one                              TRUE
## work                             TRUE
## day                              TRUE
## nokia                            TRUE
## preorder                         TRUE
## look                             TRUE
## awesom                           TRUE
## done                             TRUE
## featur                           TRUE
## said                             TRUE
## secur                            TRUE
## seem                             TRUE
## smartphon                        TRUE
## thought                          TRUE
## what                             TRUE
## windowsphon                      TRUE
## bit                              TRUE
## realli                           TRUE
## can                              TRUE
## sure                             TRUE
## dont                             TRUE
## impress                          TRUE
## call                             TRUE
## didnt                            TRUE
## drop                             TRUE
## fail                             TRUE
## first                            TRUE
## improv                           TRUE
## isnt                             TRUE
## made                             TRUE
## mean                             TRUE
## put                              TRUE
## yet                              TRUE
## next.                            TRUE
## thank                            TRUE
## market                           TRUE
## blackberri                       TRUE
## mac                              TRUE
## never                            TRUE
## plastic                          TRUE
## right                            TRUE
## sell                             TRUE
## pleas                            TRUE
## guy                              TRUE
## scanner                          TRUE
## .rnorm                          FALSE
## want                            FALSE
## black                           FALSE
## free                            FALSE
## innov                           FALSE
## via                             FALSE
## come                            FALSE
## good                            FALSE
## let                             FALSE
## stop                            FALSE
## color                           FALSE
## compani                         FALSE
## know                            FALSE
## everyth                         FALSE
## music                           FALSE
## read                            FALSE
## without                         FALSE
## price                           FALSE
## ppl                             FALSE
## just                            FALSE
## add                             FALSE
## alreadi                         FALSE
## final                           FALSE
## genius                          FALSE
## nfc                             FALSE
## pro                             FALSE
## yes                             FALSE
## store                           FALSE
## ios7                            FALSE
## peopl                           FALSE
## releas                          FALSE
## say                             FALSE
## alway                           FALSE
## condescens                      FALSE
## crack                           FALSE
## discontinu                      FALSE
## email                           FALSE
## figur                           FALSE
## internet                        FALSE
## iphone4                         FALSE
## lmao                            FALSE
## los                             FALSE
## move                            FALSE
## nuevo                           FALSE
## perfect                         FALSE
## photog                          FALSE
## photographi                     FALSE
## quiet                           FALSE
## recommend                       FALSE
## send                            FALSE
## tho                             FALSE
## true                            FALSE
## user                            FALSE
## anyon                           FALSE
## devic                           FALSE
## X244tsuyoponzu                  FALSE
## appstor                         FALSE
## away                            FALSE
## creat                           FALSE
## design                          FALSE
## happi                           FALSE
## ibrooklynb                      FALSE
## launch                          FALSE
## lock                            FALSE
## page                            FALSE
## play                            FALSE
## someth                          FALSE
## talk                            FALSE
## touch                           FALSE
## wonder                          FALSE
## actual                          FALSE
## bring                           FALSE
## share                           FALSE
## white                           FALSE
## need                            FALSE
## fire                            FALSE
## great                           FALSE
## keynot                          FALSE
## mayb                            FALSE
## mishiza                         FALSE
## natz0711                        FALSE
## offer                           FALSE
## para                            FALSE
## samsungsa                       FALSE
## show                            FALSE
## watch                           FALSE
## world                           FALSE
## burberri                        FALSE
## develop                         FALSE
## video                           FALSE
## week                            FALSE
## big                             FALSE
## emoji                           FALSE
## iphoto                          FALSE
## motorola                        FALSE
## touchid                         FALSE
## back                            FALSE
## googl                           FALSE
## avail                           FALSE
## busi                            FALSE
## follow                          FALSE
## generat                         FALSE
## wish                            FALSE
## gold                            FALSE
## mobil                           FALSE
## que                             FALSE
## finger                          FALSE
## instead                         FALSE
## wait                            FALSE
## itun                            FALSE
## microsoft                       FALSE
## news                            FALSE
## will                            FALSE
## help                            FALSE
## print                           FALSE
## servic                          FALSE
## tech                            FALSE
## tell                            FALSE
## well                            FALSE
## appl                            FALSE
## new                             FALSE
## lol                             FALSE
## announc                         FALSE
## love                            FALSE
## promo                           FALSE
## samsung                         FALSE
## iphone5                         FALSE
## iphone5c                        FALSE
## ipad                            FALSE
## ipodplayerpromo                 FALSE
## promoipodplayerpromo            FALSE
## ipod                            FALSE
## Avg                             FALSE
```

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="fit.models", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                         chunk_label chunk_step_major chunk_step_minor
## elapsed7 remove_correlated_features                4                1
## elapsed8                 fit.models                5                0
##          elapsed
## elapsed7  15.261
## elapsed8  24.668
```

## Step `5`: fit models

```r
if (glb_is_classification && glb_is_binomial && 
        (length(unique(glb_trnent_df[, glb_rsp_var])) < 2))
    stop("glb_trnent_df$", glb_rsp_var, ": contains less than 2 unique values: ",
         paste0(unique(glb_trnent_df[, glb_rsp_var]), collapse=", "))

max_cor_y_x_var <- orderBy(~ -cor.y.abs, 
        subset(glb_feats_df, (exclude.as.feat == 0) & !is.cor.y.abs.low))[1, "id"]
if (!is.null(glb_Baseline_mdl_var)) {
    if ((max_cor_y_x_var != glb_Baseline_mdl_var) & 
        (glb_feats_df[max_cor_y_x_var, "cor.y.abs"] > 
         glb_feats_df[glb_Baseline_mdl_var, "cor.y.abs"]))
        stop(max_cor_y_x_var, " has a lower correlation with ", glb_rsp_var, 
             " than the Baseline var: ", glb_Baseline_mdl_var)
}

glb_model_type <- ifelse(glb_is_regression, "regression", "classification")
    
# Baseline
if (!is.null(glb_Baseline_mdl_var)) 
    ret_lst <- myfit_mdl_fn(model_id="Baseline", model_method="mybaseln_classfr",
                            indep_vars_vctr=glb_Baseline_mdl_var,
                            rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                            fit_df=glb_trnent_df, OOB_df=glb_newent_df)

# Most Frequent Outcome "MFO" model: mean(y) for regression
#   Not using caret's nullModel since model stats not avl
#   Cannot use rpart for multinomial classification since it predicts non-MFO
ret_lst <- myfit_mdl(model_id="MFO", 
                     model_method=ifelse(glb_is_regression, "lm", "myMFO_classfr"), 
                     model_type=glb_model_type,
                        indep_vars_vctr=".rnorm",
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df)
```

```
## [1] "fitting model: MFO.myMFO_classfr"
## [1] "    indep_vars: .rnorm"
## Fitting parameter = none on full training set
## [1] "in MFO.Classifier$fit"
## [1] "unique.vals:"
## [1] N Y
## Levels: N Y
## [1] "unique.prob:"
## y
##         N         Y 
## 0.8472727 0.1527273 
## [1] "MFO.val:"
## [1] "N"
##             Length Class      Mode     
## unique.vals 2      factor     numeric  
## unique.prob 2      -none-     numeric  
## MFO.val     1      -none-     character
## x.names     1      -none-     character
## xNames      1      -none-     character
## problemType 1      -none-     character
## tuneValue   1      data.frame list     
## obsLevels   2      -none-     character
```

```
## Loading required package: ROCR
## Loading required package: gplots
## 
## Attaching package: 'gplots'
## 
## The following object is masked from 'package:stats':
## 
##     lowess
```

```
## [1] "in MFO.Classifier$predict"
## [1] "in MFO.Classifier$prob"
##           N         Y
## 1 0.8472727 0.1527273
## 2 0.8472727 0.1527273
## 3 0.8472727 0.1527273
## 4 0.8472727 0.1527273
## 5 0.8472727 0.1527273
## 6 0.8472727 0.1527273
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.fit"
##   Negative.fctr Negative.fctr.predict.MFO.myMFO_classfr.N
## 1             N                                       699
## 2             Y                                       126
##           Reference
## Prediction   N   Y
##          N 699 126
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.MFO.myMFO_classfr.N
## 1             N                                       699
## 2             Y                                       126
##   Negative.fctr.predict.MFO.myMFO_classfr.Y
## 1                                         0
## 2                                         0
##          Prediction
## Reference   N   Y
##         N 699   0
##         Y 126   0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.472727e-01   0.000000e+00   8.208848e-01   8.711498e-01   8.472727e-01 
## AccuracyPValue  McnemarPValue 
##   5.237511e-01   8.390732e-29 
## [1] "in MFO.Classifier$predict"
## [1] "in MFO.Classifier$prob"
##           N         Y
## 1 0.8472727 0.1527273
## 2 0.8472727 0.1527273
## 3 0.8472727 0.1527273
## 4 0.8472727 0.1527273
## 5 0.8472727 0.1527273
## 6 0.8472727 0.1527273
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.OOB"
##   Negative.fctr Negative.fctr.predict.MFO.myMFO_classfr.N
## 1             N                                       300
## 2             Y                                        56
##           Reference
## Prediction   N   Y
##          N 300  56
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.MFO.myMFO_classfr.N
## 1             N                                       300
## 2             Y                                        56
##   Negative.fctr.predict.MFO.myMFO_classfr.Y
## 1                                         0
## 2                                         0
##          Prediction
## Reference   N   Y
##         N 300   0
##         Y  56   0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.426966e-01   0.000000e+00   8.006423e-01   8.789342e-01   8.426966e-01 
## AccuracyPValue  McnemarPValue 
##   5.355897e-01   1.986758e-13 
##            model_id  model_method  feats max.nTuningRuns
## 1 MFO.myMFO_classfr myMFO_classfr .rnorm               0
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                      0.304                 0.002         0.5
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.5               0        0.8472727
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.8208848             0.8711498             0         0.5
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.5               0        0.8426966
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.8006423             0.8789342             0
```

```r
if (glb_is_classification)
    # "random" model - only for classification; 
    #   none needed for regression since it is same as MFO
    ret_lst <- myfit_mdl(model_id="Random", model_method="myrandom_classfr",
                            model_type=glb_model_type,                         
                            indep_vars_vctr=".rnorm",
                            rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                            fit_df=glb_trnent_df, OOB_df=glb_newent_df)
```

```
## [1] "fitting model: Random.myrandom_classfr"
## [1] "    indep_vars: .rnorm"
## Fitting parameter = none on full training set
##             Length Class      Mode     
## unique.vals 2      factor     numeric  
## unique.prob 2      table      numeric  
## xNames      1      -none-     character
## problemType 1      -none-     character
## tuneValue   1      data.frame list     
## obsLevels   2      -none-     character
## [1] "in Random.Classifier$prob"
```

![](Apple_Tweets_files/figure-html/fit.models_0-1.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 699 126
##   Negative.fctr Negative.fctr.predict.Random.myrandom_classfr.N
## 1             N                                               0
## 2             Y                                               0
##   Negative.fctr.predict.Random.myrandom_classfr.Y
## 1                                             699
## 2                                             126
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 699 126
##   Negative.fctr Negative.fctr.predict.Random.myrandom_classfr.N
## 1             N                                               0
## 2             Y                                               0
##   Negative.fctr.predict.Random.myrandom_classfr.Y
## 1                                             699
## 2                                             126
##           Reference
## Prediction   N   Y
##          N 593 110
##          Y 106  16
##   Negative.fctr Negative.fctr.predict.Random.myrandom_classfr.N
## 1             N                                             593
## 2             Y                                             110
##   Negative.fctr.predict.Random.myrandom_classfr.Y
## 1                                             106
## 2                                              16
##           Reference
## Prediction   N   Y
##          N 593 110
##          Y 106  16
##   Negative.fctr Negative.fctr.predict.Random.myrandom_classfr.N
## 1             N                                             593
## 2             Y                                             110
##   Negative.fctr.predict.Random.myrandom_classfr.Y
## 1                                             106
## 2                                              16
##           Reference
## Prediction   N   Y
##          N 593 110
##          Y 106  16
##   Negative.fctr Negative.fctr.predict.Random.myrandom_classfr.N
## 1             N                                             593
## 2             Y                                             110
##   Negative.fctr.predict.Random.myrandom_classfr.Y
## 1                                             106
## 2                                              16
##           Reference
## Prediction   N   Y
##          N 593 110
##          Y 106  16
##   Negative.fctr Negative.fctr.predict.Random.myrandom_classfr.N
## 1             N                                             593
## 2             Y                                             110
##   Negative.fctr.predict.Random.myrandom_classfr.Y
## 1                                             106
## 2                                              16
##           Reference
## Prediction   N   Y
##          N 593 110
##          Y 106  16
##   Negative.fctr Negative.fctr.predict.Random.myrandom_classfr.N
## 1             N                                             593
## 2             Y                                             110
##   Negative.fctr.predict.Random.myrandom_classfr.Y
## 1                                             106
## 2                                              16
##           Reference
## Prediction   N   Y
##          N 593 110
##          Y 106  16
##   Negative.fctr Negative.fctr.predict.Random.myrandom_classfr.N
## 1             N                                             593
## 2             Y                                             110
##   Negative.fctr.predict.Random.myrandom_classfr.Y
## 1                                             106
## 2                                              16
##           Reference
## Prediction   N   Y
##          N 593 110
##          Y 106  16
##   Negative.fctr Negative.fctr.predict.Random.myrandom_classfr.N
## 1             N                                             593
## 2             Y                                             110
##   Negative.fctr.predict.Random.myrandom_classfr.Y
## 1                                             106
## 2                                              16
##           Reference
## Prediction   N   Y
##          N 699 126
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Random.myrandom_classfr.N
## 1             N                                             699
## 2             Y                                             126
##   Negative.fctr.predict.Random.myrandom_classfr.Y
## 1                                               0
## 2                                               0
##           Reference
## Prediction   N   Y
##          N 699 126
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Random.myrandom_classfr.N
## 1             N                                             699
## 2             Y                                             126
##   Negative.fctr.predict.Random.myrandom_classfr.Y
## 1                                               0
## 2                                               0
##    threshold   f.score
## 1        0.0 0.2649842
## 2        0.1 0.2649842
## 3        0.2 0.1290323
## 4        0.3 0.1290323
## 5        0.4 0.1290323
## 6        0.5 0.1290323
## 7        0.6 0.1290323
## 8        0.7 0.1290323
## 9        0.8 0.1290323
## 10       0.9 0.0000000
## 11       1.0 0.0000000
```

![](Apple_Tweets_files/figure-html/fit.models_0-2.png) 

```
## [1] "Classifier Probability Threshold: 0.1000 to maximize f.score.fit"
##   Negative.fctr Negative.fctr.predict.Random.myrandom_classfr.Y
## 1             N                                             699
## 2             Y                                             126
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 699 126
##   Negative.fctr Negative.fctr.predict.Random.myrandom_classfr.N
## 1             N                                               0
## 2             Y                                               0
##   Negative.fctr.predict.Random.myrandom_classfr.Y
## 1                                             699
## 2                                             126
##          Prediction
## Reference   N   Y
##         N   0 699
##         Y   0 126
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   1.527273e-01   0.000000e+00   1.288502e-01   1.791152e-01   8.472727e-01 
## AccuracyPValue  McnemarPValue 
##   1.000000e+00  1.342038e-153 
## [1] "in Random.Classifier$prob"
```

![](Apple_Tweets_files/figure-html/fit.models_0-3.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 300  56
##   Negative.fctr Negative.fctr.predict.Random.myrandom_classfr.N
## 1             N                                               0
## 2             Y                                               0
##   Negative.fctr.predict.Random.myrandom_classfr.Y
## 1                                             300
## 2                                              56
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 300  56
##   Negative.fctr Negative.fctr.predict.Random.myrandom_classfr.N
## 1             N                                               0
## 2             Y                                               0
##   Negative.fctr.predict.Random.myrandom_classfr.Y
## 1                                             300
## 2                                              56
##           Reference
## Prediction   N   Y
##          N 260  48
##          Y  40   8
##   Negative.fctr Negative.fctr.predict.Random.myrandom_classfr.N
## 1             N                                             260
## 2             Y                                              48
##   Negative.fctr.predict.Random.myrandom_classfr.Y
## 1                                              40
## 2                                               8
##           Reference
## Prediction   N   Y
##          N 260  48
##          Y  40   8
##   Negative.fctr Negative.fctr.predict.Random.myrandom_classfr.N
## 1             N                                             260
## 2             Y                                              48
##   Negative.fctr.predict.Random.myrandom_classfr.Y
## 1                                              40
## 2                                               8
##           Reference
## Prediction   N   Y
##          N 260  48
##          Y  40   8
##   Negative.fctr Negative.fctr.predict.Random.myrandom_classfr.N
## 1             N                                             260
## 2             Y                                              48
##   Negative.fctr.predict.Random.myrandom_classfr.Y
## 1                                              40
## 2                                               8
##           Reference
## Prediction   N   Y
##          N 260  48
##          Y  40   8
##   Negative.fctr Negative.fctr.predict.Random.myrandom_classfr.N
## 1             N                                             260
## 2             Y                                              48
##   Negative.fctr.predict.Random.myrandom_classfr.Y
## 1                                              40
## 2                                               8
##           Reference
## Prediction   N   Y
##          N 260  48
##          Y  40   8
##   Negative.fctr Negative.fctr.predict.Random.myrandom_classfr.N
## 1             N                                             260
## 2             Y                                              48
##   Negative.fctr.predict.Random.myrandom_classfr.Y
## 1                                              40
## 2                                               8
##           Reference
## Prediction   N   Y
##          N 260  48
##          Y  40   8
##   Negative.fctr Negative.fctr.predict.Random.myrandom_classfr.N
## 1             N                                             260
## 2             Y                                              48
##   Negative.fctr.predict.Random.myrandom_classfr.Y
## 1                                              40
## 2                                               8
##           Reference
## Prediction   N   Y
##          N 260  48
##          Y  40   8
##   Negative.fctr Negative.fctr.predict.Random.myrandom_classfr.N
## 1             N                                             260
## 2             Y                                              48
##   Negative.fctr.predict.Random.myrandom_classfr.Y
## 1                                              40
## 2                                               8
##           Reference
## Prediction   N   Y
##          N 300  56
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Random.myrandom_classfr.N
## 1             N                                             300
## 2             Y                                              56
##   Negative.fctr.predict.Random.myrandom_classfr.Y
## 1                                               0
## 2                                               0
##           Reference
## Prediction   N   Y
##          N 300  56
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Random.myrandom_classfr.N
## 1             N                                             300
## 2             Y                                              56
##   Negative.fctr.predict.Random.myrandom_classfr.Y
## 1                                               0
## 2                                               0
##    threshold   f.score
## 1        0.0 0.2718447
## 2        0.1 0.2718447
## 3        0.2 0.1538462
## 4        0.3 0.1538462
## 5        0.4 0.1538462
## 6        0.5 0.1538462
## 7        0.6 0.1538462
## 8        0.7 0.1538462
## 9        0.8 0.1538462
## 10       0.9 0.0000000
## 11       1.0 0.0000000
```

![](Apple_Tweets_files/figure-html/fit.models_0-4.png) 

```
## [1] "Classifier Probability Threshold: 0.1000 to maximize f.score.OOB"
##   Negative.fctr Negative.fctr.predict.Random.myrandom_classfr.Y
## 1             N                                             300
## 2             Y                                              56
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 300  56
##   Negative.fctr Negative.fctr.predict.Random.myrandom_classfr.N
## 1             N                                               0
## 2             Y                                               0
##   Negative.fctr.predict.Random.myrandom_classfr.Y
## 1                                             300
## 2                                              56
##          Prediction
## Reference   N   Y
##         N   0 300
##         Y   0  56
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   1.573034e-01   0.000000e+00   1.210658e-01   1.993577e-01   8.426966e-01 
## AccuracyPValue  McnemarPValue 
##   1.000000e+00   8.969796e-67 
##                  model_id     model_method  feats max.nTuningRuns
## 1 Random.myrandom_classfr myrandom_classfr .rnorm               0
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                      0.226                 0.001   0.4876695
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.1       0.2649842        0.1527273
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.1288502             0.1791152             0   0.5047619
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.1       0.2718447        0.1573034
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.1210658             0.1993577             0
```

```r
# Any models that have tuning parameters has "better" results with cross-validation
#   (except rf) & "different" results for different outcome metrics

# Max.cor.Y
#   Check impact of cv
#       rpart is not a good candidate since caret does not optimize cp (only tuning parameter of rpart) well
ret_lst <- myfit_mdl(model_id="Max.cor.Y.cv.0", 
                        model_method="rpart",
                     model_type=glb_model_type,
                        indep_vars_vctr=max_cor_y_x_var,
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df)
```

```
## [1] "fitting model: Max.cor.Y.cv.0.rpart"
## [1] "    indep_vars: freak"
```

```
## Loading required package: rpart
```

```
## Fitting cp = 0.214 on full training set
```

```
## Loading required package: rpart.plot
```

![](Apple_Tweets_files/figure-html/fit.models_0-5.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 825 
## 
##          CP nsplit rel error
## 1 0.2142857      0         1
## 
## Node number 1: 825 observations
##   predicted class=N  expected loss=0.1527273  P(node) =1
##     class counts:   699   126
##    probabilities: 0.847 0.153 
## 
## n= 825 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 825 126 N (0.8472727 0.1527273) *
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.fit"
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.rpart.N
## 1             N                                          699
## 2             Y                                          126
##           Reference
## Prediction   N   Y
##          N 699 126
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.rpart.N
## 1             N                                          699
## 2             Y                                          126
##   Negative.fctr.predict.Max.cor.Y.cv.0.rpart.Y
## 1                                            0
## 2                                            0
##          Prediction
## Reference   N   Y
##         N 699   0
##         Y 126   0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.472727e-01   0.000000e+00   8.208848e-01   8.711498e-01   8.472727e-01 
## AccuracyPValue  McnemarPValue 
##   5.237511e-01   8.390732e-29 
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.OOB"
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.rpart.N
## 1             N                                          300
## 2             Y                                           56
##           Reference
## Prediction   N   Y
##          N 300  56
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.rpart.N
## 1             N                                          300
## 2             Y                                           56
##   Negative.fctr.predict.Max.cor.Y.cv.0.rpart.Y
## 1                                            0
## 2                                            0
##          Prediction
## Reference   N   Y
##         N 300   0
##         Y  56   0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.426966e-01   0.000000e+00   8.006423e-01   8.789342e-01   8.426966e-01 
## AccuracyPValue  McnemarPValue 
##   5.355897e-01   1.986758e-13 
##               model_id model_method feats max.nTuningRuns
## 1 Max.cor.Y.cv.0.rpart        rpart freak               0
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                       0.59                 0.015         0.5
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.5               0        0.8472727
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.8208848             0.8711498             0         0.5
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.5               0        0.8426966
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.8006423             0.8789342             0
```

```r
ret_lst <- myfit_mdl(model_id="Max.cor.Y.cv.0.cp.0", 
                        model_method="rpart",
                     model_type=glb_model_type,
                        indep_vars_vctr=max_cor_y_x_var,
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                        n_cv_folds=0, 
            tune_models_df=data.frame(parameter="cp", min=0.0, max=0.0, by=0.1))
```

```
## [1] "fitting model: Max.cor.Y.cv.0.cp.0.rpart"
## [1] "    indep_vars: freak"
## Fitting cp = 0 on full training set
```

![](Apple_Tweets_files/figure-html/fit.models_0-6.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 825 
## 
##          CP nsplit rel error
## 1 0.2142857      0 1.0000000
## 2 0.0000000      1 0.7857143
## 
## Variable importance
## freak 
##   100 
## 
## Node number 1: 825 observations,    complexity param=0.2142857
##   predicted class=N  expected loss=0.1527273  P(node) =1
##     class counts:   699   126
##    probabilities: 0.847 0.153 
##   left son=2 (790 obs) right son=3 (35 obs)
##   Primary splits:
##       freak < 0.5 to the left,  improve=39.27511, (0 missing)
## 
## Node number 2: 790 observations
##   predicted class=N  expected loss=0.1202532  P(node) =0.9575758
##     class counts:   695    95
##    probabilities: 0.880 0.120 
## 
## Node number 3: 35 observations
##   predicted class=Y  expected loss=0.1142857  P(node) =0.04242424
##     class counts:     4    31
##    probabilities: 0.114 0.886 
## 
## n= 825 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 825 126 N (0.8472727 0.1527273)  
##   2) freak< 0.5 790  95 N (0.8797468 0.1202532) *
##   3) freak>=0.5 35   4 Y (0.1142857 0.8857143) *
```

![](Apple_Tweets_files/figure-html/fit.models_0-7.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 699 126
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1             N                                                 0
## 2             Y                                                 0
##   Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                               699
## 2                                               126
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 699 126
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1             N                                                 0
## 2             Y                                                 0
##   Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                               699
## 2                                               126
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1             N                                               695
## 2             Y                                                95
##   Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                 4
## 2                                                31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1             N                                               695
## 2             Y                                                95
##   Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                 4
## 2                                                31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1             N                                               695
## 2             Y                                                95
##   Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                 4
## 2                                                31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1             N                                               695
## 2             Y                                                95
##   Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                 4
## 2                                                31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1             N                                               695
## 2             Y                                                95
##   Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                 4
## 2                                                31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1             N                                               695
## 2             Y                                                95
##   Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                 4
## 2                                                31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1             N                                               695
## 2             Y                                                95
##   Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                 4
## 2                                                31
##           Reference
## Prediction   N   Y
##          N 699 126
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1             N                                               699
## 2             Y                                               126
##   Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                 0
## 2                                                 0
##           Reference
## Prediction   N   Y
##          N 699 126
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1             N                                               699
## 2             Y                                               126
##   Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                 0
## 2                                                 0
##    threshold   f.score
## 1        0.0 0.2649842
## 2        0.1 0.2649842
## 3        0.2 0.3850932
## 4        0.3 0.3850932
## 5        0.4 0.3850932
## 6        0.5 0.3850932
## 7        0.6 0.3850932
## 8        0.7 0.3850932
## 9        0.8 0.3850932
## 10       0.9 0.0000000
## 11       1.0 0.0000000
```

![](Apple_Tweets_files/figure-html/fit.models_0-8.png) 

```
## [1] "Classifier Probability Threshold: 0.8000 to maximize f.score.fit"
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1             N                                               695
## 2             Y                                                95
##   Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                 4
## 2                                                31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1             N                                               695
## 2             Y                                                95
##   Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                 4
## 2                                                31
##          Prediction
## Reference   N   Y
##         N 695   4
##         Y  95  31
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.800000e-01   3.413572e-01   8.558521e-01   9.013914e-01   8.472727e-01 
## AccuracyPValue  McnemarPValue 
##   4.212456e-03   1.491997e-19
```

![](Apple_Tweets_files/figure-html/fit.models_0-9.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 300  56
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1             N                                                 0
## 2             Y                                                 0
##   Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                               300
## 2                                                56
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 300  56
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1             N                                                 0
## 2             Y                                                 0
##   Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                               300
## 2                                                56
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1             N                                               298
## 2             Y                                                40
##   Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                 2
## 2                                                16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1             N                                               298
## 2             Y                                                40
##   Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                 2
## 2                                                16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1             N                                               298
## 2             Y                                                40
##   Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                 2
## 2                                                16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1             N                                               298
## 2             Y                                                40
##   Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                 2
## 2                                                16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1             N                                               298
## 2             Y                                                40
##   Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                 2
## 2                                                16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1             N                                               298
## 2             Y                                                40
##   Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                 2
## 2                                                16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1             N                                               298
## 2             Y                                                40
##   Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                 2
## 2                                                16
##           Reference
## Prediction   N   Y
##          N 300  56
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1             N                                               300
## 2             Y                                                56
##   Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                 0
## 2                                                 0
##           Reference
## Prediction   N   Y
##          N 300  56
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1             N                                               300
## 2             Y                                                56
##   Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                 0
## 2                                                 0
##    threshold   f.score
## 1        0.0 0.2718447
## 2        0.1 0.2718447
## 3        0.2 0.4324324
## 4        0.3 0.4324324
## 5        0.4 0.4324324
## 6        0.5 0.4324324
## 7        0.6 0.4324324
## 8        0.7 0.4324324
## 9        0.8 0.4324324
## 10       0.9 0.0000000
## 11       1.0 0.0000000
```

![](Apple_Tweets_files/figure-html/fit.models_0-10.png) 

```
## [1] "Classifier Probability Threshold: 0.8000 to maximize f.score.OOB"
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1             N                                               298
## 2             Y                                                40
##   Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                 2
## 2                                                16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1             N                                               298
## 2             Y                                                40
##   Negative.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                                 2
## 2                                                16
##          Prediction
## Reference   N   Y
##         N 298   2
##         Y  40  16
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.820225e-01   3.853995e-01   8.438883e-01   9.136353e-01   8.426966e-01 
## AccuracyPValue  McnemarPValue 
##   2.170565e-02   1.134925e-08 
##                    model_id model_method feats max.nTuningRuns
## 1 Max.cor.Y.cv.0.cp.0.rpart        rpart freak               0
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                      0.449                 0.013   0.6201546
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.8       0.3850932             0.88
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.8558521             0.9013914     0.3413572   0.6395238
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.8       0.4324324        0.8820225
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.8438883             0.9136353     0.3853995
```

```r
if (glb_is_regression || glb_is_binomial) # For multinomials this model will be run next by default
ret_lst <- myfit_mdl(model_id="Max.cor.Y", 
                        model_method="rpart",
                     model_type=glb_model_type,
                        indep_vars_vctr=max_cor_y_x_var,
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                        n_cv_folds=glb_n_cv_folds, tune_models_df=NULL)
```

```
## [1] "fitting model: Max.cor.Y.rpart"
## [1] "    indep_vars: freak"
## + Fold1: cp=0 
## - Fold1: cp=0 
## + Fold2: cp=0 
## - Fold2: cp=0 
## + Fold3: cp=0 
## - Fold3: cp=0 
## Aggregating results
## Selecting tuning parameters
## Fitting cp = 0.107 on full training set
```

![](Apple_Tweets_files/figure-html/fit.models_0-11.png) ![](Apple_Tweets_files/figure-html/fit.models_0-12.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 825 
## 
##          CP nsplit rel error
## 1 0.2142857      0 1.0000000
## 2 0.0000000      1 0.7857143
## 
## Variable importance
## freak 
##   100 
## 
## Node number 1: 825 observations,    complexity param=0.2142857
##   predicted class=N  expected loss=0.1527273  P(node) =1
##     class counts:   699   126
##    probabilities: 0.847 0.153 
##   left son=2 (790 obs) right son=3 (35 obs)
##   Primary splits:
##       freak < 0.5 to the left,  improve=39.27511, (0 missing)
## 
## Node number 2: 790 observations
##   predicted class=N  expected loss=0.1202532  P(node) =0.9575758
##     class counts:   695    95
##    probabilities: 0.880 0.120 
## 
## Node number 3: 35 observations
##   predicted class=Y  expected loss=0.1142857  P(node) =0.04242424
##     class counts:     4    31
##    probabilities: 0.114 0.886 
## 
## n= 825 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 825 126 N (0.8472727 0.1527273)  
##   2) freak< 0.5 790  95 N (0.8797468 0.1202532) *
##   3) freak>=0.5 35   4 Y (0.1142857 0.8857143) *
```

![](Apple_Tweets_files/figure-html/fit.models_0-13.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 699 126
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.rpart.N
## 1             N                                       0
## 2             Y                                       0
##   Negative.fctr.predict.Max.cor.Y.rpart.Y
## 1                                     699
## 2                                     126
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 699 126
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.rpart.N
## 1             N                                       0
## 2             Y                                       0
##   Negative.fctr.predict.Max.cor.Y.rpart.Y
## 1                                     699
## 2                                     126
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.rpart.N
## 1             N                                     695
## 2             Y                                      95
##   Negative.fctr.predict.Max.cor.Y.rpart.Y
## 1                                       4
## 2                                      31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.rpart.N
## 1             N                                     695
## 2             Y                                      95
##   Negative.fctr.predict.Max.cor.Y.rpart.Y
## 1                                       4
## 2                                      31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.rpart.N
## 1             N                                     695
## 2             Y                                      95
##   Negative.fctr.predict.Max.cor.Y.rpart.Y
## 1                                       4
## 2                                      31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.rpart.N
## 1             N                                     695
## 2             Y                                      95
##   Negative.fctr.predict.Max.cor.Y.rpart.Y
## 1                                       4
## 2                                      31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.rpart.N
## 1             N                                     695
## 2             Y                                      95
##   Negative.fctr.predict.Max.cor.Y.rpart.Y
## 1                                       4
## 2                                      31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.rpart.N
## 1             N                                     695
## 2             Y                                      95
##   Negative.fctr.predict.Max.cor.Y.rpart.Y
## 1                                       4
## 2                                      31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.rpart.N
## 1             N                                     695
## 2             Y                                      95
##   Negative.fctr.predict.Max.cor.Y.rpart.Y
## 1                                       4
## 2                                      31
##           Reference
## Prediction   N   Y
##          N 699 126
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.rpart.N
## 1             N                                     699
## 2             Y                                     126
##   Negative.fctr.predict.Max.cor.Y.rpart.Y
## 1                                       0
## 2                                       0
##           Reference
## Prediction   N   Y
##          N 699 126
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.rpart.N
## 1             N                                     699
## 2             Y                                     126
##   Negative.fctr.predict.Max.cor.Y.rpart.Y
## 1                                       0
## 2                                       0
##    threshold   f.score
## 1        0.0 0.2649842
## 2        0.1 0.2649842
## 3        0.2 0.3850932
## 4        0.3 0.3850932
## 5        0.4 0.3850932
## 6        0.5 0.3850932
## 7        0.6 0.3850932
## 8        0.7 0.3850932
## 9        0.8 0.3850932
## 10       0.9 0.0000000
## 11       1.0 0.0000000
```

![](Apple_Tweets_files/figure-html/fit.models_0-14.png) 

```
## [1] "Classifier Probability Threshold: 0.8000 to maximize f.score.fit"
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.rpart.N
## 1             N                                     695
## 2             Y                                      95
##   Negative.fctr.predict.Max.cor.Y.rpart.Y
## 1                                       4
## 2                                      31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.rpart.N
## 1             N                                     695
## 2             Y                                      95
##   Negative.fctr.predict.Max.cor.Y.rpart.Y
## 1                                       4
## 2                                      31
##          Prediction
## Reference   N   Y
##         N 695   4
##         Y  95  31
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.800000e-01   3.413572e-01   8.558521e-01   9.013914e-01   8.472727e-01 
## AccuracyPValue  McnemarPValue 
##   4.212456e-03   1.491997e-19
```

![](Apple_Tweets_files/figure-html/fit.models_0-15.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 300  56
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.rpart.N
## 1             N                                       0
## 2             Y                                       0
##   Negative.fctr.predict.Max.cor.Y.rpart.Y
## 1                                     300
## 2                                      56
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 300  56
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.rpart.N
## 1             N                                       0
## 2             Y                                       0
##   Negative.fctr.predict.Max.cor.Y.rpart.Y
## 1                                     300
## 2                                      56
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.rpart.N
## 1             N                                     298
## 2             Y                                      40
##   Negative.fctr.predict.Max.cor.Y.rpart.Y
## 1                                       2
## 2                                      16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.rpart.N
## 1             N                                     298
## 2             Y                                      40
##   Negative.fctr.predict.Max.cor.Y.rpart.Y
## 1                                       2
## 2                                      16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.rpart.N
## 1             N                                     298
## 2             Y                                      40
##   Negative.fctr.predict.Max.cor.Y.rpart.Y
## 1                                       2
## 2                                      16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.rpart.N
## 1             N                                     298
## 2             Y                                      40
##   Negative.fctr.predict.Max.cor.Y.rpart.Y
## 1                                       2
## 2                                      16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.rpart.N
## 1             N                                     298
## 2             Y                                      40
##   Negative.fctr.predict.Max.cor.Y.rpart.Y
## 1                                       2
## 2                                      16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.rpart.N
## 1             N                                     298
## 2             Y                                      40
##   Negative.fctr.predict.Max.cor.Y.rpart.Y
## 1                                       2
## 2                                      16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.rpart.N
## 1             N                                     298
## 2             Y                                      40
##   Negative.fctr.predict.Max.cor.Y.rpart.Y
## 1                                       2
## 2                                      16
##           Reference
## Prediction   N   Y
##          N 300  56
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.rpart.N
## 1             N                                     300
## 2             Y                                      56
##   Negative.fctr.predict.Max.cor.Y.rpart.Y
## 1                                       0
## 2                                       0
##           Reference
## Prediction   N   Y
##          N 300  56
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.rpart.N
## 1             N                                     300
## 2             Y                                      56
##   Negative.fctr.predict.Max.cor.Y.rpart.Y
## 1                                       0
## 2                                       0
##    threshold   f.score
## 1        0.0 0.2718447
## 2        0.1 0.2718447
## 3        0.2 0.4324324
## 4        0.3 0.4324324
## 5        0.4 0.4324324
## 6        0.5 0.4324324
## 7        0.6 0.4324324
## 8        0.7 0.4324324
## 9        0.8 0.4324324
## 10       0.9 0.0000000
## 11       1.0 0.0000000
```

![](Apple_Tweets_files/figure-html/fit.models_0-16.png) 

```
## [1] "Classifier Probability Threshold: 0.8000 to maximize f.score.OOB"
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.rpart.N
## 1             N                                     298
## 2             Y                                      40
##   Negative.fctr.predict.Max.cor.Y.rpart.Y
## 1                                       2
## 2                                      16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.rpart.N
## 1             N                                     298
## 2             Y                                      40
##   Negative.fctr.predict.Max.cor.Y.rpart.Y
## 1                                       2
## 2                                      16
##          Prediction
## Reference   N   Y
##         N 298   2
##         Y  40  16
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.820225e-01   3.853995e-01   8.438883e-01   9.136353e-01   8.426966e-01 
## AccuracyPValue  McnemarPValue 
##   2.170565e-02   1.134925e-08 
##          model_id model_method feats max.nTuningRuns
## 1 Max.cor.Y.rpart        rpart freak               3
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                      0.956                 0.013   0.6201546
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.8       0.3850932             0.88
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.8558521             0.9013914     0.3378153   0.6395238
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.8       0.4324324        0.8820225
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.8438883             0.9136353     0.3853995
##   max.AccuracySD.fit max.KappaSD.fit
## 1         0.01585054       0.1193248
```

```r
# Used to compare vs. Interactions.High.cor.Y 
ret_lst <- myfit_mdl(model_id="Max.cor.Y", 
                        model_method=ifelse(glb_is_regression, "lm", 
                                        ifelse(glb_is_binomial, "glm", "rpart")),
                     model_type=glb_model_type,
                        indep_vars_vctr=max_cor_y_x_var,
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                        n_cv_folds=glb_n_cv_folds, tune_models_df=NULL)
```

```
## [1] "fitting model: Max.cor.Y.glm"
## [1] "    indep_vars: freak"
## + Fold1: parameter=none 
## - Fold1: parameter=none 
## + Fold2: parameter=none 
## - Fold2: parameter=none 
## + Fold3: parameter=none 
## - Fold3: parameter=none 
## Aggregating results
## Fitting final model on full training set
```

![](Apple_Tweets_files/figure-html/fit.models_0-17.png) ![](Apple_Tweets_files/figure-html/fit.models_0-18.png) ![](Apple_Tweets_files/figure-html/fit.models_0-19.png) ![](Apple_Tweets_files/figure-html/fit.models_0-20.png) 

```
## 
## Call:
## NULL
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -2.0556  -0.5062  -0.5062  -0.5062   2.0583  
## 
## Coefficients:
##             Estimate Std. Error z value Pr(>|z|)    
## (Intercept)  -1.9901     0.1094 -18.194  < 2e-16 ***
## freak         3.9740     0.5435   7.312 2.62e-13 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 705.23  on 824  degrees of freedom
## Residual deviance: 604.92  on 823  degrees of freedom
## AIC: 608.92
## 
## Number of Fisher Scoring iterations: 5
```

![](Apple_Tweets_files/figure-html/fit.models_0-21.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 699 126
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.glm.N
## 1             N                                     0
## 2             Y                                     0
##   Negative.fctr.predict.Max.cor.Y.glm.Y
## 1                                   699
## 2                                   126
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 699 126
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.glm.N
## 1             N                                     0
## 2             Y                                     0
##   Negative.fctr.predict.Max.cor.Y.glm.Y
## 1                                   699
## 2                                   126
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.glm.N
## 1             N                                   695
## 2             Y                                    95
##   Negative.fctr.predict.Max.cor.Y.glm.Y
## 1                                     4
## 2                                    31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.glm.N
## 1             N                                   695
## 2             Y                                    95
##   Negative.fctr.predict.Max.cor.Y.glm.Y
## 1                                     4
## 2                                    31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.glm.N
## 1             N                                   695
## 2             Y                                    95
##   Negative.fctr.predict.Max.cor.Y.glm.Y
## 1                                     4
## 2                                    31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.glm.N
## 1             N                                   695
## 2             Y                                    95
##   Negative.fctr.predict.Max.cor.Y.glm.Y
## 1                                     4
## 2                                    31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.glm.N
## 1             N                                   695
## 2             Y                                    95
##   Negative.fctr.predict.Max.cor.Y.glm.Y
## 1                                     4
## 2                                    31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.glm.N
## 1             N                                   695
## 2             Y                                    95
##   Negative.fctr.predict.Max.cor.Y.glm.Y
## 1                                     4
## 2                                    31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.glm.N
## 1             N                                   695
## 2             Y                                    95
##   Negative.fctr.predict.Max.cor.Y.glm.Y
## 1                                     4
## 2                                    31
##           Reference
## Prediction   N   Y
##          N 699 124
##          Y   0   2
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.glm.N
## 1             N                                   699
## 2             Y                                   124
##   Negative.fctr.predict.Max.cor.Y.glm.Y
## 1                                     0
## 2                                     2
##           Reference
## Prediction   N   Y
##          N 699 126
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.glm.N
## 1             N                                   699
## 2             Y                                   126
##   Negative.fctr.predict.Max.cor.Y.glm.Y
## 1                                     0
## 2                                     0
##    threshold   f.score
## 1        0.0 0.2649842
## 2        0.1 0.2649842
## 3        0.2 0.3850932
## 4        0.3 0.3850932
## 5        0.4 0.3850932
## 6        0.5 0.3850932
## 7        0.6 0.3850932
## 8        0.7 0.3850932
## 9        0.8 0.3850932
## 10       0.9 0.0312500
## 11       1.0 0.0000000
```

![](Apple_Tweets_files/figure-html/fit.models_0-22.png) 

```
## [1] "Classifier Probability Threshold: 0.8000 to maximize f.score.fit"
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.glm.N
## 1             N                                   695
## 2             Y                                    95
##   Negative.fctr.predict.Max.cor.Y.glm.Y
## 1                                     4
## 2                                    31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.glm.N
## 1             N                                   695
## 2             Y                                    95
##   Negative.fctr.predict.Max.cor.Y.glm.Y
## 1                                     4
## 2                                    31
##          Prediction
## Reference   N   Y
##         N 695   4
##         Y  95  31
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.800000e-01   3.413572e-01   8.558521e-01   9.013914e-01   8.472727e-01 
## AccuracyPValue  McnemarPValue 
##   4.212456e-03   1.491997e-19
```

![](Apple_Tweets_files/figure-html/fit.models_0-23.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 300  56
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.glm.N
## 1             N                                     0
## 2             Y                                     0
##   Negative.fctr.predict.Max.cor.Y.glm.Y
## 1                                   300
## 2                                    56
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 300  56
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.glm.N
## 1             N                                     0
## 2             Y                                     0
##   Negative.fctr.predict.Max.cor.Y.glm.Y
## 1                                   300
## 2                                    56
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.glm.N
## 1             N                                   298
## 2             Y                                    40
##   Negative.fctr.predict.Max.cor.Y.glm.Y
## 1                                     2
## 2                                    16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.glm.N
## 1             N                                   298
## 2             Y                                    40
##   Negative.fctr.predict.Max.cor.Y.glm.Y
## 1                                     2
## 2                                    16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.glm.N
## 1             N                                   298
## 2             Y                                    40
##   Negative.fctr.predict.Max.cor.Y.glm.Y
## 1                                     2
## 2                                    16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.glm.N
## 1             N                                   298
## 2             Y                                    40
##   Negative.fctr.predict.Max.cor.Y.glm.Y
## 1                                     2
## 2                                    16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.glm.N
## 1             N                                   298
## 2             Y                                    40
##   Negative.fctr.predict.Max.cor.Y.glm.Y
## 1                                     2
## 2                                    16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.glm.N
## 1             N                                   298
## 2             Y                                    40
##   Negative.fctr.predict.Max.cor.Y.glm.Y
## 1                                     2
## 2                                    16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.glm.N
## 1             N                                   298
## 2             Y                                    40
##   Negative.fctr.predict.Max.cor.Y.glm.Y
## 1                                     2
## 2                                    16
##           Reference
## Prediction   N   Y
##          N 300  55
##          Y   0   1
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.glm.N
## 1             N                                   300
## 2             Y                                    55
##   Negative.fctr.predict.Max.cor.Y.glm.Y
## 1                                     0
## 2                                     1
##           Reference
## Prediction   N   Y
##          N 300  56
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.glm.N
## 1             N                                   300
## 2             Y                                    56
##   Negative.fctr.predict.Max.cor.Y.glm.Y
## 1                                     0
## 2                                     0
##    threshold    f.score
## 1        0.0 0.27184466
## 2        0.1 0.27184466
## 3        0.2 0.43243243
## 4        0.3 0.43243243
## 5        0.4 0.43243243
## 6        0.5 0.43243243
## 7        0.6 0.43243243
## 8        0.7 0.43243243
## 9        0.8 0.43243243
## 10       0.9 0.03508772
## 11       1.0 0.00000000
```

![](Apple_Tweets_files/figure-html/fit.models_0-24.png) 

```
## [1] "Classifier Probability Threshold: 0.8000 to maximize f.score.OOB"
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.glm.N
## 1             N                                   298
## 2             Y                                    40
##   Negative.fctr.predict.Max.cor.Y.glm.Y
## 1                                     2
## 2                                    16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Max.cor.Y.glm.N
## 1             N                                   298
## 2             Y                                    40
##   Negative.fctr.predict.Max.cor.Y.glm.Y
## 1                                     2
## 2                                    16
##          Prediction
## Reference   N   Y
##         N 298   2
##         Y  40  16
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.820225e-01   3.853995e-01   8.438883e-01   9.136353e-01   8.426966e-01 
## AccuracyPValue  McnemarPValue 
##   2.170565e-02   1.134925e-08 
##        model_id model_method feats max.nTuningRuns
## 1 Max.cor.Y.glm          glm freak               1
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                      1.091                 0.015   0.6202001
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.8       0.3850932             0.88
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.8558521             0.9013914     0.3378153   0.6395833
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.8       0.4324324        0.8820225
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB min.aic.fit
## 1             0.8438883             0.9136353     0.3853995    608.9243
##   max.AccuracySD.fit max.KappaSD.fit
## 1         0.01585054       0.1193248
```

```r
# Interactions.High.cor.Y
if (length(int_feats <- setdiff(unique(glb_feats_df$cor.high.X), NA)) > 0) {
# if (nrow(int_feats_df <- subset(glb_feats_df, !is.na(cor.high.X) & 
#                                               (exclude.as.feat == 0))) > 0) {
    # lm & glm handle interaction terms; rpart & rf do not
    #   This does not work - why ???
#     indep_vars_vctr <- ifelse(glb_is_binomial, 
#         c(max_cor_y_x_var, paste(max_cor_y_x_var, 
#                         subset(glb_feats_df, is.na(cor.low))[, "id"], sep=":")),
#         union(max_cor_y_x_var, subset(glb_feats_df, is.na(cor.low))[, "id"]))
    if (glb_is_regression || glb_is_binomial) {
        indep_vars_vctr <- 
            c(max_cor_y_x_var, paste(max_cor_y_x_var, int_feats, sep=":"))       
    } else { indep_vars_vctr <- union(max_cor_y_x_var, int_feats) }
    
    ret_lst <- myfit_mdl(model_id="Interact.High.cor.y", 
                            model_method=ifelse(glb_is_regression, "lm", 
                                        ifelse(glb_is_binomial, "glm", "rpart")),
                         model_type=glb_model_type,
                            indep_vars_vctr,
                            glb_rsp_var, glb_rsp_var_out,
                            fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                            n_cv_folds=glb_n_cv_folds, tune_models_df=NULL)                        
}    
```

```
## [1] "fitting model: Interact.High.cor.y.glm"
## [1] "    indep_vars: freak, freak:cdp, freak:httpbitly18xc8dk, freak:femal, freak:refus, freak:emiss, freak:itun"
## + Fold1: parameter=none
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold1: parameter=none 
## + Fold2: parameter=none
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold2: parameter=none 
## + Fold3: parameter=none
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold3: parameter=none 
## Aggregating results
## Fitting final model on full training set
```

![](Apple_Tweets_files/figure-html/fit.models_0-25.png) ![](Apple_Tweets_files/figure-html/fit.models_0-26.png) ![](Apple_Tweets_files/figure-html/fit.models_0-27.png) 

```
## 
## Call:
## NULL
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -2.0252  -0.5062  -0.5062  -0.5062   2.0583  
## 
## Coefficients: (5 not defined because of singularities)
##                          Estimate Std. Error z value Pr(>|z|)    
## (Intercept)               -1.9901     0.1094 -18.194  < 2e-16 ***
## freak                      3.9031     0.5456   7.153 8.46e-13 ***
## `freak:cdp`                    NA         NA      NA       NA    
## `freak:httpbitly18xc8dk`       NA         NA      NA       NA    
## `freak:femal`                  NA         NA      NA       NA    
## `freak:refus`                  NA         NA      NA       NA    
## `freak:emiss`                  NA         NA      NA       NA    
## `freak:itun`              12.6531   624.1941   0.020    0.984    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 705.23  on 824  degrees of freedom
## Residual deviance: 604.39  on 822  degrees of freedom
## AIC: 610.39
## 
## Number of Fisher Scoring iterations: 13
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

![](Apple_Tweets_files/figure-html/fit.models_0-28.png) ![](Apple_Tweets_files/figure-html/fit.models_0-29.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 699 126
##   Negative.fctr Negative.fctr.predict.Interact.High.cor.y.glm.N
## 1             N                                               0
## 2             Y                                               0
##   Negative.fctr.predict.Interact.High.cor.y.glm.Y
## 1                                             699
## 2                                             126
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 699 126
##   Negative.fctr Negative.fctr.predict.Interact.High.cor.y.glm.N
## 1             N                                               0
## 2             Y                                               0
##   Negative.fctr.predict.Interact.High.cor.y.glm.Y
## 1                                             699
## 2                                             126
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Interact.High.cor.y.glm.N
## 1             N                                             695
## 2             Y                                              95
##   Negative.fctr.predict.Interact.High.cor.y.glm.Y
## 1                                               4
## 2                                              31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Interact.High.cor.y.glm.N
## 1             N                                             695
## 2             Y                                              95
##   Negative.fctr.predict.Interact.High.cor.y.glm.Y
## 1                                               4
## 2                                              31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Interact.High.cor.y.glm.N
## 1             N                                             695
## 2             Y                                              95
##   Negative.fctr.predict.Interact.High.cor.y.glm.Y
## 1                                               4
## 2                                              31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Interact.High.cor.y.glm.N
## 1             N                                             695
## 2             Y                                              95
##   Negative.fctr.predict.Interact.High.cor.y.glm.Y
## 1                                               4
## 2                                              31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Interact.High.cor.y.glm.N
## 1             N                                             695
## 2             Y                                              95
##   Negative.fctr.predict.Interact.High.cor.y.glm.Y
## 1                                               4
## 2                                              31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Interact.High.cor.y.glm.N
## 1             N                                             695
## 2             Y                                              95
##   Negative.fctr.predict.Interact.High.cor.y.glm.Y
## 1                                               4
## 2                                              31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Interact.High.cor.y.glm.N
## 1             N                                             695
## 2             Y                                              95
##   Negative.fctr.predict.Interact.High.cor.y.glm.Y
## 1                                               4
## 2                                              31
##           Reference
## Prediction   N   Y
##          N 699 122
##          Y   0   4
##   Negative.fctr Negative.fctr.predict.Interact.High.cor.y.glm.N
## 1             N                                             699
## 2             Y                                             122
##   Negative.fctr.predict.Interact.High.cor.y.glm.Y
## 1                                               0
## 2                                               4
##           Reference
## Prediction   N   Y
##          N 699 126
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Interact.High.cor.y.glm.N
## 1             N                                             699
## 2             Y                                             126
##   Negative.fctr.predict.Interact.High.cor.y.glm.Y
## 1                                               0
## 2                                               0
##    threshold    f.score
## 1        0.0 0.26498423
## 2        0.1 0.26498423
## 3        0.2 0.38509317
## 4        0.3 0.38509317
## 5        0.4 0.38509317
## 6        0.5 0.38509317
## 7        0.6 0.38509317
## 8        0.7 0.38509317
## 9        0.8 0.38509317
## 10       0.9 0.06153846
## 11       1.0 0.00000000
```

```
## [1] "Classifier Probability Threshold: 0.8000 to maximize f.score.fit"
##   Negative.fctr Negative.fctr.predict.Interact.High.cor.y.glm.N
## 1             N                                             695
## 2             Y                                              95
##   Negative.fctr.predict.Interact.High.cor.y.glm.Y
## 1                                               4
## 2                                              31
##           Reference
## Prediction   N   Y
##          N 695  95
##          Y   4  31
##   Negative.fctr Negative.fctr.predict.Interact.High.cor.y.glm.N
## 1             N                                             695
## 2             Y                                              95
##   Negative.fctr.predict.Interact.High.cor.y.glm.Y
## 1                                               4
## 2                                              31
##          Prediction
## Reference   N   Y
##         N 695   4
##         Y  95  31
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.800000e-01   3.413572e-01   8.558521e-01   9.013914e-01   8.472727e-01 
## AccuracyPValue  McnemarPValue 
##   4.212456e-03   1.491997e-19
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

![](Apple_Tweets_files/figure-html/fit.models_0-30.png) ![](Apple_Tweets_files/figure-html/fit.models_0-31.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 300  56
##   Negative.fctr Negative.fctr.predict.Interact.High.cor.y.glm.N
## 1             N                                               0
## 2             Y                                               0
##   Negative.fctr.predict.Interact.High.cor.y.glm.Y
## 1                                             300
## 2                                              56
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 300  56
##   Negative.fctr Negative.fctr.predict.Interact.High.cor.y.glm.N
## 1             N                                               0
## 2             Y                                               0
##   Negative.fctr.predict.Interact.High.cor.y.glm.Y
## 1                                             300
## 2                                              56
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Interact.High.cor.y.glm.N
## 1             N                                             298
## 2             Y                                              40
##   Negative.fctr.predict.Interact.High.cor.y.glm.Y
## 1                                               2
## 2                                              16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Interact.High.cor.y.glm.N
## 1             N                                             298
## 2             Y                                              40
##   Negative.fctr.predict.Interact.High.cor.y.glm.Y
## 1                                               2
## 2                                              16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Interact.High.cor.y.glm.N
## 1             N                                             298
## 2             Y                                              40
##   Negative.fctr.predict.Interact.High.cor.y.glm.Y
## 1                                               2
## 2                                              16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Interact.High.cor.y.glm.N
## 1             N                                             298
## 2             Y                                              40
##   Negative.fctr.predict.Interact.High.cor.y.glm.Y
## 1                                               2
## 2                                              16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Interact.High.cor.y.glm.N
## 1             N                                             298
## 2             Y                                              40
##   Negative.fctr.predict.Interact.High.cor.y.glm.Y
## 1                                               2
## 2                                              16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Interact.High.cor.y.glm.N
## 1             N                                             298
## 2             Y                                              40
##   Negative.fctr.predict.Interact.High.cor.y.glm.Y
## 1                                               2
## 2                                              16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Interact.High.cor.y.glm.N
## 1             N                                             298
## 2             Y                                              40
##   Negative.fctr.predict.Interact.High.cor.y.glm.Y
## 1                                               2
## 2                                              16
##           Reference
## Prediction   N   Y
##          N 300  54
##          Y   0   2
##   Negative.fctr Negative.fctr.predict.Interact.High.cor.y.glm.N
## 1             N                                             300
## 2             Y                                              54
##   Negative.fctr.predict.Interact.High.cor.y.glm.Y
## 1                                               0
## 2                                               2
##           Reference
## Prediction   N   Y
##          N 300  56
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Interact.High.cor.y.glm.N
## 1             N                                             300
## 2             Y                                              56
##   Negative.fctr.predict.Interact.High.cor.y.glm.Y
## 1                                               0
## 2                                               0
##    threshold    f.score
## 1        0.0 0.27184466
## 2        0.1 0.27184466
## 3        0.2 0.43243243
## 4        0.3 0.43243243
## 5        0.4 0.43243243
## 6        0.5 0.43243243
## 7        0.6 0.43243243
## 8        0.7 0.43243243
## 9        0.8 0.43243243
## 10       0.9 0.06896552
## 11       1.0 0.00000000
```

![](Apple_Tweets_files/figure-html/fit.models_0-32.png) 

```
## [1] "Classifier Probability Threshold: 0.8000 to maximize f.score.OOB"
##   Negative.fctr Negative.fctr.predict.Interact.High.cor.y.glm.N
## 1             N                                             298
## 2             Y                                              40
##   Negative.fctr.predict.Interact.High.cor.y.glm.Y
## 1                                               2
## 2                                              16
##           Reference
## Prediction   N   Y
##          N 298  40
##          Y   2  16
##   Negative.fctr Negative.fctr.predict.Interact.High.cor.y.glm.N
## 1             N                                             298
## 2             Y                                              40
##   Negative.fctr.predict.Interact.High.cor.y.glm.Y
## 1                                               2
## 2                                              16
##          Prediction
## Reference   N   Y
##         N 298   2
##         Y  40  16
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.820225e-01   3.853995e-01   8.438883e-01   9.136353e-01   8.426966e-01 
## AccuracyPValue  McnemarPValue 
##   2.170565e-02   1.134925e-08 
##                  model_id model_method
## 1 Interact.High.cor.y.glm          glm
##                                                                                         feats
## 1 freak, freak:cdp, freak:httpbitly18xc8dk, freak:femal, freak:refus, freak:emiss, freak:itun
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               1                      0.928                 0.033
##   max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1   0.6202455                    0.8       0.3850932             0.88
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.8558521             0.9013914     0.3378153   0.6396429
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.8       0.4324324        0.8820225
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB min.aic.fit
## 1             0.8438883             0.9136353     0.3853995    610.3916
##   max.AccuracySD.fit max.KappaSD.fit
## 1         0.01585054       0.1193248
```

```r
# Low.cor.X
if (glb_is_classification && glb_is_binomial)
    indep_vars_vctr <- subset(glb_feats_df, is.na(cor.high.X) & 
                                            is.ConditionalX.y & 
                                            (exclude.as.feat != 1))[, "id"] else
    indep_vars_vctr <- subset(glb_feats_df, is.na(cor.high.X) & 
                                            (exclude.as.feat != 1))[, "id"]                                                
ret_lst <- myfit_mdl(model_id="Low.cor.X", 
                        model_method=ifelse(glb_is_regression, "lm", 
                                        ifelse(glb_is_binomial, "glm", "rpart")),
                        indep_vars_vctr=indep_vars_vctr,
                        model_type=glb_model_type,                     
                        glb_rsp_var, glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                        n_cv_folds=glb_n_cv_folds, tune_models_df=NULL)
```

```
## [1] "fitting model: Low.cor.X.glm"
## [1] "    indep_vars: freak, hate, stuff, pictur, wtf, cant, shame, stupid, even, line, yooo, better, ever, fix, charger, still, charg, disappoint, short, like, yall, break., imessag, stand, togeth, cheap, wont, make, darn, httpbitly18xc8dk, dear, facebook, X7evenstarz, femal, money, theyr, batteri, china, turn, hope, life, sinc, steve, switch, everi, last, your, amaz, arent, date, ill, ive, lost, noth, worst, phone, care, way, year, updat, app, iphon, data, take, card, custom, die, event, problem, two, use, screen, thing, buy, tri, cdp, doesnt, emiss, feel, fun, got, hour, macbook, miss, siri, start, upgrad, get, think, copi, guess, person, smart, best, differ, nsa, product, twitter, old, see, download, idea, job, mani, simpl, soon, team, technolog, wow, much, support, text, time, android, that, fingerprint, now, ask, colour, gonna, happen, man, give, hey, today, case, chang, one, work, day, nokia, preorder, look, awesom, done, featur, said, secur, seem, smartphon, thought, what, windowsphon, bit, realli, can, sure, dont, impress, call, didnt, drop, fail, first, improv, isnt, made, mean, put, yet, next., thank, market, blackberri, mac, never, plastic, right, sell, pleas, guy, scanner, .rnorm, want, black, free, innov, via, come, good, let, stop, color, compani, know, price, just, store, ios7, peopl, releas, say, anyon, devic, need, back, googl, itun, microsoft, will, appl, new, iphone5"
## + Fold1: parameter=none
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold1: parameter=none 
## + Fold2: parameter=none
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
## - Fold2: parameter=none 
## + Fold3: parameter=none
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold3: parameter=none 
## Aggregating results
## Fitting final model on full training set
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

![](Apple_Tweets_files/figure-html/fit.models_0-33.png) ![](Apple_Tweets_files/figure-html/fit.models_0-34.png) ![](Apple_Tweets_files/figure-html/fit.models_0-35.png) 

```
## Warning in sqrt(crit * p * (1 - hh)/hh): NaNs produced
```

```
## Warning in sqrt(crit * p * (1 - hh)/hh): NaNs produced
```

![](Apple_Tweets_files/figure-html/fit.models_0-36.png) 

```
## 
## Call:
## NULL
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -2.3085  -0.1898  -0.0527  -0.0023   3.7379  
## 
## Coefficients:
##                    Estimate Std. Error z value Pr(>|z|)    
## (Intercept)         -6.4579     0.7889  -8.186  2.7e-16 ***
## freak                9.1809     1.3886   6.612  3.8e-11 ***
## hate                10.4590     3.0120   3.472 0.000516 ***
## stuff               -2.0019     2.1442  -0.934 0.350497    
## pictur               4.1991     2.8863   1.455 0.145716    
## wtf                  7.1805     2.8868   2.487 0.012868 *  
## cant                 4.0425     1.3180   3.067 0.002161 ** 
## shame               66.7249  2466.3610   0.027 0.978417    
## stupid              11.5297     8.4770   1.360 0.173795    
## even                 6.3322     3.2007   1.978 0.047887 *  
## line                 1.9824    13.8581   0.143 0.886252    
## yooo                 3.2529     3.7166   0.875 0.381448    
## better               2.6315     1.3363   1.969 0.048923 *  
## ever                 3.3362     4.6128   0.723 0.469523    
## fix                  4.9017     2.1968   2.231 0.025665 *  
## charger              5.0938     1.9345   2.633 0.008460 ** 
## still                1.9025     1.8135   1.049 0.294147    
## charg                0.6878     7.2813   0.094 0.924739    
## disappoint          13.8308     4.6660   2.964 0.003035 ** 
## short                3.4861     4.1900   0.832 0.405410    
## like                 3.7224     1.2323   3.021 0.002523 ** 
## yall                 1.1669     4.1452   0.281 0.778328    
## break.               6.4745     2.3095   2.803 0.005057 ** 
## imessag              6.6843     3.6420   1.835 0.066452 .  
## stand                4.3340     5.4653   0.793 0.427771    
## togeth               7.0968     4.1727   1.701 0.088984 .  
## cheap                2.3657     3.3719   0.702 0.482934    
## wont                -3.2771     2.2019  -1.488 0.136681    
## make                -1.7010     1.2881  -1.321 0.186634    
## darn                 5.5660     1.6493   3.375 0.000739 ***
## httpbitly18xc8dk    60.7538  2466.3732   0.025 0.980348    
## dear                 7.5988     3.1216   2.434 0.014921 *  
## facebook           -55.5605  2466.3738  -0.023 0.982027    
## X7evenstarz          3.8727    10.2669   0.377 0.706020    
## femal                4.9297     2.8880   1.707 0.087831 .  
## money                3.3123     3.4895   0.949 0.342501    
## theyr                9.8591     3.5911   2.745 0.006044 ** 
## batteri              3.0769     2.7941   1.101 0.270813    
## china                5.3347     1.6584   3.217 0.001296 ** 
## turn                -3.0405     3.8535  -0.789 0.430106    
## hope                 6.1148     3.4096   1.793 0.072909 .  
## life                 6.3707     1.8034   3.533 0.000411 ***
## sinc                 3.2532     3.6091   0.901 0.367391    
## steve               -1.3674     3.5482  -0.385 0.699953    
## switch               1.1880     3.7146   0.320 0.749108    
## everi                4.7127     2.1334   2.209 0.027176 *  
## last               -21.9212 48676.8709   0.000 0.999641    
## your                 3.3579     2.1863   1.536 0.124569    
## amaz                -0.2129     5.7301  -0.037 0.970365    
## arent                2.2988     3.9178   0.587 0.557368    
## date                -2.0176     6.2848  -0.321 0.748184    
## ill                 13.2760    56.8806   0.233 0.815450    
## ive                 -2.0912     5.0578  -0.413 0.679271    
## lost                 2.7784     4.1408   0.671 0.502240    
## noth                 3.4509     5.5405   0.623 0.533388    
## worst               -0.8178     2.6712  -0.306 0.759477    
## phone               -0.2324     1.3605  -0.171 0.864341    
## care                 4.8121     2.6156   1.840 0.065803 .  
## way                 -1.7727     2.3345  -0.759 0.447659    
## year                 4.0920     2.6147   1.565 0.117591    
## updat               -0.6827     2.2880  -0.298 0.765392    
## app                  2.2916     1.4806   1.548 0.121680    
## iphon                1.2711     0.7686   1.654 0.098178 .  
## data                 5.1364     1.8282   2.810 0.004961 ** 
## take                 2.9568     2.0412   1.449 0.147457    
## card                 1.2747     2.1017   0.607 0.544180    
## custom              -1.7334     5.3621  -0.323 0.746490    
## die                 -0.9854     4.1027  -0.240 0.810188    
## event                2.0671     3.1784   0.650 0.515457    
## problem              6.4260     4.4629   1.440 0.149907    
## two                  7.2052     4.5104   1.597 0.110159    
## use                  1.4373     1.5047   0.955 0.339459    
## screen               2.8497     2.7874   1.022 0.306605    
## thing                3.0792     1.6215   1.899 0.057568 .  
## buy                  5.0214     1.9637   2.557 0.010553 *  
## tri                 -4.0178     2.5437  -1.579 0.114224    
## cdp                 57.1267  2466.4167   0.023 0.981521    
## doesnt              -2.1867     3.9405  -0.555 0.578950    
## emiss              -65.3626  2466.4048  -0.027 0.978858    
## feel                -2.4023     5.7229  -0.420 0.674653    
## fun                  1.4037     2.7686   0.507 0.612137    
## got                  1.5122     3.5559   0.425 0.670652    
## hour                -0.1233     2.8784  -0.043 0.965844    
## macbook              0.7301     3.3588   0.217 0.827909    
## miss                 7.0787     3.2160   2.201 0.027729 *  
## siri                -1.3719     4.6027  -0.298 0.765664    
## start                3.2558     3.4045   0.956 0.338900    
## upgrad               8.9117     2.7618   3.227 0.001252 ** 
## get                 -0.5138     1.2664  -0.406 0.684921    
## think                2.5240     1.5250   1.655 0.097905 .  
## copi                 6.7733     4.1843   1.619 0.105504    
## guess                4.5797     3.7528   1.220 0.222334    
## person               8.0474     5.0483   1.594 0.110916    
## smart               -2.2304     4.4721  -0.499 0.617961    
## best                 4.5583     3.1502   1.447 0.147896    
## differ               0.7338     2.3510   0.312 0.754948    
## nsa                  1.8094     2.0638   0.877 0.380642    
## product              3.3203     2.6912   1.234 0.217293    
## twitter             -2.2748     2.1446  -1.061 0.288814    
## old                  3.5757     2.0715   1.726 0.084314 .  
## see                 -9.3730    43.1672  -0.217 0.828106    
## download            -3.0606     5.2950  -0.578 0.563254    
## idea                 6.3016     2.5233   2.497 0.012512 *  
## job                 -1.1333     4.6334  -0.245 0.806776    
## mani                 2.5557     3.4581   0.739 0.459887    
## simpl               -7.2732     6.2341  -1.167 0.243342    
## soon                 5.2217     5.2022   1.004 0.315500    
## team                 2.2988     4.5957   0.500 0.616924    
## technolog            4.6206     8.9688   0.515 0.606418    
## wow                 -5.4517     5.6398  -0.967 0.333721    
## much                 1.1447     3.3954   0.337 0.736011    
## support             -0.3005     2.3696  -0.127 0.899097    
## text                -2.4722     3.5759  -0.691 0.489337    
## time                 0.2476     2.1382   0.116 0.907797    
## android              1.8172     1.5819   1.149 0.250654    
## that                 2.3894     2.6910   0.888 0.374571    
## fingerprint          0.6778     2.0249   0.335 0.737810    
## now                  0.7887     1.4362   0.549 0.582891    
## ask                  0.7586     2.4126   0.314 0.753184    
## colour               1.3380     2.7149   0.493 0.622125    
## gonna               -7.5310     6.8267  -1.103 0.269954    
## happen              -3.1729    41.1630  -0.077 0.938558    
## man                  3.2858     2.4052   1.366 0.171896    
## give                 1.3190     2.2237   0.593 0.553090    
## hey                 -6.8528     3.7503  -1.827 0.067662 .  
## today               -3.2471     3.7967  -0.855 0.392419    
## case                 3.8488     3.4790   1.106 0.268596    
## chang                4.9397     2.3426   2.109 0.034974 *  
## one                 -1.4582     1.9308  -0.755 0.450092    
## work                -3.1891     3.7275  -0.856 0.392237    
## day                  5.2256     3.0478   1.715 0.086429 .  
## nokia                0.8437     3.2609   0.259 0.795837    
## preorder            -6.6698    12.9653  -0.514 0.606950    
## look                -1.1209     1.8218  -0.615 0.538368    
## awesom               2.6210     1.7157   1.528 0.126600    
## done                 5.7082     2.4483   2.331 0.019727 *  
## featur              -0.4649     6.0655  -0.077 0.938906    
## said                 4.8859     2.4953   1.958 0.050231 .  
## secur                0.7089     3.5540   0.199 0.841898    
## seem                 0.5479     2.5813   0.212 0.831898    
## smartphon            7.2719     4.3854   1.658 0.097275 .  
## thought              7.9712     3.1405   2.538 0.011143 *  
## what                 1.9916     5.7571   0.346 0.729394    
## windowsphon          4.1409     1.9940   2.077 0.037832 *  
## bit                  0.2055     1.6557   0.124 0.901244    
## realli              -5.5976     2.6374  -2.122 0.033803 *  
## can                 -1.9029     1.6440  -1.158 0.247062    
## sure                -1.0157     2.4499  -0.415 0.678434    
## dont                -3.3268     2.3315  -1.427 0.153611    
## impress             -0.8317     4.1098  -0.202 0.839635    
## call                 5.0740     1.8913   2.683 0.007300 ** 
## didnt                0.3437     2.2320   0.154 0.877616    
## drop                -1.4784     4.3729  -0.338 0.735300    
## fail                -1.0601     2.7400  -0.387 0.698829    
## first                6.1029     2.8523   2.140 0.032382 *  
## improv               0.8695     3.3464   0.260 0.794996    
## isnt                -2.7901     4.1092  -0.679 0.497144    
## made                -2.6856     3.9414  -0.681 0.495629    
## mean                 0.1190     4.4485   0.027 0.978655    
## put                 -1.4000     5.7340  -0.244 0.807103    
## yet                 -6.0535     5.8116  -1.042 0.297587    
## next.               -0.3392     4.5548  -0.074 0.940629    
## thank               -0.4397     1.3702  -0.321 0.748267    
## market               0.6426     3.0217   0.213 0.831589    
## blackberri           2.6499     2.2374   1.184 0.236261    
## mac                  2.3323     3.3632   0.693 0.488011    
## never               -9.2031    57.1015  -0.161 0.871959    
## plastic             -5.0998     3.9296  -1.298 0.194358    
## right                0.3378     3.1728   0.106 0.915220    
## sell                 4.3595     3.5363   1.233 0.217659    
## pleas               -3.0839     2.8663  -1.076 0.281967    
## guy                 -3.6423     8.3832  -0.434 0.663946    
## scanner             -6.3453     4.5032  -1.409 0.158818    
## .rnorm              -0.3661     0.3324  -1.101 0.270690    
## want                 3.3752     1.6698   2.021 0.043250 *  
## black               -6.1676     3.1152  -1.980 0.047720 *  
## free                -6.1265     3.6945  -1.658 0.097260 .  
## innov                1.8728     2.1722   0.862 0.388610    
## via                 -2.4786     7.6336  -0.325 0.745408    
## come                -2.2611     2.3578  -0.959 0.337578    
## good                 0.2994     4.3154   0.069 0.944683    
## let                 -2.0011     3.7586  -0.532 0.594452    
## stop                -2.1427     2.8574  -0.750 0.453323    
## color               -5.4951     3.5224  -1.560 0.118743    
## compani             -6.7036     4.6795  -1.433 0.151986    
## know                -3.9729     3.8882  -1.022 0.306878    
## price               -2.0591     3.8277  -0.538 0.590615    
## just                 0.2686     1.6299   0.165 0.869126    
## store               -1.6054     2.2636  -0.709 0.478167    
## ios7                 2.1131     3.0054   0.703 0.481985    
## peopl                0.1478     2.9921   0.049 0.960605    
## releas              -6.9292     4.1049  -1.688 0.091403 .  
## say                 -5.1095     3.3682  -1.517 0.129266    
## anyon                3.1128     2.1092   1.476 0.139991    
## devic              -41.9680  1447.4615  -0.029 0.976869    
## need                -3.8647     2.1098  -1.832 0.066980 .  
## back                -4.5801     3.1253  -1.466 0.142781    
## googl               -5.3238     3.3140  -1.606 0.108168    
## itun                 0.9411     0.5375   1.751 0.079985 .  
## microsoft           -3.6143     2.3832  -1.517 0.129370    
## will                -3.9255     2.1233  -1.849 0.064486 .  
## appl                -1.0739     3.4113  -0.315 0.752901    
## new                 -3.1416     1.6851  -1.864 0.062283 .  
## iphone5              0.8216     1.6849   0.488 0.625829    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 705.23  on 824  degrees of freedom
## Residual deviance: 215.13  on 621  degrees of freedom
## AIC: 623.13
## 
## Number of Fisher Scoring iterations: 18
```

![](Apple_Tweets_files/figure-html/fit.models_0-37.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 699 126
##   Negative.fctr Negative.fctr.predict.Low.cor.X.glm.N
## 1             N                                     0
## 2             Y                                     0
##   Negative.fctr.predict.Low.cor.X.glm.Y
## 1                                   699
## 2                                   126
##           Reference
## Prediction   N   Y
##          N 609   8
##          Y  90 118
##   Negative.fctr Negative.fctr.predict.Low.cor.X.glm.N
## 1             N                                   609
## 2             Y                                     8
##   Negative.fctr.predict.Low.cor.X.glm.Y
## 1                                    90
## 2                                   118
##           Reference
## Prediction   N   Y
##          N 666   9
##          Y  33 117
##   Negative.fctr Negative.fctr.predict.Low.cor.X.glm.N
## 1             N                                   666
## 2             Y                                     9
##   Negative.fctr.predict.Low.cor.X.glm.Y
## 1                                    33
## 2                                   117
##           Reference
## Prediction   N   Y
##          N 683  11
##          Y  16 115
##   Negative.fctr Negative.fctr.predict.Low.cor.X.glm.N
## 1             N                                   683
## 2             Y                                    11
##   Negative.fctr.predict.Low.cor.X.glm.Y
## 1                                    16
## 2                                   115
##           Reference
## Prediction   N   Y
##          N 693  14
##          Y   6 112
##   Negative.fctr Negative.fctr.predict.Low.cor.X.glm.N
## 1             N                                   693
## 2             Y                                    14
##   Negative.fctr.predict.Low.cor.X.glm.Y
## 1                                     6
## 2                                   112
##           Reference
## Prediction   N   Y
##          N 694  19
##          Y   5 107
##   Negative.fctr Negative.fctr.predict.Low.cor.X.glm.N
## 1             N                                   694
## 2             Y                                    19
##   Negative.fctr.predict.Low.cor.X.glm.Y
## 1                                     5
## 2                                   107
##           Reference
## Prediction   N   Y
##          N 694  25
##          Y   5 101
##   Negative.fctr Negative.fctr.predict.Low.cor.X.glm.N
## 1             N                                   694
## 2             Y                                    25
##   Negative.fctr.predict.Low.cor.X.glm.Y
## 1                                     5
## 2                                   101
##           Reference
## Prediction   N   Y
##          N 697  32
##          Y   2  94
##   Negative.fctr Negative.fctr.predict.Low.cor.X.glm.N
## 1             N                                   697
## 2             Y                                    32
##   Negative.fctr.predict.Low.cor.X.glm.Y
## 1                                     2
## 2                                    94
##           Reference
## Prediction   N   Y
##          N 698  44
##          Y   1  82
##   Negative.fctr Negative.fctr.predict.Low.cor.X.glm.N
## 1             N                                   698
## 2             Y                                    44
##   Negative.fctr.predict.Low.cor.X.glm.Y
## 1                                     1
## 2                                    82
##           Reference
## Prediction   N   Y
##          N 698  61
##          Y   1  65
##   Negative.fctr Negative.fctr.predict.Low.cor.X.glm.N
## 1             N                                   698
## 2             Y                                    61
##   Negative.fctr.predict.Low.cor.X.glm.Y
## 1                                     1
## 2                                    65
##           Reference
## Prediction   N   Y
##          N 699 126
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Low.cor.X.glm.N
## 1             N                                   699
## 2             Y                                   126
##   Negative.fctr.predict.Low.cor.X.glm.Y
## 1                                     0
## 2                                     0
##    threshold   f.score
## 1        0.0 0.2649842
## 2        0.1 0.7065868
## 3        0.2 0.8478261
## 4        0.3 0.8949416
## 5        0.4 0.9180328
## 6        0.5 0.8991597
## 7        0.6 0.8706897
## 8        0.7 0.8468468
## 9        0.8 0.7846890
## 10       0.9 0.6770833
## 11       1.0 0.0000000
```

![](Apple_Tweets_files/figure-html/fit.models_0-38.png) 

```
## [1] "Classifier Probability Threshold: 0.4000 to maximize f.score.fit"
##   Negative.fctr Negative.fctr.predict.Low.cor.X.glm.N
## 1             N                                   693
## 2             Y                                    14
##   Negative.fctr.predict.Low.cor.X.glm.Y
## 1                                     6
## 2                                   112
##           Reference
## Prediction   N   Y
##          N 693  14
##          Y   6 112
##   Negative.fctr Negative.fctr.predict.Low.cor.X.glm.N
## 1             N                                   693
## 2             Y                                    14
##   Negative.fctr.predict.Low.cor.X.glm.Y
## 1                                     6
## 2                                   112
##          Prediction
## Reference   N   Y
##         N 693   6
##         Y  14 112
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   9.757576e-01   9.038260e-01   9.628069e-01   9.851307e-01   8.472727e-01 
## AccuracyPValue  McnemarPValue 
##   4.394732e-35   1.175249e-01
```

![](Apple_Tweets_files/figure-html/fit.models_0-39.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 300  56
##   Negative.fctr Negative.fctr.predict.Low.cor.X.glm.N
## 1             N                                     0
## 2             Y                                     0
##   Negative.fctr.predict.Low.cor.X.glm.Y
## 1                                   300
## 2                                    56
##           Reference
## Prediction   N   Y
##          N 245  23
##          Y  55  33
##   Negative.fctr Negative.fctr.predict.Low.cor.X.glm.N
## 1             N                                   245
## 2             Y                                    23
##   Negative.fctr.predict.Low.cor.X.glm.Y
## 1                                    55
## 2                                    33
##           Reference
## Prediction   N   Y
##          N 253  25
##          Y  47  31
##   Negative.fctr Negative.fctr.predict.Low.cor.X.glm.N
## 1             N                                   253
## 2             Y                                    25
##   Negative.fctr.predict.Low.cor.X.glm.Y
## 1                                    47
## 2                                    31
##           Reference
## Prediction   N   Y
##          N 262  25
##          Y  38  31
##   Negative.fctr Negative.fctr.predict.Low.cor.X.glm.N
## 1             N                                   262
## 2             Y                                    25
##   Negative.fctr.predict.Low.cor.X.glm.Y
## 1                                    38
## 2                                    31
##           Reference
## Prediction   N   Y
##          N 267  27
##          Y  33  29
##   Negative.fctr Negative.fctr.predict.Low.cor.X.glm.N
## 1             N                                   267
## 2             Y                                    27
##   Negative.fctr.predict.Low.cor.X.glm.Y
## 1                                    33
## 2                                    29
##           Reference
## Prediction   N   Y
##          N 268  28
##          Y  32  28
##   Negative.fctr Negative.fctr.predict.Low.cor.X.glm.N
## 1             N                                   268
## 2             Y                                    28
##   Negative.fctr.predict.Low.cor.X.glm.Y
## 1                                    32
## 2                                    28
##           Reference
## Prediction   N   Y
##          N 273  29
##          Y  27  27
##   Negative.fctr Negative.fctr.predict.Low.cor.X.glm.N
## 1             N                                   273
## 2             Y                                    29
##   Negative.fctr.predict.Low.cor.X.glm.Y
## 1                                    27
## 2                                    27
##           Reference
## Prediction   N   Y
##          N 275  31
##          Y  25  25
##   Negative.fctr Negative.fctr.predict.Low.cor.X.glm.N
## 1             N                                   275
## 2             Y                                    31
##   Negative.fctr.predict.Low.cor.X.glm.Y
## 1                                    25
## 2                                    25
##           Reference
## Prediction   N   Y
##          N 279  34
##          Y  21  22
##   Negative.fctr Negative.fctr.predict.Low.cor.X.glm.N
## 1             N                                   279
## 2             Y                                    34
##   Negative.fctr.predict.Low.cor.X.glm.Y
## 1                                    21
## 2                                    22
##           Reference
## Prediction   N   Y
##          N 281  35
##          Y  19  21
##   Negative.fctr Negative.fctr.predict.Low.cor.X.glm.N
## 1             N                                   281
## 2             Y                                    35
##   Negative.fctr.predict.Low.cor.X.glm.Y
## 1                                    19
## 2                                    21
##           Reference
## Prediction   N   Y
##          N 300  56
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Low.cor.X.glm.N
## 1             N                                   300
## 2             Y                                    56
##   Negative.fctr.predict.Low.cor.X.glm.Y
## 1                                     0
## 2                                     0
##    threshold   f.score
## 1        0.0 0.2718447
## 2        0.1 0.4583333
## 3        0.2 0.4626866
## 4        0.3 0.4960000
## 5        0.4 0.4915254
## 6        0.5 0.4827586
## 7        0.6 0.4909091
## 8        0.7 0.4716981
## 9        0.8 0.4444444
## 10       0.9 0.4375000
## 11       1.0 0.0000000
```

![](Apple_Tweets_files/figure-html/fit.models_0-40.png) 

```
## [1] "Classifier Probability Threshold: 0.3000 to maximize f.score.OOB"
##   Negative.fctr Negative.fctr.predict.Low.cor.X.glm.N
## 1             N                                   262
## 2             Y                                    25
##   Negative.fctr.predict.Low.cor.X.glm.Y
## 1                                    38
## 2                                    31
##           Reference
## Prediction   N   Y
##          N 262  25
##          Y  38  31
##   Negative.fctr Negative.fctr.predict.Low.cor.X.glm.N
## 1             N                                   262
## 2             Y                                    25
##   Negative.fctr.predict.Low.cor.X.glm.Y
## 1                                    38
## 2                                    31
##          Prediction
## Reference   N   Y
##         N 262  38
##         Y  25  31
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.8230337      0.3900794      0.7793438      0.8612518      0.8426966 
## AccuracyPValue  McnemarPValue 
##      0.8619683      0.1305700 
##        model_id model_method
## 1 Low.cor.X.glm          glm
##                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         feats
## 1 freak, hate, stuff, pictur, wtf, cant, shame, stupid, even, line, yooo, better, ever, fix, charger, still, charg, disappoint, short, like, yall, break., imessag, stand, togeth, cheap, wont, make, darn, httpbitly18xc8dk, dear, facebook, X7evenstarz, femal, money, theyr, batteri, china, turn, hope, life, sinc, steve, switch, everi, last, your, amaz, arent, date, ill, ive, lost, noth, worst, phone, care, way, year, updat, app, iphon, data, take, card, custom, die, event, problem, two, use, screen, thing, buy, tri, cdp, doesnt, emiss, feel, fun, got, hour, macbook, miss, siri, start, upgrad, get, think, copi, guess, person, smart, best, differ, nsa, product, twitter, old, see, download, idea, job, mani, simpl, soon, team, technolog, wow, much, support, text, time, android, that, fingerprint, now, ask, colour, gonna, happen, man, give, hey, today, case, chang, one, work, day, nokia, preorder, look, awesom, done, featur, said, secur, seem, smartphon, thought, what, windowsphon, bit, realli, can, sure, dont, impress, call, didnt, drop, fail, first, improv, isnt, made, mean, put, yet, next., thank, market, blackberri, mac, never, plastic, right, sell, pleas, guy, scanner, .rnorm, want, black, free, innov, via, come, good, let, stop, color, compani, know, price, just, store, ios7, peopl, releas, say, anyon, devic, need, back, googl, itun, microsoft, will, appl, new, iphone5
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               1                      4.411                 1.087
##   max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1   0.9634853                    0.4       0.9180328         0.750303
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.9628069             0.9851307     0.2316521   0.7086607
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.3           0.496        0.8230337
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB min.aic.fit
## 1             0.7793438             0.8612518     0.3900794    623.1343
##   max.AccuracySD.fit max.KappaSD.fit
## 1         0.04963781      0.09400829
```

```r
# All X that is not user excluded
if (glb_is_classification && glb_is_binomial) {
    model_id_pfx <- "Conditional.X"
# indep_vars_vctr <- setdiff(names(glb_trnent_df), union(glb_rsp_var, glb_exclude_vars_as_features))
    indep_vars_vctr <- subset(glb_feats_df, is.ConditionalX.y & 
                                            (exclude.as.feat != 1))[, "id"]
} else {
    model_id_pfx <- "All.X"
    indep_vars_vctr <- subset(glb_feats_df, 
                                            (exclude.as.feat != 1))[, "id"]
}
for (method in glb_models_method_vctr) {
    ret_lst <- myfit_mdl(model_id=paste0(model_id_pfx, ""), model_method=method,
                            indep_vars_vctr=indep_vars_vctr,
                            model_type=glb_model_type,
                            rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                            fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                n_cv_folds=glb_n_cv_folds, tune_models_df=glb_tune_models_df)
    
    # Since caret does not optimize rpart well
    if (method == "rpart")
        ret_lst <- myfit_mdl(model_id=paste0(model_id_pfx, ".cp.0"), model_method=method,
                                indep_vars_vctr=indep_vars_vctr,
                                model_type=glb_model_type,
                                rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                                fit_df=glb_trnent_df, OOB_df=glb_newent_df,        
            n_cv_folds=0, tune_models_df=data.frame(parameter="cp", min=0.0, max=0.0, by=0.1))
    
    # Compare how rf performs w/i & w/o .rnorm
    if (method == "rf")
        ret_lst <- myfit_mdl(model_id=paste0(model_id_pfx, ".no.rnorm"), model_method=method,
                                indep_vars_vctr=setdiff(indep_vars_vctr, c(".rnorm")),
                                model_type=glb_model_type,
                                rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                                fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                    n_cv_folds=glb_n_cv_folds, tune_models_df=glb_tune_models_df)
}
```

```
## [1] "fitting model: Conditional.X.glm"
## [1] "    indep_vars: freak, hate, stuff, pictur, wtf, cant, shame, stupid, even, line, yooo, better, ever, fix, charger, still, charg, disappoint, short, like, amazon, yall, break., imessag, stand, togeth, cheap, wont, make, carbon, darn, httpbitly18xc8dk, dear, facebook, X7evenstarz, condom, femal, money, theyr, batteri, china, turn, hope, life, sinc, steve, switch, everi, last, your, amaz, arent, date, divulg, ill, ive, lost, noth, worst, phone, care, way, year, updat, app, iphon, data, take, card, custom, die, event, problem, refus, two, use, screen, thing, buy, tri, cdp, doesnt, emiss, feel, fun, got, hour, macbook, miss, siri, start, upgrad, get, think, copi, guess, person, smart, best, differ, nsa, product, twitter, old, see, download, idea, job, mani, simpl, soon, team, technolog, wow, much, support, text, time, android, that, fingerprint, now, ask, colour, gonna, happen, man, give, hey, today, case, chang, one, work, day, nokia, preorder, look, awesom, done, featur, said, secur, seem, smartphon, thought, what, windowsphon, bit, realli, can, sure, dont, impress, call, didnt, drop, fail, first, improv, isnt, made, mean, put, yet, next., thank, market, blackberri, mac, never, plastic, right, sell, pleas, guy, scanner, .rnorm, want, black, free, innov, via, come, good, let, stop, color, compani, know, price, just, store, ios7, peopl, releas, say, anyon, devic, need, back, googl, itun, microsoft, will, appl, new, iphone5, ipad"
## + Fold1: parameter=none
```

```
## Warning: glm.fit: algorithm did not converge
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold1: parameter=none 
## + Fold2: parameter=none
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold2: parameter=none 
## + Fold3: parameter=none
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold3: parameter=none 
## Aggregating results
## Fitting final model on full training set
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
## Warning: not plotting observations with leverage one:
##   98, 607, 788
```

![](Apple_Tweets_files/figure-html/fit.models_0-41.png) ![](Apple_Tweets_files/figure-html/fit.models_0-42.png) 

```
## Warning: not plotting observations with leverage one:
##   98, 607, 788
```

![](Apple_Tweets_files/figure-html/fit.models_0-43.png) 

```
## Warning in sqrt(crit * p * (1 - hh)/hh): NaNs produced
```

```
## Warning in sqrt(crit * p * (1 - hh)/hh): NaNs produced
```

```
## 
## Call:
## NULL
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -2.3211  -0.1780  -0.0512  -0.0013   3.7642  
## 
## Coefficients: (1 not defined because of singularities)
##                    Estimate Std. Error z value Pr(>|z|)    
## (Intercept)      -6.393e+00  7.872e-01  -8.121 4.61e-16 ***
## freak             9.162e+00  1.418e+00   6.463 1.03e-10 ***
## hate              1.032e+01  2.974e+00   3.469 0.000523 ***
## stuff            -2.249e+00  2.136e+00  -1.053 0.292300    
## pictur            5.058e+00  3.011e+00   1.680 0.092980 .  
## wtf               7.351e+00  2.992e+00   2.457 0.014028 *  
## cant              3.991e+00  1.316e+00   3.031 0.002434 ** 
## shame             7.109e+01  3.422e+03   0.021 0.983424    
## stupid            1.139e+01  9.189e+00   1.240 0.215091    
## even              5.968e+00  3.150e+00   1.895 0.058143 .  
## line              3.364e+00  1.523e+01   0.221 0.825263    
## yooo              3.220e+00  3.693e+00   0.872 0.383133    
## better            2.561e+00  1.352e+00   1.895 0.058139 .  
## ever              4.574e+00  4.796e+00   0.954 0.340256    
## fix               4.833e+00  2.254e+00   2.144 0.032032 *  
## charger           4.845e+00  2.033e+00   2.384 0.017147 *  
## still             1.516e+00  1.980e+00   0.766 0.443832    
## charg             3.536e-01  9.838e+00   0.036 0.971326    
## disappoint        1.527e+01  5.021e+00   3.041 0.002361 ** 
## short             3.443e+00  5.509e+00   0.625 0.531927    
## like              3.781e+00  1.252e+00   3.021 0.002520 ** 
## amazon           -1.117e+01  2.360e+06   0.000 0.999996    
## yall              1.388e+00  4.047e+00   0.343 0.731581    
## break.            5.974e+00  2.455e+00   2.434 0.014940 *  
## imessag           7.047e+00  3.689e+00   1.910 0.056113 .  
## stand             4.191e+00  5.559e+00   0.754 0.450860    
## togeth            7.190e+00  4.114e+00   1.747 0.080553 .  
## cheap             2.555e+00  3.431e+00   0.745 0.456459    
## wont             -3.728e+00  2.253e+00  -1.654 0.098058 .  
## make             -1.434e+00  1.336e+00  -1.073 0.283207    
## carbon           -3.216e+00  2.360e+06   0.000 0.999999    
## darn              5.507e+00  1.663e+00   3.312 0.000927 ***
## httpbitly18xc8dk -9.438e+00  2.404e+06   0.000 0.999997    
## dear              7.578e+00  3.183e+00   2.380 0.017294 *  
## facebook         -1.307e+01  6.419e+03  -0.002 0.998376    
## X7evenstarz       4.118e+00  1.010e+01   0.408 0.683565    
## condom            4.951e+00  2.919e+00   1.696 0.089854 .  
## femal                    NA         NA      NA       NA    
## money             2.766e+00  4.263e+00   0.649 0.516479    
## theyr             1.017e+01  3.909e+00   2.601 0.009291 ** 
## batteri           3.158e+00  2.774e+00   1.138 0.255016    
## china             5.322e+00  1.641e+00   3.243 0.001184 ** 
## turn             -3.304e+00  3.930e+00  -0.841 0.400492    
## hope              6.030e+00  3.360e+00   1.795 0.072712 .  
## life              6.307e+00  1.834e+00   3.440 0.000582 ***
## sinc              2.916e+00  3.735e+00   0.781 0.434918    
## steve            -8.925e-01  3.589e+00  -0.249 0.803629    
## switch            1.119e+00  3.680e+00   0.304 0.761102    
## everi             4.744e+00  2.178e+00   2.178 0.029381 *  
## last             -2.291e+01  8.857e+04   0.000 0.999794    
## your              3.139e+00  2.289e+00   1.371 0.170313    
## amaz             -6.639e-01  6.158e+00  -0.108 0.914138    
## arent             2.263e+00  3.959e+00   0.572 0.567653    
## date             -2.813e+00  6.932e+00  -0.406 0.684833    
## divulg            9.498e+01  4.115e+06   0.000 0.999982    
## ill               1.458e+01  5.652e+01   0.258 0.796460    
## ive              -2.613e+00  5.211e+00  -0.501 0.616124    
## lost              3.256e+00  3.906e+00   0.834 0.404518    
## noth              2.259e+00  6.336e+00   0.357 0.721396    
## worst            -8.211e-01  2.685e+00  -0.306 0.759750    
## phone            -2.554e-01  1.390e+00  -0.184 0.854178    
## care              4.786e+00  2.661e+00   1.798 0.072136 .  
## way              -1.669e+00  2.342e+00  -0.713 0.476110    
## year              4.265e+00  2.672e+00   1.596 0.110434    
## updat            -8.471e-01  2.369e+00  -0.358 0.720653    
## app               2.305e+00  1.510e+00   1.526 0.126935    
## iphon             1.147e+00  7.847e-01   1.462 0.143647    
## data              5.089e+00  1.831e+00   2.779 0.005454 ** 
## take              2.837e+00  2.132e+00   1.331 0.183319    
## card              1.504e+00  2.181e+00   0.690 0.490399    
## custom           -2.434e+00  5.539e+00  -0.439 0.660344    
## die              -1.250e+00  4.055e+00  -0.308 0.757891    
## event             2.133e+00  3.190e+00   0.669 0.503663    
## problem           6.756e+00  4.466e+00   1.513 0.130383    
## refus            -4.381e+01  4.719e+06   0.000 0.999993    
## two               1.021e+01  5.274e+00   1.936 0.052898 .  
## use               1.069e+00  1.562e+00   0.684 0.493726    
## screen            3.898e+00  2.830e+00   1.378 0.168343    
## thing             3.145e+00  1.607e+00   1.958 0.050243 .  
## buy               5.239e+00  2.030e+00   2.580 0.009868 ** 
## tri              -3.757e+00  2.506e+00  -1.499 0.133810    
## cdp               2.641e+01  2.360e+06   0.000 0.999991    
## doesnt           -2.414e+00  4.285e+00  -0.563 0.573121    
## emiss            -4.251e+01  4.719e+06   0.000 0.999993    
## feel             -2.301e+00  5.569e+00  -0.413 0.679443    
## fun               1.108e+00  2.792e+00   0.397 0.691466    
## got               1.312e+00  3.452e+00   0.380 0.703971    
## hour              6.726e-02  2.842e+00   0.024 0.981120    
## macbook           8.066e-01  3.288e+00   0.245 0.806190    
## miss              9.230e+00  3.759e+00   2.456 0.014067 *  
## siri             -2.033e+00  4.638e+00  -0.438 0.661158    
## start             2.984e+00  3.520e+00   0.848 0.396626    
## upgrad            8.819e+00  2.690e+00   3.278 0.001044 ** 
## get              -4.875e-01  1.281e+00  -0.380 0.703581    
## think             2.573e+00  1.510e+00   1.703 0.088504 .  
## copi              5.972e+00  4.265e+00   1.400 0.161445    
## guess             7.445e+00  3.768e+00   1.976 0.048165 *  
## person            7.979e+00  4.948e+00   1.613 0.106825    
## smart            -2.496e+00  4.861e+00  -0.514 0.607596    
## best              4.180e+00  3.126e+00   1.337 0.181093    
## differ            9.135e-01  2.401e+00   0.380 0.703643    
## nsa               1.665e+00  2.111e+00   0.789 0.430264    
## product           3.447e+00  2.563e+00   1.345 0.178751    
## twitter          -2.466e+00  2.274e+00  -1.084 0.278161    
## old               3.916e+00  2.135e+00   1.834 0.066670 .  
## see              -9.845e+00  4.619e+01  -0.213 0.831221    
## download         -2.486e+00  5.139e+00  -0.484 0.628648    
## idea              6.054e+00  2.689e+00   2.251 0.024385 *  
## job              -9.443e-01  4.722e+00  -0.200 0.841491    
## mani              2.593e+00  3.580e+00   0.724 0.468827    
## simpl            -9.484e+00  6.561e+00  -1.445 0.148328    
## soon              5.019e+00  5.177e+00   0.969 0.332362    
## team              2.528e+00  4.421e+00   0.572 0.567371    
## technolog         8.686e+00  1.885e+01   0.461 0.644873    
## wow              -5.301e+00  6.056e+00  -0.875 0.381414    
## much              1.343e+00  3.469e+00   0.387 0.698698    
## support          -1.211e-01  2.687e+00  -0.045 0.964046    
## text             -2.910e+00  3.643e+00  -0.799 0.424517    
## time             -4.163e-01  2.291e+00  -0.182 0.855802    
## android           1.711e+00  1.585e+00   1.079 0.280462    
## that              2.091e+00  2.749e+00   0.761 0.446806    
## fingerprint       6.299e-01  2.123e+00   0.297 0.766680    
## now               7.270e-01  1.494e+00   0.486 0.626618    
## ask               6.203e-01  2.486e+00   0.250 0.802953    
## colour            1.009e+00  2.780e+00   0.363 0.716760    
## gonna            -8.251e+00  9.123e+00  -0.905 0.365730    
## happen           -4.242e+00  3.455e+01  -0.123 0.902277    
## man               3.311e+00  2.401e+00   1.379 0.167794    
## give              1.664e+00  2.363e+00   0.704 0.481339    
## hey              -7.179e+00  4.093e+00  -1.754 0.079423 .  
## today            -2.338e+00  3.834e+00  -0.610 0.542031    
## case              4.744e+00  3.414e+00   1.389 0.164699    
## chang             4.850e+00  2.330e+00   2.081 0.037393 *  
## one              -1.989e+00  2.106e+00  -0.944 0.345006    
## work             -3.400e+00  3.736e+00  -0.910 0.362687    
## day               4.990e+00  3.129e+00   1.595 0.110812    
## nokia             2.355e-02  3.533e+00   0.007 0.994683    
## preorder         -8.001e+00  1.452e+01  -0.551 0.581487    
## look             -1.369e+00  1.922e+00  -0.712 0.476345    
## awesom            2.695e+00  1.747e+00   1.543 0.122915    
## done              5.602e+00  2.475e+00   2.263 0.023611 *  
## featur           -6.516e-01  6.298e+00  -0.103 0.917594    
## said              4.973e+00  2.508e+00   1.983 0.047393 *  
## secur             5.288e-01  3.531e+00   0.150 0.880956    
## seem              6.595e-01  2.655e+00   0.248 0.803805    
## smartphon         8.578e+00  4.571e+00   1.877 0.060557 .  
## thought           8.352e+00  3.608e+00   2.315 0.020638 *  
## what              2.027e+00  6.135e+00   0.330 0.741131    
## windowsphon       4.093e+00  1.972e+00   2.075 0.037969 *  
## bit               1.877e-01  1.665e+00   0.113 0.910220    
## realli           -5.617e+00  2.663e+00  -2.109 0.034918 *  
## can              -1.702e+00  1.685e+00  -1.010 0.312460    
## sure             -8.192e-01  2.501e+00  -0.328 0.743249    
## dont             -3.202e+00  2.317e+00  -1.382 0.167028    
## impress          -3.330e-01  3.945e+00  -0.084 0.932729    
## call              4.945e+00  1.909e+00   2.591 0.009581 ** 
## didnt             3.805e-01  2.251e+00   0.169 0.865797    
## drop             -3.880e+00  4.885e+00  -0.794 0.426965    
## fail             -1.102e+00  2.674e+00  -0.412 0.680350    
## first             6.459e+00  3.018e+00   2.140 0.032367 *  
## improv            1.003e+00  3.304e+00   0.304 0.761434    
## isnt             -2.307e+00  4.509e+00  -0.512 0.608949    
## made             -2.578e+00  3.909e+00  -0.660 0.509554    
## mean              6.633e-01  4.657e+00   0.142 0.886737    
## put              -1.536e+00  5.656e+00  -0.272 0.785943    
## yet              -6.462e+00  5.903e+00  -1.095 0.273593    
## next.             5.762e-01  4.720e+00   0.122 0.902848    
## thank            -5.940e-01  1.437e+00  -0.413 0.679354    
## market            1.899e+00  2.539e+00   0.748 0.454445    
## blackberri        2.786e+00  2.163e+00   1.288 0.197765    
## mac               2.099e+00  3.502e+00   0.599 0.548864    
## never            -1.050e+01  5.678e+01  -0.185 0.853310    
## plastic          -6.757e+00  4.268e+00  -1.583 0.113385    
## right             5.167e-01  3.213e+00   0.161 0.872238    
## sell              4.129e+00  3.546e+00   1.165 0.244146    
## pleas            -3.411e+00  3.056e+00  -1.116 0.264311    
## guy              -2.946e+00  1.120e+01  -0.263 0.792498    
## scanner          -7.743e+00  4.906e+00  -1.578 0.114539    
## .rnorm           -4.078e-01  3.364e-01  -1.212 0.225376    
## want              3.360e+00  1.707e+00   1.969 0.048991 *  
## black            -6.128e+00  3.130e+00  -1.958 0.050252 .  
## free             -7.147e+00  3.749e+00  -1.906 0.056628 .  
## innov             1.812e+00  2.144e+00   0.845 0.398013    
## via              -1.910e+01  2.656e+03  -0.007 0.994263    
## come             -2.213e+00  2.399e+00  -0.922 0.356308    
## good              2.133e-01  3.934e+00   0.054 0.956754    
## let              -2.949e+00  3.645e+00  -0.809 0.418587    
## stop             -2.276e+00  2.928e+00  -0.777 0.437016    
## color            -5.407e+00  3.264e+00  -1.656 0.097657 .  
## compani          -5.923e+00  4.523e+00  -1.310 0.190362    
## know             -3.860e+00  3.856e+00  -1.001 0.316843    
## price            -9.302e-01  3.523e+00  -0.264 0.791751    
## just              1.934e-01  1.636e+00   0.118 0.905904    
## store            -1.528e+00  2.187e+00  -0.699 0.484820    
## ios7              2.098e+00  3.101e+00   0.676 0.498724    
## peopl             2.078e-01  3.104e+00   0.067 0.946625    
## releas           -6.632e+00  4.267e+00  -1.554 0.120110    
## say              -5.207e+00  3.348e+00  -1.555 0.119883    
## anyon             2.952e+00  2.252e+00   1.311 0.189949    
## devic            -4.604e+01  1.593e+03  -0.029 0.976940    
## need             -3.808e+00  2.114e+00  -1.801 0.071640 .  
## back             -5.625e+00  3.573e+00  -1.574 0.115436    
## googl            -5.192e+00  3.239e+00  -1.603 0.108970    
## itun              2.276e+00  1.116e+00   2.039 0.041490 *  
## microsoft        -3.679e+00  2.440e+00  -1.508 0.131625    
## will             -3.733e+00  2.198e+00  -1.698 0.089485 .  
## appl             -1.461e+00  3.362e+00  -0.435 0.663772    
## new              -3.149e+00  1.727e+00  -1.823 0.068257 .  
## iphone5           8.373e-01  1.631e+00   0.513 0.607712    
## ipad             -2.375e+00  1.659e+00  -1.431 0.152377    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 705.23  on 824  degrees of freedom
## Residual deviance: 212.78  on 616  degrees of freedom
## AIC: 630.78
## 
## Number of Fisher Scoring iterations: 18
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

![](Apple_Tweets_files/figure-html/fit.models_0-44.png) ![](Apple_Tweets_files/figure-html/fit.models_0-45.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 699 126
##   Negative.fctr Negative.fctr.predict.Conditional.X.glm.N
## 1             N                                         0
## 2             Y                                         0
##   Negative.fctr.predict.Conditional.X.glm.Y
## 1                                       699
## 2                                       126
##           Reference
## Prediction   N   Y
##          N 608   8
##          Y  91 118
##   Negative.fctr Negative.fctr.predict.Conditional.X.glm.N
## 1             N                                       608
## 2             Y                                         8
##   Negative.fctr.predict.Conditional.X.glm.Y
## 1                                        91
## 2                                       118
##           Reference
## Prediction   N   Y
##          N 661   9
##          Y  38 117
##   Negative.fctr Negative.fctr.predict.Conditional.X.glm.N
## 1             N                                       661
## 2             Y                                         9
##   Negative.fctr.predict.Conditional.X.glm.Y
## 1                                        38
## 2                                       117
##           Reference
## Prediction   N   Y
##          N 683   9
##          Y  16 117
##   Negative.fctr Negative.fctr.predict.Conditional.X.glm.N
## 1             N                                       683
## 2             Y                                         9
##   Negative.fctr.predict.Conditional.X.glm.Y
## 1                                        16
## 2                                       117
##           Reference
## Prediction   N   Y
##          N 693  13
##          Y   6 113
##   Negative.fctr Negative.fctr.predict.Conditional.X.glm.N
## 1             N                                       693
## 2             Y                                        13
##   Negative.fctr.predict.Conditional.X.glm.Y
## 1                                         6
## 2                                       113
##           Reference
## Prediction   N   Y
##          N 694  17
##          Y   5 109
##   Negative.fctr Negative.fctr.predict.Conditional.X.glm.N
## 1             N                                       694
## 2             Y                                        17
##   Negative.fctr.predict.Conditional.X.glm.Y
## 1                                         5
## 2                                       109
##           Reference
## Prediction   N   Y
##          N 695  25
##          Y   4 101
##   Negative.fctr Negative.fctr.predict.Conditional.X.glm.N
## 1             N                                       695
## 2             Y                                        25
##   Negative.fctr.predict.Conditional.X.glm.Y
## 1                                         4
## 2                                       101
##           Reference
## Prediction   N   Y
##          N 697  32
##          Y   2  94
##   Negative.fctr Negative.fctr.predict.Conditional.X.glm.N
## 1             N                                       697
## 2             Y                                        32
##   Negative.fctr.predict.Conditional.X.glm.Y
## 1                                         2
## 2                                        94
##           Reference
## Prediction   N   Y
##          N 698  46
##          Y   1  80
##   Negative.fctr Negative.fctr.predict.Conditional.X.glm.N
## 1             N                                       698
## 2             Y                                        46
##   Negative.fctr.predict.Conditional.X.glm.Y
## 1                                         1
## 2                                        80
##           Reference
## Prediction   N   Y
##          N 698  59
##          Y   1  67
##   Negative.fctr Negative.fctr.predict.Conditional.X.glm.N
## 1             N                                       698
## 2             Y                                        59
##   Negative.fctr.predict.Conditional.X.glm.Y
## 1                                         1
## 2                                        67
##           Reference
## Prediction   N   Y
##          N 699 126
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Conditional.X.glm.N
## 1             N                                       699
## 2             Y                                       126
##   Negative.fctr.predict.Conditional.X.glm.Y
## 1                                         0
## 2                                         0
##    threshold   f.score
## 1        0.0 0.2649842
## 2        0.1 0.7044776
## 3        0.2 0.8327402
## 4        0.3 0.9034749
## 5        0.4 0.9224490
## 6        0.5 0.9083333
## 7        0.6 0.8744589
## 8        0.7 0.8468468
## 9        0.8 0.7729469
## 10       0.9 0.6907216
## 11       1.0 0.0000000
```

```
## [1] "Classifier Probability Threshold: 0.4000 to maximize f.score.fit"
##   Negative.fctr Negative.fctr.predict.Conditional.X.glm.N
## 1             N                                       693
## 2             Y                                        13
##   Negative.fctr.predict.Conditional.X.glm.Y
## 1                                         6
## 2                                       113
##           Reference
## Prediction   N   Y
##          N 693  13
##          Y   6 113
##   Negative.fctr Negative.fctr.predict.Conditional.X.glm.N
## 1             N                                       693
## 2             Y                                        13
##   Negative.fctr.predict.Conditional.X.glm.Y
## 1                                         6
## 2                                       113
##          Prediction
## Reference   N   Y
##         N 693   6
##         Y  13 113
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   9.769697e-01   9.089388e-01   9.642682e-01   9.860786e-01   8.472727e-01 
## AccuracyPValue  McnemarPValue 
##   6.001205e-36   1.686686e-01
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

![](Apple_Tweets_files/figure-html/fit.models_0-46.png) ![](Apple_Tweets_files/figure-html/fit.models_0-47.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 300  56
##   Negative.fctr Negative.fctr.predict.Conditional.X.glm.N
## 1             N                                         0
## 2             Y                                         0
##   Negative.fctr.predict.Conditional.X.glm.Y
## 1                                       300
## 2                                        56
##           Reference
## Prediction   N   Y
##          N 244  23
##          Y  56  33
##   Negative.fctr Negative.fctr.predict.Conditional.X.glm.N
## 1             N                                       244
## 2             Y                                        23
##   Negative.fctr.predict.Conditional.X.glm.Y
## 1                                        56
## 2                                        33
##           Reference
## Prediction   N   Y
##          N 255  25
##          Y  45  31
##   Negative.fctr Negative.fctr.predict.Conditional.X.glm.N
## 1             N                                       255
## 2             Y                                        25
##   Negative.fctr.predict.Conditional.X.glm.Y
## 1                                        45
## 2                                        31
##           Reference
## Prediction   N   Y
##          N 258  26
##          Y  42  30
##   Negative.fctr Negative.fctr.predict.Conditional.X.glm.N
## 1             N                                       258
## 2             Y                                        26
##   Negative.fctr.predict.Conditional.X.glm.Y
## 1                                        42
## 2                                        30
##           Reference
## Prediction   N   Y
##          N 266  27
##          Y  34  29
##   Negative.fctr Negative.fctr.predict.Conditional.X.glm.N
## 1             N                                       266
## 2             Y                                        27
##   Negative.fctr.predict.Conditional.X.glm.Y
## 1                                        34
## 2                                        29
##           Reference
## Prediction   N   Y
##          N 268  28
##          Y  32  28
##   Negative.fctr Negative.fctr.predict.Conditional.X.glm.N
## 1             N                                       268
## 2             Y                                        28
##   Negative.fctr.predict.Conditional.X.glm.Y
## 1                                        32
## 2                                        28
##           Reference
## Prediction   N   Y
##          N 272  29
##          Y  28  27
##   Negative.fctr Negative.fctr.predict.Conditional.X.glm.N
## 1             N                                       272
## 2             Y                                        29
##   Negative.fctr.predict.Conditional.X.glm.Y
## 1                                        28
## 2                                        27
##           Reference
## Prediction   N   Y
##          N 274  30
##          Y  26  26
##   Negative.fctr Negative.fctr.predict.Conditional.X.glm.N
## 1             N                                       274
## 2             Y                                        30
##   Negative.fctr.predict.Conditional.X.glm.Y
## 1                                        26
## 2                                        26
##           Reference
## Prediction   N   Y
##          N 277  33
##          Y  23  23
##   Negative.fctr Negative.fctr.predict.Conditional.X.glm.N
## 1             N                                       277
## 2             Y                                        33
##   Negative.fctr.predict.Conditional.X.glm.Y
## 1                                        23
## 2                                        23
##           Reference
## Prediction   N   Y
##          N 281  34
##          Y  19  22
##   Negative.fctr Negative.fctr.predict.Conditional.X.glm.N
## 1             N                                       281
## 2             Y                                        34
##   Negative.fctr.predict.Conditional.X.glm.Y
## 1                                        19
## 2                                        22
##           Reference
## Prediction   N   Y
##          N 300  56
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Conditional.X.glm.N
## 1             N                                       300
## 2             Y                                        56
##   Negative.fctr.predict.Conditional.X.glm.Y
## 1                                         0
## 2                                         0
##    threshold   f.score
## 1        0.0 0.2718447
## 2        0.1 0.4551724
## 3        0.2 0.4696970
## 4        0.3 0.4687500
## 5        0.4 0.4873950
## 6        0.5 0.4827586
## 7        0.6 0.4864865
## 8        0.7 0.4814815
## 9        0.8 0.4509804
## 10       0.9 0.4536082
## 11       1.0 0.0000000
```

![](Apple_Tweets_files/figure-html/fit.models_0-48.png) 

```
## [1] "Classifier Probability Threshold: 0.4000 to maximize f.score.OOB"
##   Negative.fctr Negative.fctr.predict.Conditional.X.glm.N
## 1             N                                       266
## 2             Y                                        27
##   Negative.fctr.predict.Conditional.X.glm.Y
## 1                                        34
## 2                                        29
##           Reference
## Prediction   N   Y
##          N 266  27
##          Y  34  29
##   Negative.fctr Negative.fctr.predict.Conditional.X.glm.N
## 1             N                                       266
## 2             Y                                        27
##   Negative.fctr.predict.Conditional.X.glm.Y
## 1                                        34
## 2                                        29
##          Prediction
## Reference   N   Y
##         N 266  34
##         Y  27  29
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.8286517      0.3849553      0.7854100      0.8663234      0.8426966 
## AccuracyPValue  McnemarPValue 
##      0.7900786      0.4423557 
##            model_id model_method
## 1 Conditional.X.glm          glm
##                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      feats
## 1 freak, hate, stuff, pictur, wtf, cant, shame, stupid, even, line, yooo, better, ever, fix, charger, still, charg, disappoint, short, like, amazon, yall, break., imessag, stand, togeth, cheap, wont, make, carbon, darn, httpbitly18xc8dk, dear, facebook, X7evenstarz, condom, femal, money, theyr, batteri, china, turn, hope, life, sinc, steve, switch, everi, last, your, amaz, arent, date, divulg, ill, ive, lost, noth, worst, phone, care, way, year, updat, app, iphon, data, take, card, custom, die, event, problem, refus, two, use, screen, thing, buy, tri, cdp, doesnt, emiss, feel, fun, got, hour, macbook, miss, siri, start, upgrad, get, think, copi, guess, person, smart, best, differ, nsa, product, twitter, old, see, download, idea, job, mani, simpl, soon, team, technolog, wow, much, support, text, time, android, that, fingerprint, now, ask, colour, gonna, happen, man, give, hey, today, case, chang, one, work, day, nokia, preorder, look, awesom, done, featur, said, secur, seem, smartphon, thought, what, windowsphon, bit, realli, can, sure, dont, impress, call, didnt, drop, fail, first, improv, isnt, made, mean, put, yet, next., thank, market, blackberri, mac, never, plastic, right, sell, pleas, guy, scanner, .rnorm, want, black, free, innov, via, come, good, let, stop, color, compani, know, price, just, store, ios7, peopl, releas, say, anyon, devic, need, back, googl, itun, microsoft, will, appl, new, iphone5, ipad
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               1                      5.138                 1.152
##   max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1   0.9641324                    0.4        0.922449        0.7612121
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.9642682             0.9860786     0.2321957   0.7152381
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.4        0.487395        0.8286517
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB min.aic.fit
## 1               0.78541             0.8663234     0.3849553    630.7763
##   max.AccuracySD.fit max.KappaSD.fit
## 1         0.03531831      0.08116562
## [1] "fitting model: Conditional.X.rpart"
## [1] "    indep_vars: freak, hate, stuff, pictur, wtf, cant, shame, stupid, even, line, yooo, better, ever, fix, charger, still, charg, disappoint, short, like, amazon, yall, break., imessag, stand, togeth, cheap, wont, make, carbon, darn, httpbitly18xc8dk, dear, facebook, X7evenstarz, condom, femal, money, theyr, batteri, china, turn, hope, life, sinc, steve, switch, everi, last, your, amaz, arent, date, divulg, ill, ive, lost, noth, worst, phone, care, way, year, updat, app, iphon, data, take, card, custom, die, event, problem, refus, two, use, screen, thing, buy, tri, cdp, doesnt, emiss, feel, fun, got, hour, macbook, miss, siri, start, upgrad, get, think, copi, guess, person, smart, best, differ, nsa, product, twitter, old, see, download, idea, job, mani, simpl, soon, team, technolog, wow, much, support, text, time, android, that, fingerprint, now, ask, colour, gonna, happen, man, give, hey, today, case, chang, one, work, day, nokia, preorder, look, awesom, done, featur, said, secur, seem, smartphon, thought, what, windowsphon, bit, realli, can, sure, dont, impress, call, didnt, drop, fail, first, improv, isnt, made, mean, put, yet, next., thank, market, blackberri, mac, never, plastic, right, sell, pleas, guy, scanner, want, black, free, innov, via, come, good, let, stop, color, compani, know, price, just, store, ios7, peopl, releas, say, anyon, devic, need, back, googl, itun, microsoft, will, appl, new, iphone5, ipad"
## + Fold1: cp=0.02381 
## - Fold1: cp=0.02381 
## + Fold2: cp=0.02381 
## - Fold2: cp=0.02381 
## + Fold3: cp=0.02381 
## - Fold3: cp=0.02381 
## Aggregating results
## Selecting tuning parameters
## Fitting cp = 0.0238 on full training set
```

```
## Warning in myfit_mdl(model_id = paste0(model_id_pfx, ""), model_method =
## method, : model's bestTune found at an extreme of tuneGrid for parameter:
## cp
```

![](Apple_Tweets_files/figure-html/fit.models_0-49.png) ![](Apple_Tweets_files/figure-html/fit.models_0-50.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 825 
## 
##           CP nsplit rel error
## 1 0.21428571      0 1.0000000
## 2 0.04761905      1 0.7857143
## 3 0.02380952      2 0.7380952
## 
## Variable importance
## freak  hate 
##    81    19 
## 
## Node number 1: 825 observations,    complexity param=0.2142857
##   predicted class=N  expected loss=0.1527273  P(node) =1
##     class counts:   699   126
##    probabilities: 0.847 0.153 
##   left son=2 (790 obs) right son=3 (35 obs)
##   Primary splits:
##       freak  < 0.5 to the left,  improve=39.275110, (0 missing)
##       hate   < 0.5 to the left,  improve= 9.873907, (0 missing)
##       stuff  < 0.5 to the left,  improve= 6.063648, (0 missing)
##       pictur < 0.5 to the left,  improve= 5.763645, (0 missing)
##       wtf    < 0.5 to the left,  improve= 5.763645, (0 missing)
## 
## Node number 2: 790 observations,    complexity param=0.04761905
##   predicted class=N  expected loss=0.1202532  P(node) =0.9575758
##     class counts:   695    95
##    probabilities: 0.880 0.120 
##   left son=4 (780 obs) right son=5 (10 obs)
##   Primary splits:
##       hate  < 0.5 to the left,  improve=9.359591, (0 missing)
##       wtf   < 0.5 to the left,  improve=6.410211, (0 missing)
##       stuff < 0.5 to the left,  improve=5.436150, (0 missing)
##       fix   < 0.5 to the left,  improve=2.875307, (0 missing)
##       cant  < 0.5 to the left,  improve=2.709704, (0 missing)
## 
## Node number 3: 35 observations
##   predicted class=Y  expected loss=0.1142857  P(node) =0.04242424
##     class counts:     4    31
##    probabilities: 0.114 0.886 
## 
## Node number 4: 780 observations
##   predicted class=N  expected loss=0.1115385  P(node) =0.9454545
##     class counts:   693    87
##    probabilities: 0.888 0.112 
## 
## Node number 5: 10 observations
##   predicted class=Y  expected loss=0.2  P(node) =0.01212121
##     class counts:     2     8
##    probabilities: 0.200 0.800 
## 
## n= 825 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 825 126 N (0.8472727 0.1527273)  
##   2) freak< 0.5 790  95 N (0.8797468 0.1202532)  
##     4) hate< 0.5 780  87 N (0.8884615 0.1115385) *
##     5) hate>=0.5 10   2 Y (0.2000000 0.8000000) *
##   3) freak>=0.5 35   4 Y (0.1142857 0.8857143) *
```

![](Apple_Tweets_files/figure-html/fit.models_0-51.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 699 126
##   Negative.fctr Negative.fctr.predict.Conditional.X.rpart.N
## 1             N                                           0
## 2             Y                                           0
##   Negative.fctr.predict.Conditional.X.rpart.Y
## 1                                         699
## 2                                         126
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 699 126
##   Negative.fctr Negative.fctr.predict.Conditional.X.rpart.N
## 1             N                                           0
## 2             Y                                           0
##   Negative.fctr.predict.Conditional.X.rpart.Y
## 1                                         699
## 2                                         126
##           Reference
## Prediction   N   Y
##          N 693  87
##          Y   6  39
##   Negative.fctr Negative.fctr.predict.Conditional.X.rpart.N
## 1             N                                         693
## 2             Y                                          87
##   Negative.fctr.predict.Conditional.X.rpart.Y
## 1                                           6
## 2                                          39
##           Reference
## Prediction   N   Y
##          N 693  87
##          Y   6  39
##   Negative.fctr Negative.fctr.predict.Conditional.X.rpart.N
## 1             N                                         693
## 2             Y                                          87
##   Negative.fctr.predict.Conditional.X.rpart.Y
## 1                                           6
## 2                                          39
##           Reference
## Prediction   N   Y
##          N 693  87
##          Y   6  39
##   Negative.fctr Negative.fctr.predict.Conditional.X.rpart.N
## 1             N                                         693
## 2             Y                                          87
##   Negative.fctr.predict.Conditional.X.rpart.Y
## 1                                           6
## 2                                          39
##           Reference
## Prediction   N   Y
##          N 693  87
##          Y   6  39
##   Negative.fctr Negative.fctr.predict.Conditional.X.rpart.N
## 1             N                                         693
## 2             Y                                          87
##   Negative.fctr.predict.Conditional.X.rpart.Y
## 1                                           6
## 2                                          39
##           Reference
## Prediction   N   Y
##          N 693  87
##          Y   6  39
##   Negative.fctr Negative.fctr.predict.Conditional.X.rpart.N
## 1             N                                         693
## 2             Y                                          87
##   Negative.fctr.predict.Conditional.X.rpart.Y
## 1                                           6
## 2                                          39
##           Reference
## Prediction   N   Y
##          N 693  87
##          Y   6  39
##   Negative.fctr Negative.fctr.predict.Conditional.X.rpart.N
## 1             N                                         693
## 2             Y                                          87
##   Negative.fctr.predict.Conditional.X.rpart.Y
## 1                                           6
## 2                                          39
##           Reference
## Prediction   N   Y
##          N 693  87
##          Y   6  39
##   Negative.fctr Negative.fctr.predict.Conditional.X.rpart.N
## 1             N                                         693
## 2             Y                                          87
##   Negative.fctr.predict.Conditional.X.rpart.Y
## 1                                           6
## 2                                          39
##           Reference
## Prediction   N   Y
##          N 699 126
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Conditional.X.rpart.N
## 1             N                                         699
## 2             Y                                         126
##   Negative.fctr.predict.Conditional.X.rpart.Y
## 1                                           0
## 2                                           0
##           Reference
## Prediction   N   Y
##          N 699 126
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Conditional.X.rpart.N
## 1             N                                         699
## 2             Y                                         126
##   Negative.fctr.predict.Conditional.X.rpart.Y
## 1                                           0
## 2                                           0
##    threshold   f.score
## 1        0.0 0.2649842
## 2        0.1 0.2649842
## 3        0.2 0.4561404
## 4        0.3 0.4561404
## 5        0.4 0.4561404
## 6        0.5 0.4561404
## 7        0.6 0.4561404
## 8        0.7 0.4561404
## 9        0.8 0.4561404
## 10       0.9 0.0000000
## 11       1.0 0.0000000
```

![](Apple_Tweets_files/figure-html/fit.models_0-52.png) 

```
## [1] "Classifier Probability Threshold: 0.8000 to maximize f.score.fit"
##   Negative.fctr Negative.fctr.predict.Conditional.X.rpart.N
## 1             N                                         693
## 2             Y                                          87
##   Negative.fctr.predict.Conditional.X.rpart.Y
## 1                                           6
## 2                                          39
##           Reference
## Prediction   N   Y
##          N 693  87
##          Y   6  39
##   Negative.fctr Negative.fctr.predict.Conditional.X.rpart.N
## 1             N                                         693
## 2             Y                                          87
##   Negative.fctr.predict.Conditional.X.rpart.Y
## 1                                           6
## 2                                          39
##          Prediction
## Reference   N   Y
##         N 693   6
##         Y  87  39
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.872727e-01   4.086022e-01   8.636885e-01   9.080451e-01   8.472727e-01 
## AccuracyPValue  McnemarPValue 
##   5.582158e-04   1.080262e-16
```

![](Apple_Tweets_files/figure-html/fit.models_0-53.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 300  56
##   Negative.fctr Negative.fctr.predict.Conditional.X.rpart.N
## 1             N                                           0
## 2             Y                                           0
##   Negative.fctr.predict.Conditional.X.rpart.Y
## 1                                         300
## 2                                          56
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 300  56
##   Negative.fctr Negative.fctr.predict.Conditional.X.rpart.N
## 1             N                                           0
## 2             Y                                           0
##   Negative.fctr.predict.Conditional.X.rpart.Y
## 1                                         300
## 2                                          56
##           Reference
## Prediction   N   Y
##          N 296  36
##          Y   4  20
##   Negative.fctr Negative.fctr.predict.Conditional.X.rpart.N
## 1             N                                         296
## 2             Y                                          36
##   Negative.fctr.predict.Conditional.X.rpart.Y
## 1                                           4
## 2                                          20
##           Reference
## Prediction   N   Y
##          N 296  36
##          Y   4  20
##   Negative.fctr Negative.fctr.predict.Conditional.X.rpart.N
## 1             N                                         296
## 2             Y                                          36
##   Negative.fctr.predict.Conditional.X.rpart.Y
## 1                                           4
## 2                                          20
##           Reference
## Prediction   N   Y
##          N 296  36
##          Y   4  20
##   Negative.fctr Negative.fctr.predict.Conditional.X.rpart.N
## 1             N                                         296
## 2             Y                                          36
##   Negative.fctr.predict.Conditional.X.rpart.Y
## 1                                           4
## 2                                          20
##           Reference
## Prediction   N   Y
##          N 296  36
##          Y   4  20
##   Negative.fctr Negative.fctr.predict.Conditional.X.rpart.N
## 1             N                                         296
## 2             Y                                          36
##   Negative.fctr.predict.Conditional.X.rpart.Y
## 1                                           4
## 2                                          20
##           Reference
## Prediction   N   Y
##          N 296  36
##          Y   4  20
##   Negative.fctr Negative.fctr.predict.Conditional.X.rpart.N
## 1             N                                         296
## 2             Y                                          36
##   Negative.fctr.predict.Conditional.X.rpart.Y
## 1                                           4
## 2                                          20
##           Reference
## Prediction   N   Y
##          N 296  36
##          Y   4  20
##   Negative.fctr Negative.fctr.predict.Conditional.X.rpart.N
## 1             N                                         296
## 2             Y                                          36
##   Negative.fctr.predict.Conditional.X.rpart.Y
## 1                                           4
## 2                                          20
##           Reference
## Prediction   N   Y
##          N 296  36
##          Y   4  20
##   Negative.fctr Negative.fctr.predict.Conditional.X.rpart.N
## 1             N                                         296
## 2             Y                                          36
##   Negative.fctr.predict.Conditional.X.rpart.Y
## 1                                           4
## 2                                          20
##           Reference
## Prediction   N   Y
##          N 300  56
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Conditional.X.rpart.N
## 1             N                                         300
## 2             Y                                          56
##   Negative.fctr.predict.Conditional.X.rpart.Y
## 1                                           0
## 2                                           0
##           Reference
## Prediction   N   Y
##          N 300  56
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Conditional.X.rpart.N
## 1             N                                         300
## 2             Y                                          56
##   Negative.fctr.predict.Conditional.X.rpart.Y
## 1                                           0
## 2                                           0
##    threshold   f.score
## 1        0.0 0.2718447
## 2        0.1 0.2718447
## 3        0.2 0.5000000
## 4        0.3 0.5000000
## 5        0.4 0.5000000
## 6        0.5 0.5000000
## 7        0.6 0.5000000
## 8        0.7 0.5000000
## 9        0.8 0.5000000
## 10       0.9 0.0000000
## 11       1.0 0.0000000
```

![](Apple_Tweets_files/figure-html/fit.models_0-54.png) 

```
## [1] "Classifier Probability Threshold: 0.8000 to maximize f.score.OOB"
##   Negative.fctr Negative.fctr.predict.Conditional.X.rpart.N
## 1             N                                         296
## 2             Y                                          36
##   Negative.fctr.predict.Conditional.X.rpart.Y
## 1                                           4
## 2                                          20
##           Reference
## Prediction   N   Y
##          N 296  36
##          Y   4  20
##   Negative.fctr Negative.fctr.predict.Conditional.X.rpart.N
## 1             N                                         296
## 2             Y                                          36
##   Negative.fctr.predict.Conditional.X.rpart.Y
## 1                                           4
## 2                                          20
##          Prediction
## Reference   N   Y
##         N 296   4
##         Y  36  20
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.876404e-01   4.478908e-01   8.501517e-01   9.185050e-01   8.426966e-01 
## AccuracyPValue  McnemarPValue 
##   9.756283e-03   9.509294e-07 
##              model_id model_method
## 1 Conditional.X.rpart        rpart
##                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              feats
## 1 freak, hate, stuff, pictur, wtf, cant, shame, stupid, even, line, yooo, better, ever, fix, charger, still, charg, disappoint, short, like, amazon, yall, break., imessag, stand, togeth, cheap, wont, make, carbon, darn, httpbitly18xc8dk, dear, facebook, X7evenstarz, condom, femal, money, theyr, batteri, china, turn, hope, life, sinc, steve, switch, everi, last, your, amaz, arent, date, divulg, ill, ive, lost, noth, worst, phone, care, way, year, updat, app, iphon, data, take, card, custom, die, event, problem, refus, two, use, screen, thing, buy, tri, cdp, doesnt, emiss, feel, fun, got, hour, macbook, miss, siri, start, upgrad, get, think, copi, guess, person, smart, best, differ, nsa, product, twitter, old, see, download, idea, job, mani, simpl, soon, team, technolog, wow, much, support, text, time, android, that, fingerprint, now, ask, colour, gonna, happen, man, give, hey, today, case, chang, one, work, day, nokia, preorder, look, awesom, done, featur, said, secur, seem, smartphon, thought, what, windowsphon, bit, realli, can, sure, dont, impress, call, didnt, drop, fail, first, improv, isnt, made, mean, put, yet, next., thank, market, blackberri, mac, never, plastic, right, sell, pleas, guy, scanner, want, black, free, innov, via, come, good, let, stop, color, compani, know, price, just, store, ios7, peopl, releas, say, anyon, devic, need, back, googl, itun, microsoft, will, appl, new, iphone5, ipad
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               3                      3.532                 0.435
##   max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1   0.6506404                    0.8       0.4561404        0.8824242
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.8636885             0.9080451     0.3699305    0.672619
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.8             0.5        0.8876404
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.8501517              0.918505     0.4478908
##   max.AccuracySD.fit max.KappaSD.fit
## 1         0.01469619       0.1001379
## [1] "fitting model: Conditional.X.cp.0.rpart"
## [1] "    indep_vars: freak, hate, stuff, pictur, wtf, cant, shame, stupid, even, line, yooo, better, ever, fix, charger, still, charg, disappoint, short, like, amazon, yall, break., imessag, stand, togeth, cheap, wont, make, carbon, darn, httpbitly18xc8dk, dear, facebook, X7evenstarz, condom, femal, money, theyr, batteri, china, turn, hope, life, sinc, steve, switch, everi, last, your, amaz, arent, date, divulg, ill, ive, lost, noth, worst, phone, care, way, year, updat, app, iphon, data, take, card, custom, die, event, problem, refus, two, use, screen, thing, buy, tri, cdp, doesnt, emiss, feel, fun, got, hour, macbook, miss, siri, start, upgrad, get, think, copi, guess, person, smart, best, differ, nsa, product, twitter, old, see, download, idea, job, mani, simpl, soon, team, technolog, wow, much, support, text, time, android, that, fingerprint, now, ask, colour, gonna, happen, man, give, hey, today, case, chang, one, work, day, nokia, preorder, look, awesom, done, featur, said, secur, seem, smartphon, thought, what, windowsphon, bit, realli, can, sure, dont, impress, call, didnt, drop, fail, first, improv, isnt, made, mean, put, yet, next., thank, market, blackberri, mac, never, plastic, right, sell, pleas, guy, scanner, want, black, free, innov, via, come, good, let, stop, color, compani, know, price, just, store, ios7, peopl, releas, say, anyon, devic, need, back, googl, itun, microsoft, will, appl, new, iphone5, ipad"
## Fitting cp = 0 on full training set
```

![](Apple_Tweets_files/figure-html/fit.models_0-55.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 825 
## 
##           CP nsplit rel error
## 1 0.21428571      0 1.0000000
## 2 0.04761905      1 0.7857143
## 3 0.02380952      2 0.7380952
## 4 0.00000000      3 0.7142857
## 
## Variable importance
## freak  hate stuff 
##    72    17    10 
## 
## Node number 1: 825 observations,    complexity param=0.2142857
##   predicted class=N  expected loss=0.1527273  P(node) =1
##     class counts:   699   126
##    probabilities: 0.847 0.153 
##   left son=2 (790 obs) right son=3 (35 obs)
##   Primary splits:
##       freak  < 0.5 to the left,  improve=39.275110, (0 missing)
##       hate   < 0.5 to the left,  improve= 9.873907, (0 missing)
##       stuff  < 0.5 to the left,  improve= 6.063648, (0 missing)
##       pictur < 0.5 to the left,  improve= 5.763645, (0 missing)
##       wtf    < 0.5 to the left,  improve= 5.763645, (0 missing)
## 
## Node number 2: 790 observations,    complexity param=0.04761905
##   predicted class=N  expected loss=0.1202532  P(node) =0.9575758
##     class counts:   695    95
##    probabilities: 0.880 0.120 
##   left son=4 (780 obs) right son=5 (10 obs)
##   Primary splits:
##       hate  < 0.5 to the left,  improve=9.359591, (0 missing)
##       wtf   < 0.5 to the left,  improve=6.410211, (0 missing)
##       stuff < 0.5 to the left,  improve=5.436150, (0 missing)
##       fix   < 0.5 to the left,  improve=2.875307, (0 missing)
##       cant  < 0.5 to the left,  improve=2.709704, (0 missing)
## 
## Node number 3: 35 observations
##   predicted class=Y  expected loss=0.1142857  P(node) =0.04242424
##     class counts:     4    31
##    probabilities: 0.114 0.886 
## 
## Node number 4: 780 observations,    complexity param=0.02380952
##   predicted class=N  expected loss=0.1115385  P(node) =0.9454545
##     class counts:   693    87
##    probabilities: 0.888 0.112 
##   left son=8 (771 obs) right son=9 (9 obs)
##   Primary splits:
##       stuff  < 0.5 to the left,  improve=5.611763, (0 missing)
##       wtf    < 0.5 to the left,  improve=5.132319, (0 missing)
##       fix    < 0.5 to the left,  improve=2.987798, (0 missing)
##       cant   < 0.5 to the left,  improve=2.865713, (0 missing)
##       better < 0.5 to the left,  improve=2.865713, (0 missing)
## 
## Node number 5: 10 observations
##   predicted class=Y  expected loss=0.2  P(node) =0.01212121
##     class counts:     2     8
##    probabilities: 0.200 0.800 
## 
## Node number 8: 771 observations
##   predicted class=N  expected loss=0.1050584  P(node) =0.9345455
##     class counts:   690    81
##    probabilities: 0.895 0.105 
## 
## Node number 9: 9 observations
##   predicted class=Y  expected loss=0.3333333  P(node) =0.01090909
##     class counts:     3     6
##    probabilities: 0.333 0.667 
## 
## n= 825 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 825 126 N (0.8472727 0.1527273)  
##   2) freak< 0.5 790  95 N (0.8797468 0.1202532)  
##     4) hate< 0.5 780  87 N (0.8884615 0.1115385)  
##       8) stuff< 0.5 771  81 N (0.8949416 0.1050584) *
##       9) stuff>=0.5 9   3 Y (0.3333333 0.6666667) *
##     5) hate>=0.5 10   2 Y (0.2000000 0.8000000) *
##   3) freak>=0.5 35   4 Y (0.1142857 0.8857143) *
```

![](Apple_Tweets_files/figure-html/fit.models_0-56.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 699 126
##   Negative.fctr Negative.fctr.predict.Conditional.X.cp.0.rpart.N
## 1             N                                                0
## 2             Y                                                0
##   Negative.fctr.predict.Conditional.X.cp.0.rpart.Y
## 1                                              699
## 2                                              126
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 699 126
##   Negative.fctr Negative.fctr.predict.Conditional.X.cp.0.rpart.N
## 1             N                                                0
## 2             Y                                                0
##   Negative.fctr.predict.Conditional.X.cp.0.rpart.Y
## 1                                              699
## 2                                              126
##           Reference
## Prediction   N   Y
##          N 690  81
##          Y   9  45
##   Negative.fctr Negative.fctr.predict.Conditional.X.cp.0.rpart.N
## 1             N                                              690
## 2             Y                                               81
##   Negative.fctr.predict.Conditional.X.cp.0.rpart.Y
## 1                                                9
## 2                                               45
##           Reference
## Prediction   N   Y
##          N 690  81
##          Y   9  45
##   Negative.fctr Negative.fctr.predict.Conditional.X.cp.0.rpart.N
## 1             N                                              690
## 2             Y                                               81
##   Negative.fctr.predict.Conditional.X.cp.0.rpart.Y
## 1                                                9
## 2                                               45
##           Reference
## Prediction   N   Y
##          N 690  81
##          Y   9  45
##   Negative.fctr Negative.fctr.predict.Conditional.X.cp.0.rpart.N
## 1             N                                              690
## 2             Y                                               81
##   Negative.fctr.predict.Conditional.X.cp.0.rpart.Y
## 1                                                9
## 2                                               45
##           Reference
## Prediction   N   Y
##          N 690  81
##          Y   9  45
##   Negative.fctr Negative.fctr.predict.Conditional.X.cp.0.rpart.N
## 1             N                                              690
## 2             Y                                               81
##   Negative.fctr.predict.Conditional.X.cp.0.rpart.Y
## 1                                                9
## 2                                               45
##           Reference
## Prediction   N   Y
##          N 690  81
##          Y   9  45
##   Negative.fctr Negative.fctr.predict.Conditional.X.cp.0.rpart.N
## 1             N                                              690
## 2             Y                                               81
##   Negative.fctr.predict.Conditional.X.cp.0.rpart.Y
## 1                                                9
## 2                                               45
##           Reference
## Prediction   N   Y
##          N 693  87
##          Y   6  39
##   Negative.fctr Negative.fctr.predict.Conditional.X.cp.0.rpart.N
## 1             N                                              693
## 2             Y                                               87
##   Negative.fctr.predict.Conditional.X.cp.0.rpart.Y
## 1                                                6
## 2                                               39
##           Reference
## Prediction   N   Y
##          N 693  87
##          Y   6  39
##   Negative.fctr Negative.fctr.predict.Conditional.X.cp.0.rpart.N
## 1             N                                              693
## 2             Y                                               87
##   Negative.fctr.predict.Conditional.X.cp.0.rpart.Y
## 1                                                6
## 2                                               39
##           Reference
## Prediction   N   Y
##          N 699 126
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Conditional.X.cp.0.rpart.N
## 1             N                                              699
## 2             Y                                              126
##   Negative.fctr.predict.Conditional.X.cp.0.rpart.Y
## 1                                                0
## 2                                                0
##           Reference
## Prediction   N   Y
##          N 699 126
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Conditional.X.cp.0.rpart.N
## 1             N                                              699
## 2             Y                                              126
##   Negative.fctr.predict.Conditional.X.cp.0.rpart.Y
## 1                                                0
## 2                                                0
##    threshold   f.score
## 1        0.0 0.2649842
## 2        0.1 0.2649842
## 3        0.2 0.5000000
## 4        0.3 0.5000000
## 5        0.4 0.5000000
## 6        0.5 0.5000000
## 7        0.6 0.5000000
## 8        0.7 0.4561404
## 9        0.8 0.4561404
## 10       0.9 0.0000000
## 11       1.0 0.0000000
```

![](Apple_Tweets_files/figure-html/fit.models_0-57.png) 

```
## [1] "Classifier Probability Threshold: 0.6000 to maximize f.score.fit"
##   Negative.fctr Negative.fctr.predict.Conditional.X.cp.0.rpart.N
## 1             N                                              690
## 2             Y                                               81
##   Negative.fctr.predict.Conditional.X.cp.0.rpart.Y
## 1                                                9
## 2                                               45
##           Reference
## Prediction   N   Y
##          N 690  81
##          Y   9  45
##   Negative.fctr Negative.fctr.predict.Conditional.X.cp.0.rpart.N
## 1             N                                              690
## 2             Y                                               81
##   Negative.fctr.predict.Conditional.X.cp.0.rpart.Y
## 1                                                9
## 2                                               45
##          Prediction
## Reference   N   Y
##         N 690   9
##         Y  81  45
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.909091e-01   4.495596e-01   8.676171e-01   9.113614e-01   8.472727e-01 
## AccuracyPValue  McnemarPValue 
##   1.744490e-04   7.206261e-14
```

![](Apple_Tweets_files/figure-html/fit.models_0-58.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 300  56
##   Negative.fctr Negative.fctr.predict.Conditional.X.cp.0.rpart.N
## 1             N                                                0
## 2             Y                                                0
##   Negative.fctr.predict.Conditional.X.cp.0.rpart.Y
## 1                                              300
## 2                                               56
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 300  56
##   Negative.fctr Negative.fctr.predict.Conditional.X.cp.0.rpart.N
## 1             N                                                0
## 2             Y                                                0
##   Negative.fctr.predict.Conditional.X.cp.0.rpart.Y
## 1                                              300
## 2                                               56
##           Reference
## Prediction   N   Y
##          N 296  34
##          Y   4  22
##   Negative.fctr Negative.fctr.predict.Conditional.X.cp.0.rpart.N
## 1             N                                              296
## 2             Y                                               34
##   Negative.fctr.predict.Conditional.X.cp.0.rpart.Y
## 1                                                4
## 2                                               22
##           Reference
## Prediction   N   Y
##          N 296  34
##          Y   4  22
##   Negative.fctr Negative.fctr.predict.Conditional.X.cp.0.rpart.N
## 1             N                                              296
## 2             Y                                               34
##   Negative.fctr.predict.Conditional.X.cp.0.rpart.Y
## 1                                                4
## 2                                               22
##           Reference
## Prediction   N   Y
##          N 296  34
##          Y   4  22
##   Negative.fctr Negative.fctr.predict.Conditional.X.cp.0.rpart.N
## 1             N                                              296
## 2             Y                                               34
##   Negative.fctr.predict.Conditional.X.cp.0.rpart.Y
## 1                                                4
## 2                                               22
##           Reference
## Prediction   N   Y
##          N 296  34
##          Y   4  22
##   Negative.fctr Negative.fctr.predict.Conditional.X.cp.0.rpart.N
## 1             N                                              296
## 2             Y                                               34
##   Negative.fctr.predict.Conditional.X.cp.0.rpart.Y
## 1                                                4
## 2                                               22
##           Reference
## Prediction   N   Y
##          N 296  34
##          Y   4  22
##   Negative.fctr Negative.fctr.predict.Conditional.X.cp.0.rpart.N
## 1             N                                              296
## 2             Y                                               34
##   Negative.fctr.predict.Conditional.X.cp.0.rpart.Y
## 1                                                4
## 2                                               22
##           Reference
## Prediction   N   Y
##          N 296  36
##          Y   4  20
##   Negative.fctr Negative.fctr.predict.Conditional.X.cp.0.rpart.N
## 1             N                                              296
## 2             Y                                               36
##   Negative.fctr.predict.Conditional.X.cp.0.rpart.Y
## 1                                                4
## 2                                               20
##           Reference
## Prediction   N   Y
##          N 296  36
##          Y   4  20
##   Negative.fctr Negative.fctr.predict.Conditional.X.cp.0.rpart.N
## 1             N                                              296
## 2             Y                                               36
##   Negative.fctr.predict.Conditional.X.cp.0.rpart.Y
## 1                                                4
## 2                                               20
##           Reference
## Prediction   N   Y
##          N 300  56
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Conditional.X.cp.0.rpart.N
## 1             N                                              300
## 2             Y                                               56
##   Negative.fctr.predict.Conditional.X.cp.0.rpart.Y
## 1                                                0
## 2                                                0
##           Reference
## Prediction   N   Y
##          N 300  56
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Conditional.X.cp.0.rpart.N
## 1             N                                              300
## 2             Y                                               56
##   Negative.fctr.predict.Conditional.X.cp.0.rpart.Y
## 1                                                0
## 2                                                0
##    threshold   f.score
## 1        0.0 0.2718447
## 2        0.1 0.2718447
## 3        0.2 0.5365854
## 4        0.3 0.5365854
## 5        0.4 0.5365854
## 6        0.5 0.5365854
## 7        0.6 0.5365854
## 8        0.7 0.5000000
## 9        0.8 0.5000000
## 10       0.9 0.0000000
## 11       1.0 0.0000000
```

```
## [1] "Classifier Probability Threshold: 0.6000 to maximize f.score.OOB"
##   Negative.fctr Negative.fctr.predict.Conditional.X.cp.0.rpart.N
## 1             N                                              296
## 2             Y                                               34
##   Negative.fctr.predict.Conditional.X.cp.0.rpart.Y
## 1                                                4
## 2                                               22
##           Reference
## Prediction   N   Y
##          N 296  34
##          Y   4  22
##   Negative.fctr Negative.fctr.predict.Conditional.X.cp.0.rpart.N
## 1             N                                              296
## 2             Y                                               34
##   Negative.fctr.predict.Conditional.X.cp.0.rpart.Y
## 1                                                4
## 2                                               22
##          Prediction
## Reference   N   Y
##         N 296   4
##         Y  34  22
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.932584e-01   4.852359e-01   8.564403e-01   9.233486e-01   8.426966e-01 
## AccuracyPValue  McnemarPValue 
##   3.956222e-03   2.545872e-06 
##                   model_id model_method
## 1 Conditional.X.cp.0.rpart        rpart
##                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              feats
## 1 freak, hate, stuff, pictur, wtf, cant, shame, stupid, even, line, yooo, better, ever, fix, charger, still, charg, disappoint, short, like, amazon, yall, break., imessag, stand, togeth, cheap, wont, make, carbon, darn, httpbitly18xc8dk, dear, facebook, X7evenstarz, condom, femal, money, theyr, batteri, china, turn, hope, life, sinc, steve, switch, everi, last, your, amaz, arent, date, divulg, ill, ive, lost, noth, worst, phone, care, way, year, updat, app, iphon, data, take, card, custom, die, event, problem, refus, two, use, screen, thing, buy, tri, cdp, doesnt, emiss, feel, fun, got, hour, macbook, miss, siri, start, upgrad, get, think, copi, guess, person, smart, best, differ, nsa, product, twitter, old, see, download, idea, job, mani, simpl, soon, team, technolog, wow, much, support, text, time, android, that, fingerprint, now, ask, colour, gonna, happen, man, give, hey, today, case, chang, one, work, day, nokia, preorder, look, awesom, done, featur, said, secur, seem, smartphon, thought, what, windowsphon, bit, realli, can, sure, dont, impress, call, didnt, drop, fail, first, improv, isnt, made, mean, put, yet, next., thank, market, blackberri, mac, never, plastic, right, sell, pleas, guy, scanner, want, black, free, innov, via, come, good, let, stop, color, compani, know, price, just, store, ios7, peopl, releas, say, anyon, devic, need, back, googl, itun, microsoft, will, appl, new, iphone5, ipad
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               0                      0.932                 0.429
##   max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1   0.6727638                    0.6             0.5        0.8909091
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.8676171             0.9113614     0.4495596   0.6902381
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.6       0.5365854        0.8932584
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.8564403             0.9233486     0.4852359
## [1] "fitting model: Conditional.X.rf"
## [1] "    indep_vars: freak, hate, stuff, pictur, wtf, cant, shame, stupid, even, line, yooo, better, ever, fix, charger, still, charg, disappoint, short, like, amazon, yall, break., imessag, stand, togeth, cheap, wont, make, carbon, darn, httpbitly18xc8dk, dear, facebook, X7evenstarz, condom, femal, money, theyr, batteri, china, turn, hope, life, sinc, steve, switch, everi, last, your, amaz, arent, date, divulg, ill, ive, lost, noth, worst, phone, care, way, year, updat, app, iphon, data, take, card, custom, die, event, problem, refus, two, use, screen, thing, buy, tri, cdp, doesnt, emiss, feel, fun, got, hour, macbook, miss, siri, start, upgrad, get, think, copi, guess, person, smart, best, differ, nsa, product, twitter, old, see, download, idea, job, mani, simpl, soon, team, technolog, wow, much, support, text, time, android, that, fingerprint, now, ask, colour, gonna, happen, man, give, hey, today, case, chang, one, work, day, nokia, preorder, look, awesom, done, featur, said, secur, seem, smartphon, thought, what, windowsphon, bit, realli, can, sure, dont, impress, call, didnt, drop, fail, first, improv, isnt, made, mean, put, yet, next., thank, market, blackberri, mac, never, plastic, right, sell, pleas, guy, scanner, .rnorm, want, black, free, innov, via, come, good, let, stop, color, compani, know, price, just, store, ios7, peopl, releas, say, anyon, devic, need, back, googl, itun, microsoft, will, appl, new, iphone5, ipad"
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

![](Apple_Tweets_files/figure-html/fit.models_0-59.png) 

```
## + : mtry=  2 
## - : mtry=  2 
## + : mtry=105 
## - : mtry=105 
## + : mtry=209 
## - : mtry=209 
## Aggregating results
## Selecting tuning parameters
## Fitting mtry = 105 on full training set
```

![](Apple_Tweets_files/figure-html/fit.models_0-60.png) ![](Apple_Tweets_files/figure-html/fit.models_0-61.png) 

```
##                 Length Class      Mode     
## call               4   -none-     call     
## type               1   -none-     character
## predicted        825   factor     numeric  
## err.rate        1500   -none-     numeric  
## confusion          6   -none-     numeric  
## votes           1650   matrix     numeric  
## oob.times        825   -none-     numeric  
## classes            2   -none-     character
## importance       209   -none-     numeric  
## importanceSD       0   -none-     NULL     
## localImportance    0   -none-     NULL     
## proximity          0   -none-     NULL     
## ntree              1   -none-     numeric  
## mtry               1   -none-     numeric  
## forest            14   -none-     list     
## y                825   factor     numeric  
## test               0   -none-     NULL     
## inbag              0   -none-     NULL     
## xNames           209   -none-     character
## problemType        1   -none-     character
## tuneValue          1   data.frame list     
## obsLevels          2   -none-     character
```

![](Apple_Tweets_files/figure-html/fit.models_0-62.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 699 126
##   Negative.fctr Negative.fctr.predict.Conditional.X.rf.N
## 1             N                                        0
## 2             Y                                        0
##   Negative.fctr.predict.Conditional.X.rf.Y
## 1                                      699
## 2                                      126
##           Reference
## Prediction   N   Y
##          N 617   0
##          Y  82 126
##   Negative.fctr Negative.fctr.predict.Conditional.X.rf.N
## 1             N                                      617
## 2             Y                                        0
##   Negative.fctr.predict.Conditional.X.rf.Y
## 1                                       82
## 2                                      126
##           Reference
## Prediction   N   Y
##          N 676   0
##          Y  23 126
##   Negative.fctr Negative.fctr.predict.Conditional.X.rf.N
## 1             N                                      676
## 2             Y                                        0
##   Negative.fctr.predict.Conditional.X.rf.Y
## 1                                       23
## 2                                      126
##           Reference
## Prediction   N   Y
##          N 689   0
##          Y  10 126
##   Negative.fctr Negative.fctr.predict.Conditional.X.rf.N
## 1             N                                      689
## 2             Y                                        0
##   Negative.fctr.predict.Conditional.X.rf.Y
## 1                                       10
## 2                                      126
##           Reference
## Prediction   N   Y
##          N 697   0
##          Y   2 126
##   Negative.fctr Negative.fctr.predict.Conditional.X.rf.N
## 1             N                                      697
## 2             Y                                        0
##   Negative.fctr.predict.Conditional.X.rf.Y
## 1                                        2
## 2                                      126
##           Reference
## Prediction   N   Y
##          N 699   0
##          Y   0 126
##   Negative.fctr Negative.fctr.predict.Conditional.X.rf.N
## 1             N                                      699
## 2             Y                                        0
##   Negative.fctr.predict.Conditional.X.rf.Y
## 1                                        0
## 2                                      126
##           Reference
## Prediction   N   Y
##          N 699   1
##          Y   0 125
##   Negative.fctr Negative.fctr.predict.Conditional.X.rf.N
## 1             N                                      699
## 2             Y                                        1
##   Negative.fctr.predict.Conditional.X.rf.Y
## 1                                        0
## 2                                      125
##           Reference
## Prediction   N   Y
##          N 699  52
##          Y   0  74
##   Negative.fctr Negative.fctr.predict.Conditional.X.rf.N
## 1             N                                      699
## 2             Y                                       52
##   Negative.fctr.predict.Conditional.X.rf.Y
## 1                                        0
## 2                                       74
##           Reference
## Prediction   N   Y
##          N 699  67
##          Y   0  59
##   Negative.fctr Negative.fctr.predict.Conditional.X.rf.N
## 1             N                                      699
## 2             Y                                       67
##   Negative.fctr.predict.Conditional.X.rf.Y
## 1                                        0
## 2                                       59
##           Reference
## Prediction   N   Y
##          N 699  88
##          Y   0  38
##   Negative.fctr Negative.fctr.predict.Conditional.X.rf.N
## 1             N                                      699
## 2             Y                                       88
##   Negative.fctr.predict.Conditional.X.rf.Y
## 1                                        0
## 2                                       38
##           Reference
## Prediction   N   Y
##          N 699 116
##          Y   0  10
##   Negative.fctr Negative.fctr.predict.Conditional.X.rf.N
## 1             N                                      699
## 2             Y                                      116
##   Negative.fctr.predict.Conditional.X.rf.Y
## 1                                        0
## 2                                       10
##    threshold   f.score
## 1        0.0 0.2649842
## 2        0.1 0.7544910
## 3        0.2 0.9163636
## 4        0.3 0.9618321
## 5        0.4 0.9921260
## 6        0.5 1.0000000
## 7        0.6 0.9960159
## 8        0.7 0.7400000
## 9        0.8 0.6378378
## 10       0.9 0.4634146
## 11       1.0 0.1470588
```

![](Apple_Tweets_files/figure-html/fit.models_0-63.png) 

```
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.fit"
##   Negative.fctr Negative.fctr.predict.Conditional.X.rf.N
## 1             N                                      699
## 2             Y                                       NA
##   Negative.fctr.predict.Conditional.X.rf.Y
## 1                                       NA
## 2                                      126
##           Reference
## Prediction   N   Y
##          N 699   0
##          Y   0 126
##   Negative.fctr Negative.fctr.predict.Conditional.X.rf.N
## 1             N                                      699
## 2             Y                                        0
##   Negative.fctr.predict.Conditional.X.rf.Y
## 1                                        0
## 2                                      126
##          Prediction
## Reference   N   Y
##         N 699   0
##         Y   0 126
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   1.000000e+00   1.000000e+00   9.955386e-01   1.000000e+00   8.472727e-01 
## AccuracyPValue  McnemarPValue 
##   4.160662e-60            NaN
```

![](Apple_Tweets_files/figure-html/fit.models_0-64.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 300  56
##   Negative.fctr Negative.fctr.predict.Conditional.X.rf.N
## 1             N                                        0
## 2             Y                                        0
##   Negative.fctr.predict.Conditional.X.rf.Y
## 1                                      300
## 2                                       56
##           Reference
## Prediction   N   Y
##          N 224  18
##          Y  76  38
##   Negative.fctr Negative.fctr.predict.Conditional.X.rf.N
## 1             N                                      224
## 2             Y                                       18
##   Negative.fctr.predict.Conditional.X.rf.Y
## 1                                       76
## 2                                       38
##           Reference
## Prediction   N   Y
##          N 252  21
##          Y  48  35
##   Negative.fctr Negative.fctr.predict.Conditional.X.rf.N
## 1             N                                      252
## 2             Y                                       21
##   Negative.fctr.predict.Conditional.X.rf.Y
## 1                                       48
## 2                                       35
##           Reference
## Prediction   N   Y
##          N 265  24
##          Y  35  32
##   Negative.fctr Negative.fctr.predict.Conditional.X.rf.N
## 1             N                                      265
## 2             Y                                       24
##   Negative.fctr.predict.Conditional.X.rf.Y
## 1                                       35
## 2                                       32
##           Reference
## Prediction   N   Y
##          N 275  26
##          Y  25  30
##   Negative.fctr Negative.fctr.predict.Conditional.X.rf.N
## 1             N                                      275
## 2             Y                                       26
##   Negative.fctr.predict.Conditional.X.rf.Y
## 1                                       25
## 2                                       30
##           Reference
## Prediction   N   Y
##          N 282  29
##          Y  18  27
##   Negative.fctr Negative.fctr.predict.Conditional.X.rf.N
## 1             N                                      282
## 2             Y                                       29
##   Negative.fctr.predict.Conditional.X.rf.Y
## 1                                       18
## 2                                       27
##           Reference
## Prediction   N   Y
##          N 288  31
##          Y  12  25
##   Negative.fctr Negative.fctr.predict.Conditional.X.rf.N
## 1             N                                      288
## 2             Y                                       31
##   Negative.fctr.predict.Conditional.X.rf.Y
## 1                                       12
## 2                                       25
##           Reference
## Prediction   N   Y
##          N 292  33
##          Y   8  23
##   Negative.fctr Negative.fctr.predict.Conditional.X.rf.N
## 1             N                                      292
## 2             Y                                       33
##   Negative.fctr.predict.Conditional.X.rf.Y
## 1                                        8
## 2                                       23
##           Reference
## Prediction   N   Y
##          N 296  38
##          Y   4  18
##   Negative.fctr Negative.fctr.predict.Conditional.X.rf.N
## 1             N                                      296
## 2             Y                                       38
##   Negative.fctr.predict.Conditional.X.rf.Y
## 1                                        4
## 2                                       18
##           Reference
## Prediction   N   Y
##          N 297  43
##          Y   3  13
##   Negative.fctr Negative.fctr.predict.Conditional.X.rf.N
## 1             N                                      297
## 2             Y                                       43
##   Negative.fctr.predict.Conditional.X.rf.Y
## 1                                        3
## 2                                       13
##           Reference
## Prediction   N   Y
##          N 299  52
##          Y   1   4
##   Negative.fctr Negative.fctr.predict.Conditional.X.rf.N
## 1             N                                      299
## 2             Y                                       52
##   Negative.fctr.predict.Conditional.X.rf.Y
## 1                                        1
## 2                                        4
##    threshold   f.score
## 1        0.0 0.2718447
## 2        0.1 0.4470588
## 3        0.2 0.5035971
## 4        0.3 0.5203252
## 5        0.4 0.5405405
## 6        0.5 0.5346535
## 7        0.6 0.5376344
## 8        0.7 0.5287356
## 9        0.8 0.4615385
## 10       0.9 0.3611111
## 11       1.0 0.1311475
```

![](Apple_Tweets_files/figure-html/fit.models_0-65.png) 

```
## [1] "Classifier Probability Threshold: 0.4000 to maximize f.score.OOB"
##   Negative.fctr Negative.fctr.predict.Conditional.X.rf.N
## 1             N                                      275
## 2             Y                                       26
##   Negative.fctr.predict.Conditional.X.rf.Y
## 1                                       25
## 2                                       30
##           Reference
## Prediction   N   Y
##          N 275  26
##          Y  25  30
##   Negative.fctr Negative.fctr.predict.Conditional.X.rf.N
## 1             N                                      275
## 2             Y                                       26
##   Negative.fctr.predict.Conditional.X.rf.Y
## 1                                       25
## 2                                       30
##          Prediction
## Reference   N   Y
##         N 275  25
##         Y  26  30
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.8567416      0.4556901      0.8159786      0.8914388      0.8426966 
## AccuracyPValue  McnemarPValue 
##      0.2593176      1.0000000 
##           model_id model_method
## 1 Conditional.X.rf           rf
##                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      feats
## 1 freak, hate, stuff, pictur, wtf, cant, shame, stupid, even, line, yooo, better, ever, fix, charger, still, charg, disappoint, short, like, amazon, yall, break., imessag, stand, togeth, cheap, wont, make, carbon, darn, httpbitly18xc8dk, dear, facebook, X7evenstarz, condom, femal, money, theyr, batteri, china, turn, hope, life, sinc, steve, switch, everi, last, your, amaz, arent, date, divulg, ill, ive, lost, noth, worst, phone, care, way, year, updat, app, iphon, data, take, card, custom, die, event, problem, refus, two, use, screen, thing, buy, tri, cdp, doesnt, emiss, feel, fun, got, hour, macbook, miss, siri, start, upgrad, get, think, copi, guess, person, smart, best, differ, nsa, product, twitter, old, see, download, idea, job, mani, simpl, soon, team, technolog, wow, much, support, text, time, android, that, fingerprint, now, ask, colour, gonna, happen, man, give, hey, today, case, chang, one, work, day, nokia, preorder, look, awesom, done, featur, said, secur, seem, smartphon, thought, what, windowsphon, bit, realli, can, sure, dont, impress, call, didnt, drop, fail, first, improv, isnt, made, mean, put, yet, next., thank, market, blackberri, mac, never, plastic, right, sell, pleas, guy, scanner, .rnorm, want, black, free, innov, via, come, good, let, stop, color, compani, know, price, just, store, ios7, peopl, releas, say, anyon, devic, need, back, googl, itun, microsoft, will, appl, new, iphone5, ipad
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               3                     91.638                23.538
##   max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1           1                    0.5               1             0.88
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.9955386                     1     0.4526428   0.7953274
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.4       0.5405405        0.8567416
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.8159786             0.8914388     0.4556901
## [1] "fitting model: Conditional.X.no.rnorm.rf"
## [1] "    indep_vars: freak, hate, stuff, pictur, wtf, cant, shame, stupid, even, line, yooo, better, ever, fix, charger, still, charg, disappoint, short, like, amazon, yall, break., imessag, stand, togeth, cheap, wont, make, carbon, darn, httpbitly18xc8dk, dear, facebook, X7evenstarz, condom, femal, money, theyr, batteri, china, turn, hope, life, sinc, steve, switch, everi, last, your, amaz, arent, date, divulg, ill, ive, lost, noth, worst, phone, care, way, year, updat, app, iphon, data, take, card, custom, die, event, problem, refus, two, use, screen, thing, buy, tri, cdp, doesnt, emiss, feel, fun, got, hour, macbook, miss, siri, start, upgrad, get, think, copi, guess, person, smart, best, differ, nsa, product, twitter, old, see, download, idea, job, mani, simpl, soon, team, technolog, wow, much, support, text, time, android, that, fingerprint, now, ask, colour, gonna, happen, man, give, hey, today, case, chang, one, work, day, nokia, preorder, look, awesom, done, featur, said, secur, seem, smartphon, thought, what, windowsphon, bit, realli, can, sure, dont, impress, call, didnt, drop, fail, first, improv, isnt, made, mean, put, yet, next., thank, market, blackberri, mac, never, plastic, right, sell, pleas, guy, scanner, want, black, free, innov, via, come, good, let, stop, color, compani, know, price, just, store, ios7, peopl, releas, say, anyon, devic, need, back, googl, itun, microsoft, will, appl, new, iphone5, ipad"
## + : mtry=  2 
## - : mtry=  2 
## + : mtry=105 
## - : mtry=105 
## + : mtry=208 
## - : mtry=208 
## Aggregating results
## Selecting tuning parameters
## Fitting mtry = 208 on full training set
```

```
## Warning in myfit_mdl(model_id = paste0(model_id_pfx, ".no.rnorm"),
## model_method = method, : model's bestTune found at an extreme of tuneGrid
## for parameter: mtry
```

![](Apple_Tweets_files/figure-html/fit.models_0-66.png) ![](Apple_Tweets_files/figure-html/fit.models_0-67.png) 

```
##                 Length Class      Mode     
## call               4   -none-     call     
## type               1   -none-     character
## predicted        825   factor     numeric  
## err.rate        1500   -none-     numeric  
## confusion          6   -none-     numeric  
## votes           1650   matrix     numeric  
## oob.times        825   -none-     numeric  
## classes            2   -none-     character
## importance       208   -none-     numeric  
## importanceSD       0   -none-     NULL     
## localImportance    0   -none-     NULL     
## proximity          0   -none-     NULL     
## ntree              1   -none-     numeric  
## mtry               1   -none-     numeric  
## forest            14   -none-     list     
## y                825   factor     numeric  
## test               0   -none-     NULL     
## inbag              0   -none-     NULL     
## xNames           208   -none-     character
## problemType        1   -none-     character
## tuneValue          1   data.frame list     
## obsLevels          2   -none-     character
```

![](Apple_Tweets_files/figure-html/fit.models_0-68.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 699 126
##   Negative.fctr Negative.fctr.predict.Conditional.X.no.rnorm.rf.N
## 1             N                                                 0
## 2             Y                                                 0
##   Negative.fctr.predict.Conditional.X.no.rnorm.rf.Y
## 1                                               699
## 2                                               126
##           Reference
## Prediction   N   Y
##          N 617   6
##          Y  82 120
##   Negative.fctr Negative.fctr.predict.Conditional.X.no.rnorm.rf.N
## 1             N                                               617
## 2             Y                                                 6
##   Negative.fctr.predict.Conditional.X.no.rnorm.rf.Y
## 1                                                82
## 2                                               120
##           Reference
## Prediction   N   Y
##          N 664   6
##          Y  35 120
##   Negative.fctr Negative.fctr.predict.Conditional.X.no.rnorm.rf.N
## 1             N                                               664
## 2             Y                                                 6
##   Negative.fctr.predict.Conditional.X.no.rnorm.rf.Y
## 1                                                35
## 2                                               120
##           Reference
## Prediction   N   Y
##          N 685   6
##          Y  14 120
##   Negative.fctr Negative.fctr.predict.Conditional.X.no.rnorm.rf.N
## 1             N                                               685
## 2             Y                                                 6
##   Negative.fctr.predict.Conditional.X.no.rnorm.rf.Y
## 1                                                14
## 2                                               120
##           Reference
## Prediction   N   Y
##          N 697   7
##          Y   2 119
##   Negative.fctr Negative.fctr.predict.Conditional.X.no.rnorm.rf.N
## 1             N                                               697
## 2             Y                                                 7
##   Negative.fctr.predict.Conditional.X.no.rnorm.rf.Y
## 1                                                 2
## 2                                               119
##           Reference
## Prediction   N   Y
##          N 697   7
##          Y   2 119
##   Negative.fctr Negative.fctr.predict.Conditional.X.no.rnorm.rf.N
## 1             N                                               697
## 2             Y                                                 7
##   Negative.fctr.predict.Conditional.X.no.rnorm.rf.Y
## 1                                                 2
## 2                                               119
##           Reference
## Prediction   N   Y
##          N 698   9
##          Y   1 117
##   Negative.fctr Negative.fctr.predict.Conditional.X.no.rnorm.rf.N
## 1             N                                               698
## 2             Y                                                 9
##   Negative.fctr.predict.Conditional.X.no.rnorm.rf.Y
## 1                                                 1
## 2                                               117
##           Reference
## Prediction   N   Y
##          N 698  52
##          Y   1  74
##   Negative.fctr Negative.fctr.predict.Conditional.X.no.rnorm.rf.N
## 1             N                                               698
## 2             Y                                                52
##   Negative.fctr.predict.Conditional.X.no.rnorm.rf.Y
## 1                                                 1
## 2                                                74
##           Reference
## Prediction   N   Y
##          N 698  65
##          Y   1  61
##   Negative.fctr Negative.fctr.predict.Conditional.X.no.rnorm.rf.N
## 1             N                                               698
## 2             Y                                                65
##   Negative.fctr.predict.Conditional.X.no.rnorm.rf.Y
## 1                                                 1
## 2                                                61
##           Reference
## Prediction   N   Y
##          N 698  89
##          Y   1  37
##   Negative.fctr Negative.fctr.predict.Conditional.X.no.rnorm.rf.N
## 1             N                                               698
## 2             Y                                                89
##   Negative.fctr.predict.Conditional.X.no.rnorm.rf.Y
## 1                                                 1
## 2                                                37
##           Reference
## Prediction   N   Y
##          N 699 120
##          Y   0   6
##   Negative.fctr Negative.fctr.predict.Conditional.X.no.rnorm.rf.N
## 1             N                                               699
## 2             Y                                               120
##   Negative.fctr.predict.Conditional.X.no.rnorm.rf.Y
## 1                                                 0
## 2                                                 6
##    threshold    f.score
## 1        0.0 0.26498423
## 2        0.1 0.73170732
## 3        0.2 0.85409253
## 4        0.3 0.92307692
## 5        0.4 0.96356275
## 6        0.5 0.96356275
## 7        0.6 0.95901639
## 8        0.7 0.73631841
## 9        0.8 0.64893617
## 10       0.9 0.45121951
## 11       1.0 0.09090909
```

![](Apple_Tweets_files/figure-html/fit.models_0-69.png) 

```
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.fit"
##   Negative.fctr Negative.fctr.predict.Conditional.X.no.rnorm.rf.N
## 1             N                                               697
## 2             Y                                                 7
##   Negative.fctr.predict.Conditional.X.no.rnorm.rf.Y
## 1                                                 2
## 2                                               119
##           Reference
## Prediction   N   Y
##          N 697   7
##          Y   2 119
##   Negative.fctr Negative.fctr.predict.Conditional.X.no.rnorm.rf.N
## 1             N                                               697
## 2             Y                                                 7
##   Negative.fctr.predict.Conditional.X.no.rnorm.rf.Y
## 1                                                 2
## 2                                               119
##          Prediction
## Reference   N   Y
##         N 697   2
##         Y   7 119
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   9.890909e-01   9.571510e-01   9.793925e-01   9.949999e-01   8.472727e-01 
## AccuracyPValue  McnemarPValue 
##   4.156156e-46   1.824224e-01
```

![](Apple_Tweets_files/figure-html/fit.models_0-70.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 300  56
##   Negative.fctr Negative.fctr.predict.Conditional.X.no.rnorm.rf.N
## 1             N                                                 0
## 2             Y                                                 0
##   Negative.fctr.predict.Conditional.X.no.rnorm.rf.Y
## 1                                               300
## 2                                                56
##           Reference
## Prediction   N   Y
##          N 238  18
##          Y  62  38
##   Negative.fctr Negative.fctr.predict.Conditional.X.no.rnorm.rf.N
## 1             N                                               238
## 2             Y                                                18
##   Negative.fctr.predict.Conditional.X.no.rnorm.rf.Y
## 1                                                62
## 2                                                38
##           Reference
## Prediction   N   Y
##          N 266  20
##          Y  34  36
##   Negative.fctr Negative.fctr.predict.Conditional.X.no.rnorm.rf.N
## 1             N                                               266
## 2             Y                                                20
##   Negative.fctr.predict.Conditional.X.no.rnorm.rf.Y
## 1                                                34
## 2                                                36
##           Reference
## Prediction   N   Y
##          N 269  23
##          Y  31  33
##   Negative.fctr Negative.fctr.predict.Conditional.X.no.rnorm.rf.N
## 1             N                                               269
## 2             Y                                                23
##   Negative.fctr.predict.Conditional.X.no.rnorm.rf.Y
## 1                                                31
## 2                                                33
##           Reference
## Prediction   N   Y
##          N 275  23
##          Y  25  33
##   Negative.fctr Negative.fctr.predict.Conditional.X.no.rnorm.rf.N
## 1             N                                               275
## 2             Y                                                23
##   Negative.fctr.predict.Conditional.X.no.rnorm.rf.Y
## 1                                                25
## 2                                                33
##           Reference
## Prediction   N   Y
##          N 280  26
##          Y  20  30
##   Negative.fctr Negative.fctr.predict.Conditional.X.no.rnorm.rf.N
## 1             N                                               280
## 2             Y                                                26
##   Negative.fctr.predict.Conditional.X.no.rnorm.rf.Y
## 1                                                20
## 2                                                30
##           Reference
## Prediction   N   Y
##          N 291  31
##          Y   9  25
##   Negative.fctr Negative.fctr.predict.Conditional.X.no.rnorm.rf.N
## 1             N                                               291
## 2             Y                                                31
##   Negative.fctr.predict.Conditional.X.no.rnorm.rf.Y
## 1                                                 9
## 2                                                25
##           Reference
## Prediction   N   Y
##          N 292  32
##          Y   8  24
##   Negative.fctr Negative.fctr.predict.Conditional.X.no.rnorm.rf.N
## 1             N                                               292
## 2             Y                                                32
##   Negative.fctr.predict.Conditional.X.no.rnorm.rf.Y
## 1                                                 8
## 2                                                24
##           Reference
## Prediction   N   Y
##          N 295  37
##          Y   5  19
##   Negative.fctr Negative.fctr.predict.Conditional.X.no.rnorm.rf.N
## 1             N                                               295
## 2             Y                                                37
##   Negative.fctr.predict.Conditional.X.no.rnorm.rf.Y
## 1                                                 5
## 2                                                19
##           Reference
## Prediction   N   Y
##          N 297  37
##          Y   3  19
##   Negative.fctr Negative.fctr.predict.Conditional.X.no.rnorm.rf.N
## 1             N                                               297
## 2             Y                                                37
##   Negative.fctr.predict.Conditional.X.no.rnorm.rf.Y
## 1                                                 3
## 2                                                19
##           Reference
## Prediction   N   Y
##          N 300  55
##          Y   0   1
##   Negative.fctr Negative.fctr.predict.Conditional.X.no.rnorm.rf.N
## 1             N                                               300
## 2             Y                                                55
##   Negative.fctr.predict.Conditional.X.no.rnorm.rf.Y
## 1                                                 0
## 2                                                 1
##    threshold    f.score
## 1        0.0 0.27184466
## 2        0.1 0.48717949
## 3        0.2 0.57142857
## 4        0.3 0.55000000
## 5        0.4 0.57894737
## 6        0.5 0.56603774
## 7        0.6 0.55555556
## 8        0.7 0.54545455
## 9        0.8 0.47500000
## 10       0.9 0.48717949
## 11       1.0 0.03508772
```

![](Apple_Tweets_files/figure-html/fit.models_0-71.png) 

```
## [1] "Classifier Probability Threshold: 0.4000 to maximize f.score.OOB"
##   Negative.fctr Negative.fctr.predict.Conditional.X.no.rnorm.rf.N
## 1             N                                               275
## 2             Y                                                23
##   Negative.fctr.predict.Conditional.X.no.rnorm.rf.Y
## 1                                                25
## 2                                                33
##           Reference
## Prediction   N   Y
##          N 275  23
##          Y  25  33
##   Negative.fctr Negative.fctr.predict.Conditional.X.no.rnorm.rf.N
## 1             N                                               275
## 2             Y                                                23
##   Negative.fctr.predict.Conditional.X.no.rnorm.rf.Y
## 1                                                25
## 2                                                33
##          Prediction
## Reference   N   Y
##         N 275  25
##         Y  23  33
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.8651685      0.4987092      0.8252354      0.8988853      0.8426966 
## AccuracyPValue  McnemarPValue 
##      0.1366213      0.8852339 
##                    model_id model_method
## 1 Conditional.X.no.rnorm.rf           rf
##                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              feats
## 1 freak, hate, stuff, pictur, wtf, cant, shame, stupid, even, line, yooo, better, ever, fix, charger, still, charg, disappoint, short, like, amazon, yall, break., imessag, stand, togeth, cheap, wont, make, carbon, darn, httpbitly18xc8dk, dear, facebook, X7evenstarz, condom, femal, money, theyr, batteri, china, turn, hope, life, sinc, steve, switch, everi, last, your, amaz, arent, date, divulg, ill, ive, lost, noth, worst, phone, care, way, year, updat, app, iphon, data, take, card, custom, die, event, problem, refus, two, use, screen, thing, buy, tri, cdp, doesnt, emiss, feel, fun, got, hour, macbook, miss, siri, start, upgrad, get, think, copi, guess, person, smart, best, differ, nsa, product, twitter, old, see, download, idea, job, mani, simpl, soon, team, technolog, wow, much, support, text, time, android, that, fingerprint, now, ask, colour, gonna, happen, man, give, hey, today, case, chang, one, work, day, nokia, preorder, look, awesom, done, featur, said, secur, seem, smartphon, thought, what, windowsphon, bit, realli, can, sure, dont, impress, call, didnt, drop, fail, first, improv, isnt, made, mean, put, yet, next., thank, market, blackberri, mac, never, plastic, right, sell, pleas, guy, scanner, want, black, free, innov, via, come, good, let, stop, color, compani, know, price, just, store, ios7, peopl, releas, say, anyon, devic, need, back, googl, itun, microsoft, will, appl, new, iphone5, ipad
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               3                    126.053                43.174
##   max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1   0.9619184                    0.5       0.9635628        0.8715152
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.9793925             0.9949999     0.4376929     0.81125
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.4       0.5789474        0.8651685
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.8252354             0.8988853     0.4987092
```

```r
# User specified
    # easier to exclude features
#model_id_pfx <- "";
# indep_vars_vctr <- setdiff(names(glb_trnent_df), 
#                         union(union(glb_rsp_var, glb_exclude_vars_as_features), 
#                                 c("<feat1_name>", "<feat2_name>")))
# method <- ""                                

    # easier to include features
#model_id_pfx <- ""; indep_vars_vctr <- c("<feat1_name>", "<feat1_name>"); method <- ""

    # User specified bivariate models
#     indep_vars_vctr_lst <- list()
#     for (feat in setdiff(names(glb_trnent_df), 
#                          union(glb_rsp_var, glb_exclude_vars_as_features)))
#         indep_vars_vctr_lst[["feat"]] <- feat

    # User specified combinatorial models
#     indep_vars_vctr_lst <- list()
#     combn_mtrx <- combn(c("<feat1_name>", "<feat2_name>", "<featn_name>"), 
#                           <num_feats_to_choose>)
#     for (combn_ix in 1:ncol(combn_mtrx))
#         #print(combn_mtrx[, combn_ix])
#         indep_vars_vctr_lst[[combn_ix]] <- combn_mtrx[, combn_ix]
    
    # template for myfit_mdl
    #   rf is hard-coded in caret to recognize only Accuracy / Kappa evaluation metrics
    #       only for OOB in trainControl ?
    
#     ret_lst <- myfit_mdl_fn(model_id=paste0(model_id_pfx, ""), model_method=method,
#                             indep_vars_vctr=indep_vars_vctr,
#                             rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
#                             fit_df=glb_trnent_df, OOB_df=glb_newent_df,
#                             n_cv_folds=glb_n_cv_folds, tune_models_df=glb_tune_models_df,
#                             model_loss_mtrx=glb_model_metric_terms,
#                             model_summaryFunction=glb_model_metric_smmry,
#                             model_metric=glb_model_metric,
#                             model_metric_maximize=glb_model_metric_maximize)

# Simplify a model
# fit_df <- glb_trnent_df; glb_mdl <- step(<complex>_mdl)

# Non-caret models
#     rpart_area_mdl <- rpart(reformulate("Area", response=glb_rsp_var), 
#                                data=glb_trnent_df, #method="class", 
#                                control=rpart.control(cp=0.12),
#                            parms=list(loss=glb_model_metric_terms))
#     print("rpart_sel_wlm_mdl"); prp(rpart_sel_wlm_mdl)
# 

print(glb_models_df)
```

```
##                     model_id     model_method
## 1          MFO.myMFO_classfr    myMFO_classfr
## 2    Random.myrandom_classfr myrandom_classfr
## 3       Max.cor.Y.cv.0.rpart            rpart
## 4  Max.cor.Y.cv.0.cp.0.rpart            rpart
## 5            Max.cor.Y.rpart            rpart
## 6              Max.cor.Y.glm              glm
## 7    Interact.High.cor.y.glm              glm
## 8              Low.cor.X.glm              glm
## 9          Conditional.X.glm              glm
## 10       Conditional.X.rpart            rpart
## 11  Conditional.X.cp.0.rpart            rpart
## 12          Conditional.X.rf               rf
## 13 Conditional.X.no.rnorm.rf               rf
##                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       feats
## 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    .rnorm
## 2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    .rnorm
## 3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     freak
## 4                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     freak
## 5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     freak
## 6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     freak
## 7                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               freak, freak:cdp, freak:httpbitly18xc8dk, freak:femal, freak:refus, freak:emiss, freak:itun
## 8                                               freak, hate, stuff, pictur, wtf, cant, shame, stupid, even, line, yooo, better, ever, fix, charger, still, charg, disappoint, short, like, yall, break., imessag, stand, togeth, cheap, wont, make, darn, httpbitly18xc8dk, dear, facebook, X7evenstarz, femal, money, theyr, batteri, china, turn, hope, life, sinc, steve, switch, everi, last, your, amaz, arent, date, ill, ive, lost, noth, worst, phone, care, way, year, updat, app, iphon, data, take, card, custom, die, event, problem, two, use, screen, thing, buy, tri, cdp, doesnt, emiss, feel, fun, got, hour, macbook, miss, siri, start, upgrad, get, think, copi, guess, person, smart, best, differ, nsa, product, twitter, old, see, download, idea, job, mani, simpl, soon, team, technolog, wow, much, support, text, time, android, that, fingerprint, now, ask, colour, gonna, happen, man, give, hey, today, case, chang, one, work, day, nokia, preorder, look, awesom, done, featur, said, secur, seem, smartphon, thought, what, windowsphon, bit, realli, can, sure, dont, impress, call, didnt, drop, fail, first, improv, isnt, made, mean, put, yet, next., thank, market, blackberri, mac, never, plastic, right, sell, pleas, guy, scanner, .rnorm, want, black, free, innov, via, come, good, let, stop, color, compani, know, price, just, store, ios7, peopl, releas, say, anyon, devic, need, back, googl, itun, microsoft, will, appl, new, iphone5
## 9  freak, hate, stuff, pictur, wtf, cant, shame, stupid, even, line, yooo, better, ever, fix, charger, still, charg, disappoint, short, like, amazon, yall, break., imessag, stand, togeth, cheap, wont, make, carbon, darn, httpbitly18xc8dk, dear, facebook, X7evenstarz, condom, femal, money, theyr, batteri, china, turn, hope, life, sinc, steve, switch, everi, last, your, amaz, arent, date, divulg, ill, ive, lost, noth, worst, phone, care, way, year, updat, app, iphon, data, take, card, custom, die, event, problem, refus, two, use, screen, thing, buy, tri, cdp, doesnt, emiss, feel, fun, got, hour, macbook, miss, siri, start, upgrad, get, think, copi, guess, person, smart, best, differ, nsa, product, twitter, old, see, download, idea, job, mani, simpl, soon, team, technolog, wow, much, support, text, time, android, that, fingerprint, now, ask, colour, gonna, happen, man, give, hey, today, case, chang, one, work, day, nokia, preorder, look, awesom, done, featur, said, secur, seem, smartphon, thought, what, windowsphon, bit, realli, can, sure, dont, impress, call, didnt, drop, fail, first, improv, isnt, made, mean, put, yet, next., thank, market, blackberri, mac, never, plastic, right, sell, pleas, guy, scanner, .rnorm, want, black, free, innov, via, come, good, let, stop, color, compani, know, price, just, store, ios7, peopl, releas, say, anyon, devic, need, back, googl, itun, microsoft, will, appl, new, iphone5, ipad
## 10         freak, hate, stuff, pictur, wtf, cant, shame, stupid, even, line, yooo, better, ever, fix, charger, still, charg, disappoint, short, like, amazon, yall, break., imessag, stand, togeth, cheap, wont, make, carbon, darn, httpbitly18xc8dk, dear, facebook, X7evenstarz, condom, femal, money, theyr, batteri, china, turn, hope, life, sinc, steve, switch, everi, last, your, amaz, arent, date, divulg, ill, ive, lost, noth, worst, phone, care, way, year, updat, app, iphon, data, take, card, custom, die, event, problem, refus, two, use, screen, thing, buy, tri, cdp, doesnt, emiss, feel, fun, got, hour, macbook, miss, siri, start, upgrad, get, think, copi, guess, person, smart, best, differ, nsa, product, twitter, old, see, download, idea, job, mani, simpl, soon, team, technolog, wow, much, support, text, time, android, that, fingerprint, now, ask, colour, gonna, happen, man, give, hey, today, case, chang, one, work, day, nokia, preorder, look, awesom, done, featur, said, secur, seem, smartphon, thought, what, windowsphon, bit, realli, can, sure, dont, impress, call, didnt, drop, fail, first, improv, isnt, made, mean, put, yet, next., thank, market, blackberri, mac, never, plastic, right, sell, pleas, guy, scanner, want, black, free, innov, via, come, good, let, stop, color, compani, know, price, just, store, ios7, peopl, releas, say, anyon, devic, need, back, googl, itun, microsoft, will, appl, new, iphone5, ipad
## 11         freak, hate, stuff, pictur, wtf, cant, shame, stupid, even, line, yooo, better, ever, fix, charger, still, charg, disappoint, short, like, amazon, yall, break., imessag, stand, togeth, cheap, wont, make, carbon, darn, httpbitly18xc8dk, dear, facebook, X7evenstarz, condom, femal, money, theyr, batteri, china, turn, hope, life, sinc, steve, switch, everi, last, your, amaz, arent, date, divulg, ill, ive, lost, noth, worst, phone, care, way, year, updat, app, iphon, data, take, card, custom, die, event, problem, refus, two, use, screen, thing, buy, tri, cdp, doesnt, emiss, feel, fun, got, hour, macbook, miss, siri, start, upgrad, get, think, copi, guess, person, smart, best, differ, nsa, product, twitter, old, see, download, idea, job, mani, simpl, soon, team, technolog, wow, much, support, text, time, android, that, fingerprint, now, ask, colour, gonna, happen, man, give, hey, today, case, chang, one, work, day, nokia, preorder, look, awesom, done, featur, said, secur, seem, smartphon, thought, what, windowsphon, bit, realli, can, sure, dont, impress, call, didnt, drop, fail, first, improv, isnt, made, mean, put, yet, next., thank, market, blackberri, mac, never, plastic, right, sell, pleas, guy, scanner, want, black, free, innov, via, come, good, let, stop, color, compani, know, price, just, store, ios7, peopl, releas, say, anyon, devic, need, back, googl, itun, microsoft, will, appl, new, iphone5, ipad
## 12 freak, hate, stuff, pictur, wtf, cant, shame, stupid, even, line, yooo, better, ever, fix, charger, still, charg, disappoint, short, like, amazon, yall, break., imessag, stand, togeth, cheap, wont, make, carbon, darn, httpbitly18xc8dk, dear, facebook, X7evenstarz, condom, femal, money, theyr, batteri, china, turn, hope, life, sinc, steve, switch, everi, last, your, amaz, arent, date, divulg, ill, ive, lost, noth, worst, phone, care, way, year, updat, app, iphon, data, take, card, custom, die, event, problem, refus, two, use, screen, thing, buy, tri, cdp, doesnt, emiss, feel, fun, got, hour, macbook, miss, siri, start, upgrad, get, think, copi, guess, person, smart, best, differ, nsa, product, twitter, old, see, download, idea, job, mani, simpl, soon, team, technolog, wow, much, support, text, time, android, that, fingerprint, now, ask, colour, gonna, happen, man, give, hey, today, case, chang, one, work, day, nokia, preorder, look, awesom, done, featur, said, secur, seem, smartphon, thought, what, windowsphon, bit, realli, can, sure, dont, impress, call, didnt, drop, fail, first, improv, isnt, made, mean, put, yet, next., thank, market, blackberri, mac, never, plastic, right, sell, pleas, guy, scanner, .rnorm, want, black, free, innov, via, come, good, let, stop, color, compani, know, price, just, store, ios7, peopl, releas, say, anyon, devic, need, back, googl, itun, microsoft, will, appl, new, iphone5, ipad
## 13         freak, hate, stuff, pictur, wtf, cant, shame, stupid, even, line, yooo, better, ever, fix, charger, still, charg, disappoint, short, like, amazon, yall, break., imessag, stand, togeth, cheap, wont, make, carbon, darn, httpbitly18xc8dk, dear, facebook, X7evenstarz, condom, femal, money, theyr, batteri, china, turn, hope, life, sinc, steve, switch, everi, last, your, amaz, arent, date, divulg, ill, ive, lost, noth, worst, phone, care, way, year, updat, app, iphon, data, take, card, custom, die, event, problem, refus, two, use, screen, thing, buy, tri, cdp, doesnt, emiss, feel, fun, got, hour, macbook, miss, siri, start, upgrad, get, think, copi, guess, person, smart, best, differ, nsa, product, twitter, old, see, download, idea, job, mani, simpl, soon, team, technolog, wow, much, support, text, time, android, that, fingerprint, now, ask, colour, gonna, happen, man, give, hey, today, case, chang, one, work, day, nokia, preorder, look, awesom, done, featur, said, secur, seem, smartphon, thought, what, windowsphon, bit, realli, can, sure, dont, impress, call, didnt, drop, fail, first, improv, isnt, made, mean, put, yet, next., thank, market, blackberri, mac, never, plastic, right, sell, pleas, guy, scanner, want, black, free, innov, via, come, good, let, stop, color, compani, know, price, just, store, ios7, peopl, releas, say, anyon, devic, need, back, googl, itun, microsoft, will, appl, new, iphone5, ipad
##    max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1                0                      0.304                 0.002
## 2                0                      0.226                 0.001
## 3                0                      0.590                 0.015
## 4                0                      0.449                 0.013
## 5                3                      0.956                 0.013
## 6                1                      1.091                 0.015
## 7                1                      0.928                 0.033
## 8                1                      4.411                 1.087
## 9                1                      5.138                 1.152
## 10               3                      3.532                 0.435
## 11               0                      0.932                 0.429
## 12               3                     91.638                23.538
## 13               3                    126.053                43.174
##    max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1    0.5000000                    0.5       0.0000000        0.8472727
## 2    0.4876695                    0.1       0.2649842        0.1527273
## 3    0.5000000                    0.5       0.0000000        0.8472727
## 4    0.6201546                    0.8       0.3850932        0.8800000
## 5    0.6201546                    0.8       0.3850932        0.8800000
## 6    0.6202001                    0.8       0.3850932        0.8800000
## 7    0.6202455                    0.8       0.3850932        0.8800000
## 8    0.9634853                    0.4       0.9180328        0.7503030
## 9    0.9641324                    0.4       0.9224490        0.7612121
## 10   0.6506404                    0.8       0.4561404        0.8824242
## 11   0.6727638                    0.6       0.5000000        0.8909091
## 12   1.0000000                    0.5       1.0000000        0.8800000
## 13   0.9619184                    0.5       0.9635628        0.8715152
##    max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1              0.8208848             0.8711498     0.0000000   0.5000000
## 2              0.1288502             0.1791152     0.0000000   0.5047619
## 3              0.8208848             0.8711498     0.0000000   0.5000000
## 4              0.8558521             0.9013914     0.3413572   0.6395238
## 5              0.8558521             0.9013914     0.3378153   0.6395238
## 6              0.8558521             0.9013914     0.3378153   0.6395833
## 7              0.8558521             0.9013914     0.3378153   0.6396429
## 8              0.9628069             0.9851307     0.2316521   0.7086607
## 9              0.9642682             0.9860786     0.2321957   0.7152381
## 10             0.8636885             0.9080451     0.3699305   0.6726190
## 11             0.8676171             0.9113614     0.4495596   0.6902381
## 12             0.9955386             1.0000000     0.4526428   0.7953274
## 13             0.9793925             0.9949999     0.4376929   0.8112500
##    opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                     0.5       0.0000000        0.8426966
## 2                     0.1       0.2718447        0.1573034
## 3                     0.5       0.0000000        0.8426966
## 4                     0.8       0.4324324        0.8820225
## 5                     0.8       0.4324324        0.8820225
## 6                     0.8       0.4324324        0.8820225
## 7                     0.8       0.4324324        0.8820225
## 8                     0.3       0.4960000        0.8230337
## 9                     0.4       0.4873950        0.8286517
## 10                    0.8       0.5000000        0.8876404
## 11                    0.6       0.5365854        0.8932584
## 12                    0.4       0.5405405        0.8567416
## 13                    0.4       0.5789474        0.8651685
##    max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1              0.8006423             0.8789342     0.0000000
## 2              0.1210658             0.1993577     0.0000000
## 3              0.8006423             0.8789342     0.0000000
## 4              0.8438883             0.9136353     0.3853995
## 5              0.8438883             0.9136353     0.3853995
## 6              0.8438883             0.9136353     0.3853995
## 7              0.8438883             0.9136353     0.3853995
## 8              0.7793438             0.8612518     0.3900794
## 9              0.7854100             0.8663234     0.3849553
## 10             0.8501517             0.9185050     0.4478908
## 11             0.8564403             0.9233486     0.4852359
## 12             0.8159786             0.8914388     0.4556901
## 13             0.8252354             0.8988853     0.4987092
##    max.AccuracySD.fit max.KappaSD.fit min.aic.fit
## 1                  NA              NA          NA
## 2                  NA              NA          NA
## 3                  NA              NA          NA
## 4                  NA              NA          NA
## 5          0.01585054      0.11932477          NA
## 6          0.01585054      0.11932477    608.9243
## 7          0.01585054      0.11932477    610.3916
## 8          0.04963781      0.09400829    623.1343
## 9          0.03531831      0.08116562    630.7763
## 10         0.01469619      0.10013786          NA
## 11                 NA              NA          NA
## 12                 NA              NA          NA
## 13                 NA              NA          NA
```

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="fit.models", 
    chunk_step_major=glb_script_df[nrow(glb_script_df), "chunk_step_major"], 
    chunk_step_minor=glb_script_df[nrow(glb_script_df), "chunk_step_minor"]+1,                              
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##          chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed8  fit.models                5                0  24.668
## elapsed9  fit.models                5                1 299.425
```


```r
if (!is.null(glb_model_metric_smmry)) {
    stats_df <- glb_models_df[, "model_id", FALSE]

    stats_mdl_df <- data.frame()
    for (model_id in stats_df$model_id) {
        stats_mdl_df <- rbind(stats_mdl_df, 
            mypredict_mdl(glb_models_lst[[model_id]], glb_trnent_df, glb_rsp_var, 
                          glb_rsp_var_out, model_id, "fit",
        						glb_model_metric_smmry, glb_model_metric, 
        						glb_model_metric_maximize, ret_type="stats"))
    }
    stats_df <- merge(stats_df, stats_mdl_df, all.x=TRUE)
    
    stats_mdl_df <- data.frame()
    for (model_id in stats_df$model_id) {
        stats_mdl_df <- rbind(stats_mdl_df, 
            mypredict_mdl(glb_models_lst[[model_id]], glb_newent_df, glb_rsp_var, 
                          glb_rsp_var_out, model_id, "OOB",
            					glb_model_metric_smmry, glb_model_metric, 
        						glb_model_metric_maximize, ret_type="stats"))
    }
    stats_df <- merge(stats_df, stats_mdl_df, all.x=TRUE)
    
#     tmp_models_df <- orderBy(~model_id, glb_models_df)
#     rownames(tmp_models_df) <- seq(1, nrow(tmp_models_df))
#     all.equal(subset(tmp_models_df[, names(stats_df)], model_id != "Random.myrandom_classfr"),
#               subset(stats_df, model_id != "Random.myrandom_classfr"))
#     print(subset(tmp_models_df[, names(stats_df)], model_id != "Random.myrandom_classfr")[, c("model_id", "max.Accuracy.fit")])
#     print(subset(stats_df, model_id != "Random.myrandom_classfr")[, c("model_id", "max.Accuracy.fit")])

    print("Merging following data into glb_models_df:")
    print(stats_mrg_df <- stats_df[, c(1, grep(glb_model_metric, names(stats_df)))])
    print(tmp_models_df <- orderBy(~model_id, glb_models_df[, c("model_id", grep(glb_model_metric, names(stats_df), value=TRUE))]))

    tmp2_models_df <- glb_models_df[, c("model_id", setdiff(names(glb_models_df), grep(glb_model_metric, names(stats_df), value=TRUE)))]
    tmp3_models_df <- merge(tmp2_models_df, stats_mrg_df, all.x=TRUE, sort=FALSE)
    print(tmp3_models_df)
    print(names(tmp3_models_df))
    print(glb_models_df <- subset(tmp3_models_df, select=-model_id.1))
}

plt_models_df <- glb_models_df[, -grep("SD|Upper|Lower", names(glb_models_df))]
for (var in grep("^min.", names(plt_models_df), value=TRUE)) {
    plt_models_df[, sub("min.", "inv.", var)] <- 
        #ifelse(all(is.na(tmp <- plt_models_df[, var])), NA, 1.0 / tmp)
        1.0 / plt_models_df[, var]
    plt_models_df <- plt_models_df[ , -grep(var, names(plt_models_df))]
}
print(plt_models_df)
```

```
##                     model_id     model_method
## 1          MFO.myMFO_classfr    myMFO_classfr
## 2    Random.myrandom_classfr myrandom_classfr
## 3       Max.cor.Y.cv.0.rpart            rpart
## 4  Max.cor.Y.cv.0.cp.0.rpart            rpart
## 5            Max.cor.Y.rpart            rpart
## 6              Max.cor.Y.glm              glm
## 7    Interact.High.cor.y.glm              glm
## 8              Low.cor.X.glm              glm
## 9          Conditional.X.glm              glm
## 10       Conditional.X.rpart            rpart
## 11  Conditional.X.cp.0.rpart            rpart
## 12          Conditional.X.rf               rf
## 13 Conditional.X.no.rnorm.rf               rf
##                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       feats
## 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    .rnorm
## 2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    .rnorm
## 3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     freak
## 4                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     freak
## 5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     freak
## 6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     freak
## 7                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               freak, freak:cdp, freak:httpbitly18xc8dk, freak:femal, freak:refus, freak:emiss, freak:itun
## 8                                               freak, hate, stuff, pictur, wtf, cant, shame, stupid, even, line, yooo, better, ever, fix, charger, still, charg, disappoint, short, like, yall, break., imessag, stand, togeth, cheap, wont, make, darn, httpbitly18xc8dk, dear, facebook, X7evenstarz, femal, money, theyr, batteri, china, turn, hope, life, sinc, steve, switch, everi, last, your, amaz, arent, date, ill, ive, lost, noth, worst, phone, care, way, year, updat, app, iphon, data, take, card, custom, die, event, problem, two, use, screen, thing, buy, tri, cdp, doesnt, emiss, feel, fun, got, hour, macbook, miss, siri, start, upgrad, get, think, copi, guess, person, smart, best, differ, nsa, product, twitter, old, see, download, idea, job, mani, simpl, soon, team, technolog, wow, much, support, text, time, android, that, fingerprint, now, ask, colour, gonna, happen, man, give, hey, today, case, chang, one, work, day, nokia, preorder, look, awesom, done, featur, said, secur, seem, smartphon, thought, what, windowsphon, bit, realli, can, sure, dont, impress, call, didnt, drop, fail, first, improv, isnt, made, mean, put, yet, next., thank, market, blackberri, mac, never, plastic, right, sell, pleas, guy, scanner, .rnorm, want, black, free, innov, via, come, good, let, stop, color, compani, know, price, just, store, ios7, peopl, releas, say, anyon, devic, need, back, googl, itun, microsoft, will, appl, new, iphone5
## 9  freak, hate, stuff, pictur, wtf, cant, shame, stupid, even, line, yooo, better, ever, fix, charger, still, charg, disappoint, short, like, amazon, yall, break., imessag, stand, togeth, cheap, wont, make, carbon, darn, httpbitly18xc8dk, dear, facebook, X7evenstarz, condom, femal, money, theyr, batteri, china, turn, hope, life, sinc, steve, switch, everi, last, your, amaz, arent, date, divulg, ill, ive, lost, noth, worst, phone, care, way, year, updat, app, iphon, data, take, card, custom, die, event, problem, refus, two, use, screen, thing, buy, tri, cdp, doesnt, emiss, feel, fun, got, hour, macbook, miss, siri, start, upgrad, get, think, copi, guess, person, smart, best, differ, nsa, product, twitter, old, see, download, idea, job, mani, simpl, soon, team, technolog, wow, much, support, text, time, android, that, fingerprint, now, ask, colour, gonna, happen, man, give, hey, today, case, chang, one, work, day, nokia, preorder, look, awesom, done, featur, said, secur, seem, smartphon, thought, what, windowsphon, bit, realli, can, sure, dont, impress, call, didnt, drop, fail, first, improv, isnt, made, mean, put, yet, next., thank, market, blackberri, mac, never, plastic, right, sell, pleas, guy, scanner, .rnorm, want, black, free, innov, via, come, good, let, stop, color, compani, know, price, just, store, ios7, peopl, releas, say, anyon, devic, need, back, googl, itun, microsoft, will, appl, new, iphone5, ipad
## 10         freak, hate, stuff, pictur, wtf, cant, shame, stupid, even, line, yooo, better, ever, fix, charger, still, charg, disappoint, short, like, amazon, yall, break., imessag, stand, togeth, cheap, wont, make, carbon, darn, httpbitly18xc8dk, dear, facebook, X7evenstarz, condom, femal, money, theyr, batteri, china, turn, hope, life, sinc, steve, switch, everi, last, your, amaz, arent, date, divulg, ill, ive, lost, noth, worst, phone, care, way, year, updat, app, iphon, data, take, card, custom, die, event, problem, refus, two, use, screen, thing, buy, tri, cdp, doesnt, emiss, feel, fun, got, hour, macbook, miss, siri, start, upgrad, get, think, copi, guess, person, smart, best, differ, nsa, product, twitter, old, see, download, idea, job, mani, simpl, soon, team, technolog, wow, much, support, text, time, android, that, fingerprint, now, ask, colour, gonna, happen, man, give, hey, today, case, chang, one, work, day, nokia, preorder, look, awesom, done, featur, said, secur, seem, smartphon, thought, what, windowsphon, bit, realli, can, sure, dont, impress, call, didnt, drop, fail, first, improv, isnt, made, mean, put, yet, next., thank, market, blackberri, mac, never, plastic, right, sell, pleas, guy, scanner, want, black, free, innov, via, come, good, let, stop, color, compani, know, price, just, store, ios7, peopl, releas, say, anyon, devic, need, back, googl, itun, microsoft, will, appl, new, iphone5, ipad
## 11         freak, hate, stuff, pictur, wtf, cant, shame, stupid, even, line, yooo, better, ever, fix, charger, still, charg, disappoint, short, like, amazon, yall, break., imessag, stand, togeth, cheap, wont, make, carbon, darn, httpbitly18xc8dk, dear, facebook, X7evenstarz, condom, femal, money, theyr, batteri, china, turn, hope, life, sinc, steve, switch, everi, last, your, amaz, arent, date, divulg, ill, ive, lost, noth, worst, phone, care, way, year, updat, app, iphon, data, take, card, custom, die, event, problem, refus, two, use, screen, thing, buy, tri, cdp, doesnt, emiss, feel, fun, got, hour, macbook, miss, siri, start, upgrad, get, think, copi, guess, person, smart, best, differ, nsa, product, twitter, old, see, download, idea, job, mani, simpl, soon, team, technolog, wow, much, support, text, time, android, that, fingerprint, now, ask, colour, gonna, happen, man, give, hey, today, case, chang, one, work, day, nokia, preorder, look, awesom, done, featur, said, secur, seem, smartphon, thought, what, windowsphon, bit, realli, can, sure, dont, impress, call, didnt, drop, fail, first, improv, isnt, made, mean, put, yet, next., thank, market, blackberri, mac, never, plastic, right, sell, pleas, guy, scanner, want, black, free, innov, via, come, good, let, stop, color, compani, know, price, just, store, ios7, peopl, releas, say, anyon, devic, need, back, googl, itun, microsoft, will, appl, new, iphone5, ipad
## 12 freak, hate, stuff, pictur, wtf, cant, shame, stupid, even, line, yooo, better, ever, fix, charger, still, charg, disappoint, short, like, amazon, yall, break., imessag, stand, togeth, cheap, wont, make, carbon, darn, httpbitly18xc8dk, dear, facebook, X7evenstarz, condom, femal, money, theyr, batteri, china, turn, hope, life, sinc, steve, switch, everi, last, your, amaz, arent, date, divulg, ill, ive, lost, noth, worst, phone, care, way, year, updat, app, iphon, data, take, card, custom, die, event, problem, refus, two, use, screen, thing, buy, tri, cdp, doesnt, emiss, feel, fun, got, hour, macbook, miss, siri, start, upgrad, get, think, copi, guess, person, smart, best, differ, nsa, product, twitter, old, see, download, idea, job, mani, simpl, soon, team, technolog, wow, much, support, text, time, android, that, fingerprint, now, ask, colour, gonna, happen, man, give, hey, today, case, chang, one, work, day, nokia, preorder, look, awesom, done, featur, said, secur, seem, smartphon, thought, what, windowsphon, bit, realli, can, sure, dont, impress, call, didnt, drop, fail, first, improv, isnt, made, mean, put, yet, next., thank, market, blackberri, mac, never, plastic, right, sell, pleas, guy, scanner, .rnorm, want, black, free, innov, via, come, good, let, stop, color, compani, know, price, just, store, ios7, peopl, releas, say, anyon, devic, need, back, googl, itun, microsoft, will, appl, new, iphone5, ipad
## 13         freak, hate, stuff, pictur, wtf, cant, shame, stupid, even, line, yooo, better, ever, fix, charger, still, charg, disappoint, short, like, amazon, yall, break., imessag, stand, togeth, cheap, wont, make, carbon, darn, httpbitly18xc8dk, dear, facebook, X7evenstarz, condom, femal, money, theyr, batteri, china, turn, hope, life, sinc, steve, switch, everi, last, your, amaz, arent, date, divulg, ill, ive, lost, noth, worst, phone, care, way, year, updat, app, iphon, data, take, card, custom, die, event, problem, refus, two, use, screen, thing, buy, tri, cdp, doesnt, emiss, feel, fun, got, hour, macbook, miss, siri, start, upgrad, get, think, copi, guess, person, smart, best, differ, nsa, product, twitter, old, see, download, idea, job, mani, simpl, soon, team, technolog, wow, much, support, text, time, android, that, fingerprint, now, ask, colour, gonna, happen, man, give, hey, today, case, chang, one, work, day, nokia, preorder, look, awesom, done, featur, said, secur, seem, smartphon, thought, what, windowsphon, bit, realli, can, sure, dont, impress, call, didnt, drop, fail, first, improv, isnt, made, mean, put, yet, next., thank, market, blackberri, mac, never, plastic, right, sell, pleas, guy, scanner, want, black, free, innov, via, come, good, let, stop, color, compani, know, price, just, store, ios7, peopl, releas, say, anyon, devic, need, back, googl, itun, microsoft, will, appl, new, iphone5, ipad
##    max.nTuningRuns max.auc.fit opt.prob.threshold.fit max.f.score.fit
## 1                0   0.5000000                    0.5       0.0000000
## 2                0   0.4876695                    0.1       0.2649842
## 3                0   0.5000000                    0.5       0.0000000
## 4                0   0.6201546                    0.8       0.3850932
## 5                3   0.6201546                    0.8       0.3850932
## 6                1   0.6202001                    0.8       0.3850932
## 7                1   0.6202455                    0.8       0.3850932
## 8                1   0.9634853                    0.4       0.9180328
## 9                1   0.9641324                    0.4       0.9224490
## 10               3   0.6506404                    0.8       0.4561404
## 11               0   0.6727638                    0.6       0.5000000
## 12               3   1.0000000                    0.5       1.0000000
## 13               3   0.9619184                    0.5       0.9635628
##    max.Accuracy.fit max.Kappa.fit max.auc.OOB opt.prob.threshold.OOB
## 1         0.8472727     0.0000000   0.5000000                    0.5
## 2         0.1527273     0.0000000   0.5047619                    0.1
## 3         0.8472727     0.0000000   0.5000000                    0.5
## 4         0.8800000     0.3413572   0.6395238                    0.8
## 5         0.8800000     0.3378153   0.6395238                    0.8
## 6         0.8800000     0.3378153   0.6395833                    0.8
## 7         0.8800000     0.3378153   0.6396429                    0.8
## 8         0.7503030     0.2316521   0.7086607                    0.3
## 9         0.7612121     0.2321957   0.7152381                    0.4
## 10        0.8824242     0.3699305   0.6726190                    0.8
## 11        0.8909091     0.4495596   0.6902381                    0.6
## 12        0.8800000     0.4526428   0.7953274                    0.4
## 13        0.8715152     0.4376929   0.8112500                    0.4
##    max.f.score.OOB max.Accuracy.OOB max.Kappa.OOB
## 1        0.0000000        0.8426966     0.0000000
## 2        0.2718447        0.1573034     0.0000000
## 3        0.0000000        0.8426966     0.0000000
## 4        0.4324324        0.8820225     0.3853995
## 5        0.4324324        0.8820225     0.3853995
## 6        0.4324324        0.8820225     0.3853995
## 7        0.4324324        0.8820225     0.3853995
## 8        0.4960000        0.8230337     0.3900794
## 9        0.4873950        0.8286517     0.3849553
## 10       0.5000000        0.8876404     0.4478908
## 11       0.5365854        0.8932584     0.4852359
## 12       0.5405405        0.8567416     0.4556901
## 13       0.5789474        0.8651685     0.4987092
##    inv.elapsedtime.everything inv.elapsedtime.final inv.aic.fit
## 1                 3.289473684          5.000000e+02          NA
## 2                 4.424778761          1.000000e+03          NA
## 3                 1.694915254          6.666667e+01          NA
## 4                 2.227171492          7.692308e+01          NA
## 5                 1.046025105          7.692308e+01          NA
## 6                 0.916590284          6.666667e+01 0.001642240
## 7                 1.077586207          3.030303e+01 0.001638293
## 8                 0.226705962          9.199632e-01 0.001604790
## 9                 0.194628260          8.680556e-01 0.001585348
## 10                0.283125708          2.298851e+00          NA
## 11                1.072961373          2.331002e+00          NA
## 12                0.010912504          4.248449e-02          NA
## 13                0.007933171          2.316209e-02          NA
```

```r
print(myplot_radar(radar_inp_df=plt_models_df))
```

```
## Warning in RColorBrewer::brewer.pal(n, pal): n too large, allowed maximum for palette Set1 is 9
## Returning the palette you asked for with that many colors
```

```
## Warning: The shape palette can deal with a maximum of 6 discrete values
## because more than 6 becomes difficult to discriminate; you have
## 13. Consider specifying shapes manually. if you must have them.
```

```
## Warning: Removed 6 rows containing missing values (geom_path).
```

```
## Warning: Removed 104 rows containing missing values (geom_point).
```

```
## Warning: Removed 9 rows containing missing values (geom_text).
```

```
## Warning in RColorBrewer::brewer.pal(n, pal): n too large, allowed maximum for palette Set1 is 9
## Returning the palette you asked for with that many colors
```

```
## Warning: The shape palette can deal with a maximum of 6 discrete values
## because more than 6 becomes difficult to discriminate; you have
## 13. Consider specifying shapes manually. if you must have them.
```

![](Apple_Tweets_files/figure-html/fit.models_1-1.png) 

```r
# print(myplot_radar(radar_inp_df=subset(plt_models_df, 
#         !(model_id %in% grep("random|MFO", plt_models_df$model_id, value=TRUE)))))

# Compute CI for <metric>SD
glb_models_df <- mutate(glb_models_df, 
                max.df = ifelse(max.nTuningRuns > 1, max.nTuningRuns - 1, NA),
                min.sd2ci.scaler = ifelse(is.na(max.df), NA, qt(0.975, max.df)))
for (var in grep("SD", names(glb_models_df), value=TRUE)) {
    # Does CI alredy exist ?
    var_components <- unlist(strsplit(var, "SD"))
    varActul <- paste0(var_components[1],          var_components[2])
    varUpper <- paste0(var_components[1], "Upper", var_components[2])
    varLower <- paste0(var_components[1], "Lower", var_components[2])
    if (varUpper %in% names(glb_models_df)) {
        warning(varUpper, " already exists in glb_models_df")
        # Assuming Lower also exists
        next
    }    
    print(sprintf("var:%s", var))
    # CI is dependent on sample size in t distribution; df=n-1
    glb_models_df[, varUpper] <- glb_models_df[, varActul] + 
        glb_models_df[, "min.sd2ci.scaler"] * glb_models_df[, var]
    glb_models_df[, varLower] <- glb_models_df[, varActul] - 
        glb_models_df[, "min.sd2ci.scaler"] * glb_models_df[, var]
}
```

```
## Warning: max.AccuracyUpper.fit already exists in glb_models_df
```

```
## [1] "var:max.KappaSD.fit"
```

```r
# Plot metrics with CI
plt_models_df <- glb_models_df[, "model_id", FALSE]
pltCI_models_df <- glb_models_df[, "model_id", FALSE]
for (var in grep("Upper", names(glb_models_df), value=TRUE)) {
    var_components <- unlist(strsplit(var, "Upper"))
    col_name <- unlist(paste(var_components, collapse=""))
    plt_models_df[, col_name] <- glb_models_df[, col_name]
    for (name in paste0(var_components[1], c("Upper", "Lower"), var_components[2]))
        pltCI_models_df[, name] <- glb_models_df[, name]
}

build_statsCI_data <- function(plt_models_df) {
    mltd_models_df <- melt(plt_models_df, id.vars="model_id")
    mltd_models_df$data <- sapply(1:nrow(mltd_models_df), 
        function(row_ix) tail(unlist(strsplit(as.character(
            mltd_models_df[row_ix, "variable"]), "[.]")), 1))
    mltd_models_df$label <- sapply(1:nrow(mltd_models_df), 
        function(row_ix) head(unlist(strsplit(as.character(
            mltd_models_df[row_ix, "variable"]), paste0(".", mltd_models_df[row_ix, "data"]))), 1))
    #print(mltd_models_df)
    
    return(mltd_models_df)
}
mltd_models_df <- build_statsCI_data(plt_models_df)

mltdCI_models_df <- melt(pltCI_models_df, id.vars="model_id")
for (row_ix in 1:nrow(mltdCI_models_df)) {
    for (type in c("Upper", "Lower")) {
        if (length(var_components <- unlist(strsplit(
                as.character(mltdCI_models_df[row_ix, "variable"]), type))) > 1) {
            #print(sprintf("row_ix:%d; type:%s; ", row_ix, type))
            mltdCI_models_df[row_ix, "label"] <- var_components[1]
            mltdCI_models_df[row_ix, "data"] <- unlist(strsplit(var_components[2], "[.]"))[2]
            mltdCI_models_df[row_ix, "type"] <- type
            break
        }
    }    
}
#print(mltdCI_models_df)
# castCI_models_df <- dcast(mltdCI_models_df, value ~ type, fun.aggregate=sum)
# print(castCI_models_df)
wideCI_models_df <- reshape(subset(mltdCI_models_df, select=-variable), 
                            timevar="type", 
        idvar=setdiff(names(mltdCI_models_df), c("type", "value", "variable")), 
                            direction="wide")
#print(wideCI_models_df)
mrgdCI_models_df <- merge(wideCI_models_df, mltd_models_df, all.x=TRUE)
#print(mrgdCI_models_df)

# Merge stats back in if CIs don't exist
goback_vars <- c()
for (var in unique(mltd_models_df$label)) {
    for (type in unique(mltd_models_df$data)) {
        var_type <- paste0(var, ".", type)
        # if this data is already present, next
        if (var_type %in% unique(paste(mltd_models_df$label, mltd_models_df$data, sep=".")))
            next
        #print(sprintf("var_type:%s", var_type))
        goback_vars <- c(goback_vars, var_type)
    }
}

if (length(goback_vars) > 0) {
    mltd_goback_df <- build_statsCI_data(glb_models_df[, c("model_id", goback_vars)])
    mltd_models_df <- rbind(mltd_models_df, mltd_goback_df)
}

mltd_models_df <- merge(mltd_models_df, glb_models_df[, c("model_id", "model_method")], all.x=TRUE)

png(paste0(glb_out_pfx, "models_bar.png"), width=480*3, height=480*2)
print(gp <- myplot_bar(mltd_models_df, "model_id", "value", colorcol_name="model_method") + 
        geom_errorbar(data=mrgdCI_models_df, 
            mapping=aes(x=model_id, ymax=value.Upper, ymin=value.Lower), width=0.5) + 
          facet_grid(label ~ data, scales="free") + 
          theme(axis.text.x = element_text(angle = 90,vjust = 0.5)))
dev.off()
```

```
## quartz_off_screen 
##                 2
```

```r
print(gp)
```

![](Apple_Tweets_files/figure-html/fit.models_1-2.png) 

```r
# used for console inspection
model_evl_terms <- c(NULL)
for (metric in glb_model_evl_criteria)
    model_evl_terms <- c(model_evl_terms, 
                         ifelse(length(grep("max", metric)) > 0, "-", "+"), metric)
model_sel_frmla <- as.formula(paste(c("~ ", model_evl_terms), collapse=" "))
print(tmp_models_df <- orderBy(model_sel_frmla, glb_models_df)[, c("model_id", glb_model_evl_criteria)])
```

```
##                     model_id max.Accuracy.OOB max.Kappa.OOB min.aic.fit
## 11  Conditional.X.cp.0.rpart        0.8932584     0.4852359          NA
## 10       Conditional.X.rpart        0.8876404     0.4478908          NA
## 6              Max.cor.Y.glm        0.8820225     0.3853995    608.9243
## 7    Interact.High.cor.y.glm        0.8820225     0.3853995    610.3916
## 4  Max.cor.Y.cv.0.cp.0.rpart        0.8820225     0.3853995          NA
## 5            Max.cor.Y.rpart        0.8820225     0.3853995          NA
## 13 Conditional.X.no.rnorm.rf        0.8651685     0.4987092          NA
## 12          Conditional.X.rf        0.8567416     0.4556901          NA
## 1          MFO.myMFO_classfr        0.8426966     0.0000000          NA
## 3       Max.cor.Y.cv.0.rpart        0.8426966     0.0000000          NA
## 9          Conditional.X.glm        0.8286517     0.3849553    630.7763
## 8              Low.cor.X.glm        0.8230337     0.3900794    623.1343
## 2    Random.myrandom_classfr        0.1573034     0.0000000          NA
```

```r
print(myplot_radar(radar_inp_df=tmp_models_df))
```

```
## Warning in RColorBrewer::brewer.pal(n, pal): n too large, allowed maximum for palette Set1 is 9
## Returning the palette you asked for with that many colors
```

```
## Warning: The shape palette can deal with a maximum of 6 discrete values
## because more than 6 becomes difficult to discriminate; you have
## 13. Consider specifying shapes manually. if you must have them.
```

```
## Warning: Removed 6 rows containing missing values (geom_path).
```

```
## Warning: Removed 27 rows containing missing values (geom_point).
```

```
## Warning: Removed 9 rows containing missing values (geom_text).
```

```
## Warning in RColorBrewer::brewer.pal(n, pal): n too large, allowed maximum for palette Set1 is 9
## Returning the palette you asked for with that many colors
```

```
## Warning: The shape palette can deal with a maximum of 6 discrete values
## because more than 6 becomes difficult to discriminate; you have
## 13. Consider specifying shapes manually. if you must have them.
```

![](Apple_Tweets_files/figure-html/fit.models_1-3.png) 

```r
print("Metrics used for model selection:"); print(model_sel_frmla)
```

```
## [1] "Metrics used for model selection:"
```

```
## ~-max.Accuracy.OOB - max.Kappa.OOB + min.aic.fit
```

```r
print(sprintf("Best model id: %s", tmp_models_df[1, "model_id"]))
```

```
## [1] "Best model id: Conditional.X.cp.0.rpart"
```

```r
if (is.null(glb_sel_mdl_id)) 
    { glb_sel_mdl_id <- tmp_models_df[1, "model_id"] } else 
        print(sprintf("User specified selection: %s", glb_sel_mdl_id))   
    
myprint_mdl(glb_sel_mdl <- glb_models_lst[[glb_sel_mdl_id]])
```

![](Apple_Tweets_files/figure-html/fit.models_1-4.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 825 
## 
##           CP nsplit rel error
## 1 0.21428571      0 1.0000000
## 2 0.04761905      1 0.7857143
## 3 0.02380952      2 0.7380952
## 4 0.00000000      3 0.7142857
## 
## Variable importance
## freak  hate stuff 
##    72    17    10 
## 
## Node number 1: 825 observations,    complexity param=0.2142857
##   predicted class=N  expected loss=0.1527273  P(node) =1
##     class counts:   699   126
##    probabilities: 0.847 0.153 
##   left son=2 (790 obs) right son=3 (35 obs)
##   Primary splits:
##       freak  < 0.5 to the left,  improve=39.275110, (0 missing)
##       hate   < 0.5 to the left,  improve= 9.873907, (0 missing)
##       stuff  < 0.5 to the left,  improve= 6.063648, (0 missing)
##       pictur < 0.5 to the left,  improve= 5.763645, (0 missing)
##       wtf    < 0.5 to the left,  improve= 5.763645, (0 missing)
## 
## Node number 2: 790 observations,    complexity param=0.04761905
##   predicted class=N  expected loss=0.1202532  P(node) =0.9575758
##     class counts:   695    95
##    probabilities: 0.880 0.120 
##   left son=4 (780 obs) right son=5 (10 obs)
##   Primary splits:
##       hate  < 0.5 to the left,  improve=9.359591, (0 missing)
##       wtf   < 0.5 to the left,  improve=6.410211, (0 missing)
##       stuff < 0.5 to the left,  improve=5.436150, (0 missing)
##       fix   < 0.5 to the left,  improve=2.875307, (0 missing)
##       cant  < 0.5 to the left,  improve=2.709704, (0 missing)
## 
## Node number 3: 35 observations
##   predicted class=Y  expected loss=0.1142857  P(node) =0.04242424
##     class counts:     4    31
##    probabilities: 0.114 0.886 
## 
## Node number 4: 780 observations,    complexity param=0.02380952
##   predicted class=N  expected loss=0.1115385  P(node) =0.9454545
##     class counts:   693    87
##    probabilities: 0.888 0.112 
##   left son=8 (771 obs) right son=9 (9 obs)
##   Primary splits:
##       stuff  < 0.5 to the left,  improve=5.611763, (0 missing)
##       wtf    < 0.5 to the left,  improve=5.132319, (0 missing)
##       fix    < 0.5 to the left,  improve=2.987798, (0 missing)
##       cant   < 0.5 to the left,  improve=2.865713, (0 missing)
##       better < 0.5 to the left,  improve=2.865713, (0 missing)
## 
## Node number 5: 10 observations
##   predicted class=Y  expected loss=0.2  P(node) =0.01212121
##     class counts:     2     8
##    probabilities: 0.200 0.800 
## 
## Node number 8: 771 observations
##   predicted class=N  expected loss=0.1050584  P(node) =0.9345455
##     class counts:   690    81
##    probabilities: 0.895 0.105 
## 
## Node number 9: 9 observations
##   predicted class=Y  expected loss=0.3333333  P(node) =0.01090909
##     class counts:     3     6
##    probabilities: 0.333 0.667 
## 
## n= 825 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 825 126 N (0.8472727 0.1527273)  
##   2) freak< 0.5 790  95 N (0.8797468 0.1202532)  
##     4) hate< 0.5 780  87 N (0.8884615 0.1115385)  
##       8) stuff< 0.5 771  81 N (0.8949416 0.1050584) *
##       9) stuff>=0.5 9   3 Y (0.3333333 0.6666667) *
##     5) hate>=0.5 10   2 Y (0.2000000 0.8000000) *
##   3) freak>=0.5 35   4 Y (0.1142857 0.8857143) *
```

```
## [1] TRUE
```

```r
replay.petrisim(pn=glb_analytics_pn, 
    replay.trans=(glb_analytics_avl_objs <- c(glb_analytics_avl_objs, 
        "model.selected")), flip_coord=TRUE)
```

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0 
## 2.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction data.new.prediction 	firing:  model.selected 
## 3.0000 	 3 	 0 2 1 0
```

![](Apple_Tweets_files/figure-html/fit.models_1-5.png) 

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="fit.data.training.all", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                     chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed9             fit.models                5                1 299.425
## elapsed10 fit.data.training.all                6                0 313.567
```

## Step `6`: fit.data.training.all

```r
if (!is.null(glb_fin_mdl_id) && (glb_fin_mdl_id %in% names(glb_models_lst))) {
    warning("Final model same as user selected model")
    glb_fin_mdl <- glb_sel_mdl
} else {    
    print(mdl_feats_df <- myextract_mdl_feats(sel_mdl=glb_sel_mdl, entity_df=glb_trnent_df))
    
    if ((model_method <- glb_sel_mdl$method) == "custom")
        # get actual method from the model_id
        model_method <- tail(unlist(strsplit(glb_sel_mdl_id, "[.]")), 1)
    
    tune_finmdl_df <- NULL
    if (nrow(glb_sel_mdl$bestTune) > 0) {
        for (param in names(glb_sel_mdl$bestTune)) {
            #print(sprintf("param: %s", param))
            tune_finmdl_df <- rbind(tune_finmdl_df, 
                data.frame(parameter=param, 
                           min=glb_sel_mdl$bestTune[1, param], 
                           max=glb_sel_mdl$bestTune[1, param], 
                           by=1)) # by val does not matter
        }
    } 
    
    # Sync with parameters in mydsutils.R
    ret_lst <- myfit_mdl(model_id="Final", model_method=model_method,
                            indep_vars_vctr=mdl_feats_df$id, model_type=glb_model_type,
                            rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out, 
                            fit_df=glb_trnent_df, OOB_df=NULL,
                            n_cv_folds=glb_n_cv_folds, tune_models_df=tune_finmdl_df,
                         # Automate from here
                         #  Issues if glb_sel_mdl$method == "rf" b/c trainControl is "oob"; not "cv"
                            model_loss_mtrx=glb_model_metric_terms,
                            model_summaryFunction=glb_sel_mdl$control$summaryFunction,
                            model_metric=glb_sel_mdl$metric,
                            model_metric_maximize=glb_sel_mdl$maximize)
    glb_fin_mdl <- glb_models_lst[[length(glb_models_lst)]] 
    glb_fin_mdl_id <- glb_models_df[length(glb_models_lst), "model_id"]
}
```

```
##                  importance               id fit.feat
## freak            100.000000            freak     TRUE
## hate              48.971207             hate     TRUE
## wtf               44.063971              wtf     TRUE
## stuff             43.568454            stuff     TRUE
## fix               14.928296              fix     TRUE
## pictur            14.675057           pictur     TRUE
## cant              14.195802             cant     TRUE
## better             7.296511           better     TRUE
## shame              0.000000            shame     TRUE
## stupid             0.000000           stupid     TRUE
## even               0.000000             even     TRUE
## line               0.000000             line     TRUE
## yooo               0.000000             yooo     TRUE
## ever               0.000000             ever     TRUE
## charger            0.000000          charger     TRUE
## still              0.000000            still     TRUE
## charg              0.000000            charg     TRUE
## disappoint         0.000000       disappoint     TRUE
## short              0.000000            short     TRUE
## like               0.000000             like     TRUE
## amazon             0.000000           amazon     TRUE
## yall               0.000000             yall     TRUE
## break.             0.000000           break.     TRUE
## imessag            0.000000          imessag     TRUE
## stand              0.000000            stand     TRUE
## togeth             0.000000           togeth     TRUE
## cheap              0.000000            cheap     TRUE
## wont               0.000000             wont     TRUE
## make               0.000000             make     TRUE
## carbon             0.000000           carbon     TRUE
## darn               0.000000             darn     TRUE
## httpbitly18xc8dk   0.000000 httpbitly18xc8dk     TRUE
## dear               0.000000             dear     TRUE
## facebook           0.000000         facebook     TRUE
## X7evenstarz        0.000000      X7evenstarz     TRUE
## condom             0.000000           condom     TRUE
## femal              0.000000            femal     TRUE
## money              0.000000            money     TRUE
## theyr              0.000000            theyr     TRUE
## batteri            0.000000          batteri     TRUE
## china              0.000000            china     TRUE
## turn               0.000000             turn     TRUE
## hope               0.000000             hope     TRUE
## life               0.000000             life     TRUE
## sinc               0.000000             sinc     TRUE
## steve              0.000000            steve     TRUE
## switch             0.000000           switch     TRUE
## everi              0.000000            everi     TRUE
## last               0.000000             last     TRUE
## your               0.000000             your     TRUE
## amaz               0.000000             amaz     TRUE
## arent              0.000000            arent     TRUE
## date               0.000000             date     TRUE
## divulg             0.000000           divulg     TRUE
## ill                0.000000              ill     TRUE
## ive                0.000000              ive     TRUE
## lost               0.000000             lost     TRUE
## noth               0.000000             noth     TRUE
## worst              0.000000            worst     TRUE
## phone              0.000000            phone     TRUE
## care               0.000000             care     TRUE
## way                0.000000              way     TRUE
## year               0.000000             year     TRUE
## updat              0.000000            updat     TRUE
## app                0.000000              app     TRUE
## iphon              0.000000            iphon     TRUE
## data               0.000000             data     TRUE
## take               0.000000             take     TRUE
## card               0.000000             card     TRUE
## custom             0.000000           custom     TRUE
## die                0.000000              die     TRUE
## event              0.000000            event     TRUE
## problem            0.000000          problem     TRUE
## refus              0.000000            refus     TRUE
## two                0.000000              two     TRUE
## use                0.000000              use     TRUE
## screen             0.000000           screen     TRUE
## thing              0.000000            thing     TRUE
## buy                0.000000              buy     TRUE
## tri                0.000000              tri     TRUE
## cdp                0.000000              cdp     TRUE
## doesnt             0.000000           doesnt     TRUE
## emiss              0.000000            emiss     TRUE
## feel               0.000000             feel     TRUE
## fun                0.000000              fun     TRUE
## got                0.000000              got     TRUE
## hour               0.000000             hour     TRUE
## macbook            0.000000          macbook     TRUE
## miss               0.000000             miss     TRUE
## siri               0.000000             siri     TRUE
## start              0.000000            start     TRUE
## upgrad             0.000000           upgrad     TRUE
## get                0.000000              get     TRUE
## think              0.000000            think     TRUE
## copi               0.000000             copi     TRUE
## guess              0.000000            guess     TRUE
## person             0.000000           person     TRUE
## smart              0.000000            smart     TRUE
## best               0.000000             best     TRUE
## differ             0.000000           differ     TRUE
## nsa                0.000000              nsa     TRUE
## product            0.000000          product     TRUE
## twitter            0.000000          twitter     TRUE
## old                0.000000              old     TRUE
## see                0.000000              see     TRUE
## download           0.000000         download     TRUE
## idea               0.000000             idea     TRUE
## job                0.000000              job     TRUE
## mani               0.000000             mani     TRUE
## simpl              0.000000            simpl     TRUE
## soon               0.000000             soon     TRUE
## team               0.000000             team     TRUE
## technolog          0.000000        technolog     TRUE
## wow                0.000000              wow     TRUE
## much               0.000000             much     TRUE
## support            0.000000          support     TRUE
## text               0.000000             text     TRUE
## time               0.000000             time     TRUE
## android            0.000000          android     TRUE
## that               0.000000             that     TRUE
## fingerprint        0.000000      fingerprint     TRUE
## now                0.000000              now     TRUE
## ask                0.000000              ask     TRUE
## colour             0.000000           colour     TRUE
## gonna              0.000000            gonna     TRUE
## happen             0.000000           happen     TRUE
## man                0.000000              man     TRUE
## give               0.000000             give     TRUE
## hey                0.000000              hey     TRUE
## today              0.000000            today     TRUE
## case               0.000000             case     TRUE
## chang              0.000000            chang     TRUE
## one                0.000000              one     TRUE
## work               0.000000             work     TRUE
## day                0.000000              day     TRUE
## nokia              0.000000            nokia     TRUE
## preorder           0.000000         preorder     TRUE
## look               0.000000             look     TRUE
## awesom             0.000000           awesom     TRUE
## done               0.000000             done     TRUE
## featur             0.000000           featur     TRUE
## said               0.000000             said     TRUE
## secur              0.000000            secur     TRUE
## seem               0.000000             seem     TRUE
## smartphon          0.000000        smartphon     TRUE
## thought            0.000000          thought     TRUE
## what               0.000000             what     TRUE
## windowsphon        0.000000      windowsphon     TRUE
## bit                0.000000              bit     TRUE
## realli             0.000000           realli     TRUE
## can                0.000000              can     TRUE
## sure               0.000000             sure     TRUE
## dont               0.000000             dont     TRUE
## impress            0.000000          impress     TRUE
## call               0.000000             call     TRUE
## didnt              0.000000            didnt     TRUE
## drop               0.000000             drop     TRUE
## fail               0.000000             fail     TRUE
## first              0.000000            first     TRUE
## improv             0.000000           improv     TRUE
## isnt               0.000000             isnt     TRUE
## made               0.000000             made     TRUE
## mean               0.000000             mean     TRUE
## put                0.000000              put     TRUE
## yet                0.000000              yet     TRUE
## next.              0.000000            next.     TRUE
## thank              0.000000            thank     TRUE
## market             0.000000           market     TRUE
## blackberri         0.000000       blackberri     TRUE
## mac                0.000000              mac     TRUE
## never              0.000000            never     TRUE
## plastic            0.000000          plastic     TRUE
## right              0.000000            right     TRUE
## sell               0.000000             sell     TRUE
## pleas              0.000000            pleas     TRUE
## guy                0.000000              guy     TRUE
## scanner            0.000000          scanner     TRUE
## want               0.000000             want     TRUE
## black              0.000000            black     TRUE
## free               0.000000             free     TRUE
## innov              0.000000            innov     TRUE
## via                0.000000              via     TRUE
## come               0.000000             come     TRUE
## good               0.000000             good     TRUE
## let                0.000000              let     TRUE
## stop               0.000000             stop     TRUE
## color              0.000000            color     TRUE
## compani            0.000000          compani     TRUE
## know               0.000000             know     TRUE
## price              0.000000            price     TRUE
## just               0.000000             just     TRUE
## store              0.000000            store     TRUE
## ios7               0.000000             ios7     TRUE
## peopl              0.000000            peopl     TRUE
## releas             0.000000           releas     TRUE
## say                0.000000              say     TRUE
## anyon              0.000000            anyon     TRUE
## devic              0.000000            devic     TRUE
## need               0.000000             need     TRUE
## back               0.000000             back     TRUE
## googl              0.000000            googl     TRUE
## itun               0.000000             itun     TRUE
## microsoft          0.000000        microsoft     TRUE
## will               0.000000             will     TRUE
## appl               0.000000             appl     TRUE
## new                0.000000              new     TRUE
## iphone5            0.000000          iphone5     TRUE
## ipad               0.000000             ipad     TRUE
## [1] "fitting model: Final.rpart"
## [1] "    indep_vars: freak, hate, wtf, stuff, fix, pictur, cant, better, shame, stupid, even, line, yooo, ever, charger, still, charg, disappoint, short, like, amazon, yall, break., imessag, stand, togeth, cheap, wont, make, carbon, darn, httpbitly18xc8dk, dear, facebook, X7evenstarz, condom, femal, money, theyr, batteri, china, turn, hope, life, sinc, steve, switch, everi, last, your, amaz, arent, date, divulg, ill, ive, lost, noth, worst, phone, care, way, year, updat, app, iphon, data, take, card, custom, die, event, problem, refus, two, use, screen, thing, buy, tri, cdp, doesnt, emiss, feel, fun, got, hour, macbook, miss, siri, start, upgrad, get, think, copi, guess, person, smart, best, differ, nsa, product, twitter, old, see, download, idea, job, mani, simpl, soon, team, technolog, wow, much, support, text, time, android, that, fingerprint, now, ask, colour, gonna, happen, man, give, hey, today, case, chang, one, work, day, nokia, preorder, look, awesom, done, featur, said, secur, seem, smartphon, thought, what, windowsphon, bit, realli, can, sure, dont, impress, call, didnt, drop, fail, first, improv, isnt, made, mean, put, yet, next., thank, market, blackberri, mac, never, plastic, right, sell, pleas, guy, scanner, want, black, free, innov, via, come, good, let, stop, color, compani, know, price, just, store, ios7, peopl, releas, say, anyon, devic, need, back, googl, itun, microsoft, will, appl, new, iphone5, ipad"
## + Fold1: cp=0 
## - Fold1: cp=0 
## + Fold2: cp=0 
## - Fold2: cp=0 
## + Fold3: cp=0 
## - Fold3: cp=0 
## Aggregating results
## Fitting final model on full training set
```

![](Apple_Tweets_files/figure-html/fit.data.training.all_0-1.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 825 
## 
##           CP nsplit rel error
## 1 0.21428571      0 1.0000000
## 2 0.04761905      1 0.7857143
## 3 0.02380952      2 0.7380952
## 4 0.00000000      3 0.7142857
## 
## Variable importance
## freak  hate stuff 
##    72    17    10 
## 
## Node number 1: 825 observations,    complexity param=0.2142857
##   predicted class=N  expected loss=0.1527273  P(node) =1
##     class counts:   699   126
##    probabilities: 0.847 0.153 
##   left son=2 (790 obs) right son=3 (35 obs)
##   Primary splits:
##       freak  < 0.5 to the left,  improve=39.275110, (0 missing)
##       hate   < 0.5 to the left,  improve= 9.873907, (0 missing)
##       stuff  < 0.5 to the left,  improve= 6.063648, (0 missing)
##       wtf    < 0.5 to the left,  improve= 5.763645, (0 missing)
##       pictur < 0.5 to the left,  improve= 5.763645, (0 missing)
## 
## Node number 2: 790 observations,    complexity param=0.04761905
##   predicted class=N  expected loss=0.1202532  P(node) =0.9575758
##     class counts:   695    95
##    probabilities: 0.880 0.120 
##   left son=4 (780 obs) right son=5 (10 obs)
##   Primary splits:
##       hate  < 0.5 to the left,  improve=9.359591, (0 missing)
##       wtf   < 0.5 to the left,  improve=6.410211, (0 missing)
##       stuff < 0.5 to the left,  improve=5.436150, (0 missing)
##       fix   < 0.5 to the left,  improve=2.875307, (0 missing)
##       cant  < 0.5 to the left,  improve=2.709704, (0 missing)
## 
## Node number 3: 35 observations
##   predicted class=Y  expected loss=0.1142857  P(node) =0.04242424
##     class counts:     4    31
##    probabilities: 0.114 0.886 
## 
## Node number 4: 780 observations,    complexity param=0.02380952
##   predicted class=N  expected loss=0.1115385  P(node) =0.9454545
##     class counts:   693    87
##    probabilities: 0.888 0.112 
##   left son=8 (771 obs) right son=9 (9 obs)
##   Primary splits:
##       stuff  < 0.5 to the left,  improve=5.611763, (0 missing)
##       wtf    < 0.5 to the left,  improve=5.132319, (0 missing)
##       fix    < 0.5 to the left,  improve=2.987798, (0 missing)
##       cant   < 0.5 to the left,  improve=2.865713, (0 missing)
##       better < 0.5 to the left,  improve=2.865713, (0 missing)
## 
## Node number 5: 10 observations
##   predicted class=Y  expected loss=0.2  P(node) =0.01212121
##     class counts:     2     8
##    probabilities: 0.200 0.800 
## 
## Node number 8: 771 observations
##   predicted class=N  expected loss=0.1050584  P(node) =0.9345455
##     class counts:   690    81
##    probabilities: 0.895 0.105 
## 
## Node number 9: 9 observations
##   predicted class=Y  expected loss=0.3333333  P(node) =0.01090909
##     class counts:     3     6
##    probabilities: 0.333 0.667 
## 
## n= 825 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 825 126 N (0.8472727 0.1527273)  
##   2) freak< 0.5 790  95 N (0.8797468 0.1202532)  
##     4) hate< 0.5 780  87 N (0.8884615 0.1115385)  
##       8) stuff< 0.5 771  81 N (0.8949416 0.1050584) *
##       9) stuff>=0.5 9   3 Y (0.3333333 0.6666667) *
##     5) hate>=0.5 10   2 Y (0.2000000 0.8000000) *
##   3) freak>=0.5 35   4 Y (0.1142857 0.8857143) *
```

![](Apple_Tweets_files/figure-html/fit.data.training.all_0-2.png) 

```
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 699 126
##   Negative.fctr Negative.fctr.predict.Final.rpart.N
## 1             N                                   0
## 2             Y                                   0
##   Negative.fctr.predict.Final.rpart.Y
## 1                                 699
## 2                                 126
##           Reference
## Prediction   N   Y
##          N   0   0
##          Y 699 126
##   Negative.fctr Negative.fctr.predict.Final.rpart.N
## 1             N                                   0
## 2             Y                                   0
##   Negative.fctr.predict.Final.rpart.Y
## 1                                 699
## 2                                 126
##           Reference
## Prediction   N   Y
##          N 690  81
##          Y   9  45
##   Negative.fctr Negative.fctr.predict.Final.rpart.N
## 1             N                                 690
## 2             Y                                  81
##   Negative.fctr.predict.Final.rpart.Y
## 1                                   9
## 2                                  45
##           Reference
## Prediction   N   Y
##          N 690  81
##          Y   9  45
##   Negative.fctr Negative.fctr.predict.Final.rpart.N
## 1             N                                 690
## 2             Y                                  81
##   Negative.fctr.predict.Final.rpart.Y
## 1                                   9
## 2                                  45
##           Reference
## Prediction   N   Y
##          N 690  81
##          Y   9  45
##   Negative.fctr Negative.fctr.predict.Final.rpart.N
## 1             N                                 690
## 2             Y                                  81
##   Negative.fctr.predict.Final.rpart.Y
## 1                                   9
## 2                                  45
##           Reference
## Prediction   N   Y
##          N 690  81
##          Y   9  45
##   Negative.fctr Negative.fctr.predict.Final.rpart.N
## 1             N                                 690
## 2             Y                                  81
##   Negative.fctr.predict.Final.rpart.Y
## 1                                   9
## 2                                  45
##           Reference
## Prediction   N   Y
##          N 690  81
##          Y   9  45
##   Negative.fctr Negative.fctr.predict.Final.rpart.N
## 1             N                                 690
## 2             Y                                  81
##   Negative.fctr.predict.Final.rpart.Y
## 1                                   9
## 2                                  45
##           Reference
## Prediction   N   Y
##          N 693  87
##          Y   6  39
##   Negative.fctr Negative.fctr.predict.Final.rpart.N
## 1             N                                 693
## 2             Y                                  87
##   Negative.fctr.predict.Final.rpart.Y
## 1                                   6
## 2                                  39
##           Reference
## Prediction   N   Y
##          N 693  87
##          Y   6  39
##   Negative.fctr Negative.fctr.predict.Final.rpart.N
## 1             N                                 693
## 2             Y                                  87
##   Negative.fctr.predict.Final.rpart.Y
## 1                                   6
## 2                                  39
##           Reference
## Prediction   N   Y
##          N 699 126
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Final.rpart.N
## 1             N                                 699
## 2             Y                                 126
##   Negative.fctr.predict.Final.rpart.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction   N   Y
##          N 699 126
##          Y   0   0
##   Negative.fctr Negative.fctr.predict.Final.rpart.N
## 1             N                                 699
## 2             Y                                 126
##   Negative.fctr.predict.Final.rpart.Y
## 1                                   0
## 2                                   0
##    threshold   f.score
## 1        0.0 0.2649842
## 2        0.1 0.2649842
## 3        0.2 0.5000000
## 4        0.3 0.5000000
## 5        0.4 0.5000000
## 6        0.5 0.5000000
## 7        0.6 0.5000000
## 8        0.7 0.4561404
## 9        0.8 0.4561404
## 10       0.9 0.0000000
## 11       1.0 0.0000000
```

```
## [1] "Classifier Probability Threshold: 0.6000 to maximize f.score.fit"
##   Negative.fctr Negative.fctr.predict.Final.rpart.N
## 1             N                                 690
## 2             Y                                  81
##   Negative.fctr.predict.Final.rpart.Y
## 1                                   9
## 2                                  45
##           Reference
## Prediction   N   Y
##          N 690  81
##          Y   9  45
##   Negative.fctr Negative.fctr.predict.Final.rpart.N
## 1             N                                 690
## 2             Y                                  81
##   Negative.fctr.predict.Final.rpart.Y
## 1                                   9
## 2                                  45
##          Prediction
## Reference   N   Y
##         N 690   9
##         Y  81  45
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   8.909091e-01   4.495596e-01   8.676171e-01   9.113614e-01   8.472727e-01 
## AccuracyPValue  McnemarPValue 
##   1.744490e-04   7.206261e-14
```

```
## Warning in mypredict_mdl(mdl, df = fit_df, rsp_var, rsp_var_out,
## model_id_method, : Expecting 1 metric: Accuracy; recd: Accuracy, Kappa;
## retaining Accuracy only
```

![](Apple_Tweets_files/figure-html/fit.data.training.all_0-3.png) 

```
##      model_id model_method
## 1 Final.rpart        rpart
##                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              feats
## 1 freak, hate, wtf, stuff, fix, pictur, cant, better, shame, stupid, even, line, yooo, ever, charger, still, charg, disappoint, short, like, amazon, yall, break., imessag, stand, togeth, cheap, wont, make, carbon, darn, httpbitly18xc8dk, dear, facebook, X7evenstarz, condom, femal, money, theyr, batteri, china, turn, hope, life, sinc, steve, switch, everi, last, your, amaz, arent, date, divulg, ill, ive, lost, noth, worst, phone, care, way, year, updat, app, iphon, data, take, card, custom, die, event, problem, refus, two, use, screen, thing, buy, tri, cdp, doesnt, emiss, feel, fun, got, hour, macbook, miss, siri, start, upgrad, get, think, copi, guess, person, smart, best, differ, nsa, product, twitter, old, see, download, idea, job, mani, simpl, soon, team, technolog, wow, much, support, text, time, android, that, fingerprint, now, ask, colour, gonna, happen, man, give, hey, today, case, chang, one, work, day, nokia, preorder, look, awesom, done, featur, said, secur, seem, smartphon, thought, what, windowsphon, bit, realli, can, sure, dont, impress, call, didnt, drop, fail, first, improv, isnt, made, mean, put, yet, next., thank, market, blackberri, mac, never, plastic, right, sell, pleas, guy, scanner, want, black, free, innov, via, come, good, let, stop, color, compani, know, price, just, store, ios7, peopl, releas, say, anyon, devic, need, back, googl, itun, microsoft, will, appl, new, iphone5, ipad
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               1                        2.3                 0.418
##   max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1   0.6727638                    0.6             0.5        0.8812121
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit
## 1             0.8676171             0.9113614      0.366133
##   max.AccuracySD.fit max.KappaSD.fit
## 1         0.01376705      0.09747058
```

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="fit.data.training.all", 
    chunk_step_major=glb_script_df[nrow(glb_script_df), "chunk_step_major"], 
    chunk_step_minor=glb_script_df[nrow(glb_script_df), "chunk_step_minor"]+1,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                     chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed10 fit.data.training.all                6                0 313.567
## elapsed11 fit.data.training.all                6                1 319.780
```


```r
glb_rsp_var_out <- paste0(glb_rsp_var_out, tail(names(glb_models_lst), 1))

# Used again in predict.data.new chunk
glb_get_predictions <- function(df) {
    if (glb_is_regression) {
        df[, glb_rsp_var_out] <- predict(glb_fin_mdl, newdata=df, type="raw")
        print(myplot_scatter(df, glb_rsp_var, glb_rsp_var_out, 
                             smooth=TRUE))
        df[, paste0(glb_rsp_var_out, ".err")] <- 
            abs(df[, glb_rsp_var_out] - df[, glb_rsp_var])
        print(head(orderBy(reformulate(c("-", paste0(glb_rsp_var_out, ".err"))), 
                           df)))                             
    }

    if (glb_is_classification && glb_is_binomial) {
        # incorporate glb_clf_proba_threshold
        #   shd it only be for glb_fin_mdl or for earlier models ?
        if (glb_models_df[glb_models_df$model_id == glb_fin_mdl_id, 
                          "opt.prob.threshold.fit"] != 
            glb_models_df[glb_models_df$model_id == glb_sel_mdl_id, 
                          "opt.prob.threshold.fit"])
            stop("user specification for probability threshold required")
        else prob_threshold <- 
    glb_models_df[glb_models_df$model_id == glb_sel_mdl_id, "opt.prob.threshold.OOB"]
        
        df[, paste0(glb_rsp_var_out, ".prob")] <- 
            predict(glb_fin_mdl, newdata=df, type="prob")[, 2]
        df[, glb_rsp_var_out] <- 
    			factor(levels(df[, glb_rsp_var])[
    				(df[, paste0(glb_rsp_var_out, ".prob")] >=
    					prob_threshold) * 1 + 1], levels(df[, glb_rsp_var]))
    
        # prediction stats already reported by myfit_mdl ???
    }    
    
    if (glb_is_classification && !glb_is_binomial) {
        df[, glb_rsp_var_out] <- predict(glb_fin_mdl, newdata=df, type="raw")
    }

    return(df)
}    
glb_trnent_df <- glb_get_predictions(df=glb_trnent_df)

print(glb_feats_df <- mymerge_feats_importance(feats_df=glb_feats_df, sel_mdl=glb_fin_mdl, 
                                               entity_df=glb_trnent_df))
```

```
##                       id         cor.y exclude.as.feat    cor.y.abs
## 96                 freak  0.4220126658               0 0.4220126658
## 113                 hate  0.2150466216               0 0.2150466216
## 304                  wtf  0.1642996876               0 0.1642996876
## 249                stuff  0.1685214122               0 0.1685214122
## 94                   fix  0.1076729878               0 0.1076729878
## 198               pictur  0.1642996876               0 0.1642996876
## 35                  cant  0.1518891001               0 0.1518891001
## 23                better  0.1114365401               0 0.1114365401
## 6                   amaz  0.0536765239               0 0.0536765239
## 7                 amazon  0.0851663937               0 0.0851663937
## 8                android  0.0149137629               0 0.0149137629
## 10                 anyon -0.0305482407               0 0.0305482407
## 11                   app  0.0506802204               0 0.0506802204
## 12                  appl -0.0533458207               0 0.0533458207
## 14                 arent  0.0536765239               0 0.0536765239
## 15                   ask  0.0102616884               0 0.0102616884
## 19                awesom  0.0033167112               0 0.0033167112
## 20                  back -0.0405914861               0 0.0405914861
## 21               batteri  0.0626301156               0 0.0626301156
## 22                  best  0.0267580922               0 0.0267580922
## 25                   bit  0.0028697294               0 0.0028697294
## 26                 black -0.0148064982               0 0.0148064982
## 27            blackberri -0.0076273067               0 0.0076273067
## 28                break.  0.0826293697               0 0.0826293697
## 32                   buy  0.0380283000               0 0.0380283000
## 33                  call -0.0025381970               0 0.0025381970
## 34                   can  0.0017524556               0 0.0017524556
## 36                carbon  0.0709359262               0 0.0709359262
## 37                  card  0.0429730405               0 0.0429730405
## 38                  care  0.0527276703               0 0.0527276703
## 39                  case  0.0086651648               0 0.0086651648
## 40                   cdp  0.0341988646               0 0.0341988646
## 41                 chang  0.0086651648               0 0.0086651648
## 42                 charg  0.0970913594               0 0.0970913594
## 43               charger  0.1069355141               0 0.1069355141
## 44                 cheap  0.0769587757               0 0.0769587757
## 45                 china  0.0625002078               0 0.0625002078
## 46                 color -0.0199770086               0 0.0199770086
## 47                colour  0.0102616884               0 0.0102616884
## 48                  come -0.0172779254               0 0.0172779254
## 49               compani -0.0199770086               0 0.0199770086
## 51                condom  0.0673843716               0 0.0673843716
## 52                  copi  0.0303310278               0 0.0303310278
## 55                custom  0.0429730405               0 0.0429730405
## 56                  darn  0.0709359262               0 0.0709359262
## 57                  data  0.0453496159               0 0.0453496159
## 58                  date  0.0536765239               0 0.0536765239
## 59                   day  0.0047078144               0 0.0047078144
## 60                  dear  0.0681568527               0 0.0681568527
## 63                 devic -0.0325564924               0 0.0325564924
## 64                 didnt -0.0025381970               0 0.0025381970
## 65                   die  0.0429730405               0 0.0429730405
## 66                differ  0.0267580922               0 0.0267580922
## 67            disappoint  0.0970913594               0 0.0970913594
## 69                divulg  0.0536765239               0 0.0536765239
## 70                doesnt  0.0341988646               0 0.0341988646
## 71                  done  0.0033167112               0 0.0033167112
## 72                  dont -0.0006481129               0 0.0006481129
## 73              download  0.0188746800               0 0.0188746800
## 74                  drop -0.0025381970               0 0.0025381970
## 76                 emiss  0.0341988646               0 0.0341988646
## 78                  even  0.1158940633               0 0.1158940633
## 79                 event  0.0429730405               0 0.0429730405
## 80                  ever  0.1076729878               0 0.1076729878
## 81                 everi  0.0564127109               0 0.0564127109
## 83              facebook  0.0681568527               0 0.0681568527
## 84                  fail -0.0025381970               0 0.0025381970
## 85                featur  0.0033167112               0 0.0033167112
## 86                  feel  0.0341988646               0 0.0341988646
## 87                 femal  0.0673843716               0 0.0673843716
## 91           fingerprint  0.0129942941               0 0.0129942941
## 93                 first -0.0025381970               0 0.0025381970
## 97                  free -0.0162362822               0 0.0162362822
## 98                   fun  0.0341988646               0 0.0341988646
## 101                  get  0.0332332539               0 0.0332332539
## 102                 give  0.0094009452               0 0.0094009452
## 104                gonna  0.0102616884               0 0.0102616884
## 105                 good -0.0199770086               0 0.0199770086
## 106                googl -0.0405914861               0 0.0405914861
## 107                  got  0.0341988646               0 0.0341988646
## 109                guess  0.0303310278               0 0.0303310278
## 110                  guy -0.0109788835               0 0.0109788835
## 111               happen  0.0102616884               0 0.0102616884
## 115                  hey  0.0094009452               0 0.0094009452
## 116                 hope  0.0611434911               0 0.0611434911
## 117                 hour  0.0341988646               0 0.0341988646
## 118     httpbitly18xc8dk  0.0709359262               0 0.0709359262
## 120                 idea  0.0188746800               0 0.0188746800
## 121                  ill  0.0536765239               0 0.0536765239
## 122              imessag  0.0826293697               0 0.0826293697
## 123              impress -0.0022363540               0 0.0022363540
## 124               improv -0.0025381970               0 0.0025381970
## 125                innov -0.0162362822               0 0.0162362822
## 128                 ios7 -0.0266634900               0 0.0266634900
## 129                 ipad -0.0646730918               0 0.0646730918
## 130                iphon  0.0473646206               0 0.0473646206
## 132              iphone5 -0.0621379328               0 0.0621379328
## 137                 isnt -0.0025381970               0 0.0025381970
## 138                 itun -0.0449613971               0 0.0449613971
## 139                  ive  0.0536765239               0 0.0536765239
## 140                  job  0.0188746800               0 0.0188746800
## 141                 just -0.0237594870               0 0.0237594870
## 143                 know -0.0199770086               0 0.0199770086
## 144                 last  0.0549744869               0 0.0549744869
## 146                  let -0.0199770086               0 0.0199770086
## 147                 life  0.0611434911               0 0.0611434911
## 148                 like  0.0962377709               0 0.0962377709
## 149                 line  0.1158940633               0 0.1158940633
## 153                 look  0.0035739823               0 0.0035739823
## 155                 lost  0.0536765239               0 0.0536765239
## 157                  mac -0.0076273067               0 0.0076273067
## 158              macbook  0.0341988646               0 0.0341988646
## 159                 made -0.0025381970               0 0.0025381970
## 160                 make  0.0727362480               0 0.0727362480
## 161                  man  0.0102616884               0 0.0102616884
## 162                 mani  0.0188746800               0 0.0188746800
## 163               market -0.0068841106               0 0.0068841106
## 165                 mean -0.0025381970               0 0.0025381970
## 166            microsoft -0.0450714475               0 0.0450714475
## 168                 miss  0.0341988646               0 0.0341988646
## 170                money  0.0673843716               0 0.0673843716
## 173                 much  0.0183336696               0 0.0183336696
## 176                 need -0.0357398233               0 0.0357398233
## 177                never -0.0076273067               0 0.0076273067
## 178                  new -0.0536221452               0 0.0536221452
## 180                next. -0.0036050106               0 0.0036050106
## 182                nokia  0.0047078144               0 0.0047078144
## 183                 noth  0.0536765239               0 0.0536765239
## 184                  now  0.0129148701               0 0.0129148701
## 185                  nsa  0.0267580922               0 0.0267580922
## 188                  old  0.0202889470               0 0.0202889470
## 189                  one  0.0070552161               0 0.0070552161
## 192                peopl -0.0266634900               0 0.0266634900
## 194               person  0.0303310278               0 0.0303310278
## 195                phone  0.0536044876               0 0.0536044876
## 199              plastic -0.0076273067               0 0.0076273067
## 201                pleas -0.0108398425               0 0.0108398425
## 203             preorder  0.0047078144               0 0.0047078144
## 204                price -0.0216753537               0 0.0216753537
## 207              problem  0.0429730405               0 0.0429730405
## 208              product  0.0267580922               0 0.0267580922
## 211                  put -0.0025381970               0 0.0025381970
## 215               realli  0.0020955452               0 0.0020955452
## 217                refus  0.0429730405               0 0.0429730405
## 218               releas -0.0266634900               0 0.0266634900
## 219                right -0.0076273067               0 0.0076273067
## 220                 said  0.0033167112               0 0.0033167112
## 223                  say -0.0266634900               0 0.0266634900
## 224              scanner -0.0121497764               0 0.0121497764
## 225               screen  0.0387788990               0 0.0387788990
## 226                secur  0.0033167112               0 0.0033167112
## 227                  see  0.0202889470               0 0.0202889470
## 228                 seem  0.0033167112               0 0.0033167112
## 229                 sell -0.0076273067               0 0.0076273067
## 232                shame  0.1405061949               0 0.1405061949
## 234                short  0.0970913594               0 0.0970913594
## 236                simpl  0.0188746800               0 0.0188746800
## 237                 sinc  0.0611434911               0 0.0611434911
## 238                 siri  0.0341988646               0 0.0341988646
## 239                smart  0.0303310278               0 0.0303310278
## 240            smartphon  0.0033167112               0 0.0033167112
## 242                 soon  0.0188746800               0 0.0188746800
## 243                stand  0.0826293697               0 0.0826293697
## 244                start  0.0341988646               0 0.0341988646
## 245                steve  0.0611434911               0 0.0611434911
## 246                still  0.1007505594               0 0.1007505594
## 247                 stop -0.0199770086               0 0.0199770086
## 248                store -0.0259113719               0 0.0259113719
## 250               stupid  0.1186461031               0 0.1186461031
## 252              support  0.0183336696               0 0.0183336696
## 253                 sure  0.0003935570               0 0.0003935570
## 254               switch  0.0611434911               0 0.0611434911
## 255                 take  0.0453496159               0 0.0453496159
## 257                 team  0.0188746800               0 0.0188746800
## 259            technolog  0.0188746800               0 0.0188746800
## 261                 text  0.0183336696               0 0.0183336696
## 262                thank -0.0049626450               0 0.0049626450
## 263                 that  0.0145566668               0 0.0145566668
## 264                theyr  0.0673843716               0 0.0673843716
## 265                thing  0.0387788990               0 0.0387788990
## 266                think  0.0314859024               0 0.0314859024
## 268              thought  0.0033167112               0 0.0033167112
## 269                 time  0.0159958708               0 0.0159958708
## 270                today  0.0094009452               0 0.0094009452
## 271               togeth  0.0826293697               0 0.0826293697
## 274                  tri  0.0356343761               0 0.0356343761
## 276                 turn  0.0625002078               0 0.0625002078
## 278              twitter  0.0207137058               0 0.0207137058
## 279                  two  0.0429730405               0 0.0429730405
## 280                updat  0.0519176496               0 0.0519176496
## 281               upgrad  0.0341988646               0 0.0341988646
## 282                  use  0.0401415689               0 0.0401415689
## 284                  via -0.0162362822               0 0.0162362822
## 287                 want -0.0141452654               0 0.0141452654
## 289                  way  0.0527276703               0 0.0527276703
## 292                 what  0.0033167112               0 0.0033167112
## 294                 will -0.0463979478               0 0.0463979478
## 295          windowsphon  0.0033167112               0 0.0033167112
## 299                 wont  0.0758516866               0 0.0758516866
## 300                 work  0.0057872663               0 0.0057872663
## 302                worst  0.0536765239               0 0.0536765239
## 303                  wow  0.0188746800               0 0.0188746800
## 306          X7evenstarz  0.0673843716               0 0.0673843716
## 307                 yall  0.0851663937               0 0.0851663937
## 308                 year  0.0527276703               0 0.0527276703
## 310                  yet -0.0025381970               0 0.0025381970
## 311                 yooo  0.1158940633               0 0.1158940633
## 312                 your  0.0546349290               0 0.0546349290
## 1                 .rnorm -0.0127552181               0 0.0127552181
## 2                 actual -0.0346046348               0 0.0346046348
## 3                    add -0.0256490570               0 0.0256490570
## 4                alreadi -0.0256490570               0 0.0256490570
## 5                  alway -0.0296350116               0 0.0296350116
## 9                announc -0.0557827956               0 0.0557827956
## 13               appstor -0.0331531471               0 0.0331531471
## 16                 avail -0.0420127055               0 0.0420127055
## 17                   Avg -0.7061583707               1 0.7061583707
## 18                  away -0.0331531471               0 0.0331531471
## 24                   big -0.0392752586               0 0.0392752586
## 29                 bring -0.0346046348               0 0.0346046348
## 30              burberri -0.0375405647               0 0.0375405647
## 31                  busi -0.0420127055               0 0.0420127055
## 50            condescens -0.0296350116               0 0.0296350116
## 53                 crack -0.0296350116               0 0.0296350116
## 54                 creat -0.0331531471               0 0.0331531471
## 61                design -0.0331531471               0 0.0331531471
## 62               develop -0.0375405647               0 0.0375405647
## 68            discontinu -0.0296350116               0 0.0296350116
## 75                 email -0.0296350116               0 0.0296350116
## 77                 emoji -0.0392752586               0 0.0392752586
## 82               everyth -0.0209296403               0 0.0209296403
## 88                 figur -0.0296350116               0 0.0296350116
## 89                 final -0.0256490570               0 0.0256490570
## 90                finger -0.0445884997               0 0.0445884997
## 92                  fire -0.0363396181               0 0.0363396181
## 95                follow -0.0420127055               0 0.0420127055
## 99               generat -0.0420127055               0 0.0420127055
## 100               genius -0.0256490570               0 0.0256490570
## 103                 gold -0.0428877879               0 0.0428877879
## 108                great -0.0363396181               0 0.0363396181
## 112                happi -0.0331531471               0 0.0331531471
## 114                 help -0.0470292313               0 0.0470292313
## 119           ibrooklynb -0.0331531471               0 0.0331531471
## 126              instead -0.0445884997               0 0.0445884997
## 127             internet -0.0296350116               0 0.0296350116
## 131              iphone4 -0.0296350116               0 0.0296350116
## 133             iphone5c -0.0645077026               0 0.0645077026
## 134               iphoto -0.0392752586               0 0.0392752586
## 135                 ipod -0.0852874403               0 0.0852874403
## 136      ipodplayerpromo -0.0793460925               0 0.0793460925
## 142               keynot -0.0363396181               0 0.0363396181
## 145               launch -0.0331531471               0 0.0331531471
## 150                 lmao -0.0296350116               0 0.0296350116
## 151                 lock -0.0331531471               0 0.0331531471
## 152                  lol -0.0537205371               0 0.0537205371
## 154                  los -0.0296350116               0 0.0296350116
## 156                 love -0.0577763105               0 0.0577763105
## 164                 mayb -0.0363396181               0 0.0363396181
## 167              mishiza -0.0363396181               0 0.0363396181
## 169                mobil -0.0428877879               0 0.0428877879
## 171             motorola -0.0392752586               0 0.0392752586
## 172                 move -0.0296350116               0 0.0296350116
## 174                music -0.0209296403               0 0.0209296403
## 175             natz0711 -0.0363396181               0 0.0363396181
## 179                 news -0.0453528423               0 0.0453528423
## 181                  nfc -0.0256490570               0 0.0256490570
## 186                nuevo -0.0296350116               0 0.0296350116
## 187                offer -0.0363396181               0 0.0363396181
## 190                 page -0.0331531471               0 0.0331531471
## 191                 para -0.0363396181               0 0.0363396181
## 193              perfect -0.0296350116               0 0.0296350116
## 196               photog -0.0296350116               0 0.0296350116
## 197          photographi -0.0296350116               0 0.0296350116
## 200                 play -0.0331531471               0 0.0331531471
## 202                  ppl -0.0223147692               0 0.0223147692
## 205                print -0.0470292313               0 0.0470292313
## 206                  pro -0.0256490570               0 0.0256490570
## 209                promo -0.0615836556               0 0.0615836556
## 210 promoipodplayerpromo -0.0838913688               0 0.0838913688
## 212                  que -0.0428877879               0 0.0428877879
## 213                quiet -0.0296350116               0 0.0296350116
## 214                 read -0.0209296403               0 0.0209296403
## 216            recommend -0.0296350116               0 0.0296350116
## 221              samsung -0.0615836556               0 0.0615836556
## 222            samsungsa -0.0363396181               0 0.0363396181
## 230                 send -0.0296350116               0 0.0296350116
## 231               servic -0.0470292313               0 0.0470292313
## 233                share -0.0346046348               0 0.0346046348
## 235                 show -0.0363396181               0 0.0363396181
## 241               someth -0.0331531471               0 0.0331531471
## 251                 suck  0.1644037549               0 0.1644037549
## 256                 talk -0.0331531471               0 0.0331531471
## 258                 tech -0.0470292313               0 0.0470292313
## 260                 tell -0.0470292313               0 0.0470292313
## 267                  tho -0.0296350116               0 0.0296350116
## 272                touch -0.0331531471               0 0.0331531471
## 273              touchid -0.0392752586               0 0.0392752586
## 275                 true -0.0296350116               0 0.0296350116
## 277           Tweet.fctr  0.6354986940               1 0.6354986940
## 283                 user -0.0296350116               0 0.0296350116
## 285                video -0.0375405647               0 0.0375405647
## 286                 wait -0.0445884997               0 0.0445884997
## 288                watch -0.0363396181               0 0.0363396181
## 290                 week -0.0375405647               0 0.0375405647
## 291                 well -0.0506130885               0 0.0506130885
## 293                white -0.0346046348               0 0.0346046348
## 296                 wish -0.0420127055               0 0.0420127055
## 297              without -0.0209296403               0 0.0209296403
## 298               wonder -0.0331531471               0 0.0331531471
## 301                world -0.0363396181               0 0.0363396181
## 305       X244tsuyoponzu -0.0331531471               0 0.0331531471
## 309                  yes -0.0256490570               0 0.0256490570
##           cor.high.X is.ConditionalX.y is.cor.y.abs.low importance
## 96              <NA>              TRUE            FALSE 100.000000
## 113             <NA>              TRUE            FALSE  48.971207
## 304             <NA>              TRUE            FALSE  44.063971
## 249             <NA>              TRUE            FALSE  43.568454
## 94              <NA>              TRUE            FALSE  14.928296
## 198             <NA>              TRUE            FALSE  14.675057
## 35              <NA>              TRUE            FALSE  14.195802
## 23              <NA>              TRUE            FALSE   7.296511
## 6               <NA>              TRUE            FALSE   0.000000
## 7                cdp              TRUE            FALSE   0.000000
## 8               <NA>              TRUE            FALSE   0.000000
## 10              <NA>              TRUE            FALSE   0.000000
## 11              <NA>              TRUE            FALSE   0.000000
## 12              <NA>              TRUE            FALSE   0.000000
## 14              <NA>              TRUE            FALSE   0.000000
## 15              <NA>              TRUE             TRUE   0.000000
## 19              <NA>              TRUE             TRUE   0.000000
## 20              <NA>              TRUE            FALSE   0.000000
## 21              <NA>              TRUE            FALSE   0.000000
## 22              <NA>              TRUE            FALSE   0.000000
## 25              <NA>              TRUE             TRUE   0.000000
## 26              <NA>              TRUE            FALSE   0.000000
## 27              <NA>              TRUE             TRUE   0.000000
## 28              <NA>              TRUE            FALSE   0.000000
## 32              <NA>              TRUE            FALSE   0.000000
## 33              <NA>              TRUE             TRUE   0.000000
## 34              <NA>              TRUE             TRUE   0.000000
## 36  httpbitly18xc8dk              TRUE            FALSE   0.000000
## 37              <NA>              TRUE            FALSE   0.000000
## 38              <NA>              TRUE            FALSE   0.000000
## 39              <NA>              TRUE             TRUE   0.000000
## 40              <NA>              TRUE            FALSE   0.000000
## 41              <NA>              TRUE             TRUE   0.000000
## 42              <NA>              TRUE            FALSE   0.000000
## 43              <NA>              TRUE            FALSE   0.000000
## 44              <NA>              TRUE            FALSE   0.000000
## 45              <NA>              TRUE            FALSE   0.000000
## 46              <NA>              TRUE            FALSE   0.000000
## 47              <NA>              TRUE             TRUE   0.000000
## 48              <NA>              TRUE            FALSE   0.000000
## 49              <NA>              TRUE            FALSE   0.000000
## 51             femal              TRUE            FALSE   0.000000
## 52              <NA>              TRUE            FALSE   0.000000
## 55              <NA>              TRUE            FALSE   0.000000
## 56              <NA>              TRUE            FALSE   0.000000
## 57              <NA>              TRUE            FALSE   0.000000
## 58              <NA>              TRUE            FALSE   0.000000
## 59              <NA>              TRUE             TRUE   0.000000
## 60              <NA>              TRUE            FALSE   0.000000
## 63              <NA>              TRUE            FALSE   0.000000
## 64              <NA>              TRUE             TRUE   0.000000
## 65              <NA>              TRUE            FALSE   0.000000
## 66              <NA>              TRUE            FALSE   0.000000
## 67              <NA>              TRUE            FALSE   0.000000
## 69             refus              TRUE            FALSE   0.000000
## 70              <NA>              TRUE            FALSE   0.000000
## 71              <NA>              TRUE             TRUE   0.000000
## 72              <NA>              TRUE             TRUE   0.000000
## 73              <NA>              TRUE            FALSE   0.000000
## 74              <NA>              TRUE             TRUE   0.000000
## 76              <NA>              TRUE            FALSE   0.000000
## 78              <NA>              TRUE            FALSE   0.000000
## 79              <NA>              TRUE            FALSE   0.000000
## 80              <NA>              TRUE            FALSE   0.000000
## 81              <NA>              TRUE            FALSE   0.000000
## 83              <NA>              TRUE            FALSE   0.000000
## 84              <NA>              TRUE             TRUE   0.000000
## 85              <NA>              TRUE             TRUE   0.000000
## 86              <NA>              TRUE            FALSE   0.000000
## 87              <NA>              TRUE            FALSE   0.000000
## 91              <NA>              TRUE            FALSE   0.000000
## 93              <NA>              TRUE             TRUE   0.000000
## 97              <NA>              TRUE            FALSE   0.000000
## 98              <NA>              TRUE            FALSE   0.000000
## 101             <NA>              TRUE            FALSE   0.000000
## 102             <NA>              TRUE             TRUE   0.000000
## 104             <NA>              TRUE             TRUE   0.000000
## 105             <NA>              TRUE            FALSE   0.000000
## 106             <NA>              TRUE            FALSE   0.000000
## 107             <NA>              TRUE            FALSE   0.000000
## 109             <NA>              TRUE            FALSE   0.000000
## 110             <NA>              TRUE             TRUE   0.000000
## 111             <NA>              TRUE             TRUE   0.000000
## 115             <NA>              TRUE             TRUE   0.000000
## 116             <NA>              TRUE            FALSE   0.000000
## 117             <NA>              TRUE            FALSE   0.000000
## 118             <NA>              TRUE            FALSE   0.000000
## 120             <NA>              TRUE            FALSE   0.000000
## 121             <NA>              TRUE            FALSE   0.000000
## 122             <NA>              TRUE            FALSE   0.000000
## 123             <NA>              TRUE             TRUE   0.000000
## 124             <NA>              TRUE             TRUE   0.000000
## 125             <NA>              TRUE            FALSE   0.000000
## 128             <NA>              TRUE            FALSE   0.000000
## 129             itun              TRUE            FALSE   0.000000
## 130             <NA>              TRUE            FALSE   0.000000
## 132             <NA>              TRUE            FALSE   0.000000
## 137             <NA>              TRUE             TRUE   0.000000
## 138             <NA>              TRUE            FALSE   0.000000
## 139             <NA>              TRUE            FALSE   0.000000
## 140             <NA>              TRUE            FALSE   0.000000
## 141             <NA>              TRUE            FALSE   0.000000
## 143             <NA>              TRUE            FALSE   0.000000
## 144             <NA>              TRUE            FALSE   0.000000
## 146             <NA>              TRUE            FALSE   0.000000
## 147             <NA>              TRUE            FALSE   0.000000
## 148             <NA>              TRUE            FALSE   0.000000
## 149             <NA>              TRUE            FALSE   0.000000
## 153             <NA>              TRUE             TRUE   0.000000
## 155             <NA>              TRUE            FALSE   0.000000
## 157             <NA>              TRUE             TRUE   0.000000
## 158             <NA>              TRUE            FALSE   0.000000
## 159             <NA>              TRUE             TRUE   0.000000
## 160             <NA>              TRUE            FALSE   0.000000
## 161             <NA>              TRUE             TRUE   0.000000
## 162             <NA>              TRUE            FALSE   0.000000
## 163             <NA>              TRUE             TRUE   0.000000
## 165             <NA>              TRUE             TRUE   0.000000
## 166             <NA>              TRUE            FALSE   0.000000
## 168             <NA>              TRUE            FALSE   0.000000
## 170             <NA>              TRUE            FALSE   0.000000
## 173             <NA>              TRUE            FALSE   0.000000
## 176             <NA>              TRUE            FALSE   0.000000
## 177             <NA>              TRUE             TRUE   0.000000
## 178             <NA>              TRUE            FALSE   0.000000
## 180             <NA>              TRUE             TRUE   0.000000
## 182             <NA>              TRUE             TRUE   0.000000
## 183             <NA>              TRUE            FALSE   0.000000
## 184             <NA>              TRUE            FALSE   0.000000
## 185             <NA>              TRUE            FALSE   0.000000
## 188             <NA>              TRUE            FALSE   0.000000
## 189             <NA>              TRUE             TRUE   0.000000
## 192             <NA>              TRUE            FALSE   0.000000
## 194             <NA>              TRUE            FALSE   0.000000
## 195             <NA>              TRUE            FALSE   0.000000
## 199             <NA>              TRUE             TRUE   0.000000
## 201             <NA>              TRUE             TRUE   0.000000
## 203             <NA>              TRUE             TRUE   0.000000
## 204             <NA>              TRUE            FALSE   0.000000
## 207             <NA>              TRUE            FALSE   0.000000
## 208             <NA>              TRUE            FALSE   0.000000
## 211             <NA>              TRUE             TRUE   0.000000
## 215             <NA>              TRUE             TRUE   0.000000
## 217            emiss              TRUE            FALSE   0.000000
## 218             <NA>              TRUE            FALSE   0.000000
## 219             <NA>              TRUE             TRUE   0.000000
## 220             <NA>              TRUE             TRUE   0.000000
## 223             <NA>              TRUE            FALSE   0.000000
## 224             <NA>              TRUE             TRUE   0.000000
## 225             <NA>              TRUE            FALSE   0.000000
## 226             <NA>              TRUE             TRUE   0.000000
## 227             <NA>              TRUE            FALSE   0.000000
## 228             <NA>              TRUE             TRUE   0.000000
## 229             <NA>              TRUE             TRUE   0.000000
## 232             <NA>              TRUE            FALSE   0.000000
## 234             <NA>              TRUE            FALSE   0.000000
## 236             <NA>              TRUE            FALSE   0.000000
## 237             <NA>              TRUE            FALSE   0.000000
## 238             <NA>              TRUE            FALSE   0.000000
## 239             <NA>              TRUE            FALSE   0.000000
## 240             <NA>              TRUE             TRUE   0.000000
## 242             <NA>              TRUE            FALSE   0.000000
## 243             <NA>              TRUE            FALSE   0.000000
## 244             <NA>              TRUE            FALSE   0.000000
## 245             <NA>              TRUE            FALSE   0.000000
## 246             <NA>              TRUE            FALSE   0.000000
## 247             <NA>              TRUE            FALSE   0.000000
## 248             <NA>              TRUE            FALSE   0.000000
## 250             <NA>              TRUE            FALSE   0.000000
## 252             <NA>              TRUE            FALSE   0.000000
## 253             <NA>              TRUE             TRUE   0.000000
## 254             <NA>              TRUE            FALSE   0.000000
## 255             <NA>              TRUE            FALSE   0.000000
## 257             <NA>              TRUE            FALSE   0.000000
## 259             <NA>              TRUE            FALSE   0.000000
## 261             <NA>              TRUE            FALSE   0.000000
## 262             <NA>              TRUE             TRUE   0.000000
## 263             <NA>              TRUE            FALSE   0.000000
## 264             <NA>              TRUE            FALSE   0.000000
## 265             <NA>              TRUE            FALSE   0.000000
## 266             <NA>              TRUE            FALSE   0.000000
## 268             <NA>              TRUE             TRUE   0.000000
## 269             <NA>              TRUE            FALSE   0.000000
## 270             <NA>              TRUE             TRUE   0.000000
## 271             <NA>              TRUE            FALSE   0.000000
## 274             <NA>              TRUE            FALSE   0.000000
## 276             <NA>              TRUE            FALSE   0.000000
## 278             <NA>              TRUE            FALSE   0.000000
## 279             <NA>              TRUE            FALSE   0.000000
## 280             <NA>              TRUE            FALSE   0.000000
## 281             <NA>              TRUE            FALSE   0.000000
## 282             <NA>              TRUE            FALSE   0.000000
## 284             <NA>              TRUE            FALSE   0.000000
## 287             <NA>              TRUE            FALSE   0.000000
## 289             <NA>              TRUE            FALSE   0.000000
## 292             <NA>              TRUE             TRUE   0.000000
## 294             <NA>              TRUE            FALSE   0.000000
## 295             <NA>              TRUE             TRUE   0.000000
## 299             <NA>              TRUE            FALSE   0.000000
## 300             <NA>              TRUE             TRUE   0.000000
## 302             <NA>              TRUE            FALSE   0.000000
## 303             <NA>              TRUE            FALSE   0.000000
## 306             <NA>              TRUE            FALSE   0.000000
## 307             <NA>              TRUE            FALSE   0.000000
## 308             <NA>              TRUE            FALSE   0.000000
## 310             <NA>              TRUE             TRUE   0.000000
## 311             <NA>              TRUE            FALSE   0.000000
## 312             <NA>              TRUE            FALSE   0.000000
## 1               <NA>              TRUE            FALSE         NA
## 2               <NA>             FALSE            FALSE         NA
## 3               <NA>             FALSE            FALSE         NA
## 4               <NA>             FALSE            FALSE         NA
## 5               <NA>             FALSE            FALSE         NA
## 9               <NA>             FALSE            FALSE         NA
## 13              <NA>             FALSE            FALSE         NA
## 16              <NA>             FALSE            FALSE         NA
## 17              <NA>                NA            FALSE         NA
## 18              <NA>             FALSE            FALSE         NA
## 24              <NA>             FALSE            FALSE         NA
## 29              <NA>             FALSE            FALSE         NA
## 30              <NA>             FALSE            FALSE         NA
## 31              <NA>             FALSE            FALSE         NA
## 50              <NA>             FALSE            FALSE         NA
## 53              <NA>             FALSE            FALSE         NA
## 54              <NA>             FALSE            FALSE         NA
## 61              <NA>             FALSE            FALSE         NA
## 62              <NA>             FALSE            FALSE         NA
## 68              <NA>             FALSE            FALSE         NA
## 75              <NA>             FALSE            FALSE         NA
## 77              <NA>             FALSE            FALSE         NA
## 82              <NA>             FALSE            FALSE         NA
## 88              <NA>             FALSE            FALSE         NA
## 89              <NA>             FALSE            FALSE         NA
## 90              <NA>             FALSE            FALSE         NA
## 92              <NA>             FALSE            FALSE         NA
## 95              <NA>             FALSE            FALSE         NA
## 99              <NA>             FALSE            FALSE         NA
## 100             <NA>             FALSE            FALSE         NA
## 103             <NA>             FALSE            FALSE         NA
## 108             <NA>             FALSE            FALSE         NA
## 112             <NA>             FALSE            FALSE         NA
## 114             <NA>             FALSE            FALSE         NA
## 119             <NA>             FALSE            FALSE         NA
## 126             <NA>             FALSE            FALSE         NA
## 127             <NA>             FALSE            FALSE         NA
## 131             <NA>             FALSE            FALSE         NA
## 133             <NA>             FALSE            FALSE         NA
## 134             <NA>             FALSE            FALSE         NA
## 135             <NA>             FALSE            FALSE         NA
## 136             <NA>             FALSE            FALSE         NA
## 142             <NA>             FALSE            FALSE         NA
## 145             <NA>             FALSE            FALSE         NA
## 150             <NA>             FALSE            FALSE         NA
## 151             <NA>             FALSE            FALSE         NA
## 152             <NA>             FALSE            FALSE         NA
## 154             <NA>             FALSE            FALSE         NA
## 156             <NA>             FALSE            FALSE         NA
## 164             <NA>             FALSE            FALSE         NA
## 167             <NA>             FALSE            FALSE         NA
## 169             <NA>             FALSE            FALSE         NA
## 171             <NA>             FALSE            FALSE         NA
## 172             <NA>             FALSE            FALSE         NA
## 174             <NA>             FALSE            FALSE         NA
## 175             <NA>             FALSE            FALSE         NA
## 179             <NA>             FALSE            FALSE         NA
## 181             <NA>             FALSE            FALSE         NA
## 186             <NA>             FALSE            FALSE         NA
## 187             <NA>             FALSE            FALSE         NA
## 190             <NA>             FALSE            FALSE         NA
## 191             <NA>             FALSE            FALSE         NA
## 193             <NA>             FALSE            FALSE         NA
## 196             <NA>             FALSE            FALSE         NA
## 197             <NA>             FALSE            FALSE         NA
## 200             <NA>             FALSE            FALSE         NA
## 202             <NA>             FALSE            FALSE         NA
## 205             <NA>             FALSE            FALSE         NA
## 206             <NA>             FALSE            FALSE         NA
## 209             <NA>             FALSE            FALSE         NA
## 210             <NA>             FALSE            FALSE         NA
## 212             <NA>             FALSE            FALSE         NA
## 213             <NA>             FALSE            FALSE         NA
## 214             <NA>             FALSE            FALSE         NA
## 216             <NA>             FALSE            FALSE         NA
## 221             <NA>             FALSE            FALSE         NA
## 222             <NA>             FALSE            FALSE         NA
## 230             <NA>             FALSE            FALSE         NA
## 231             <NA>             FALSE            FALSE         NA
## 233             <NA>             FALSE            FALSE         NA
## 235             <NA>             FALSE            FALSE         NA
## 241             <NA>             FALSE            FALSE         NA
## 251             <NA>             FALSE            FALSE         NA
## 256             <NA>             FALSE            FALSE         NA
## 258             <NA>             FALSE            FALSE         NA
## 260             <NA>             FALSE            FALSE         NA
## 267             <NA>             FALSE            FALSE         NA
## 272             <NA>             FALSE            FALSE         NA
## 273             <NA>             FALSE            FALSE         NA
## 275             <NA>             FALSE            FALSE         NA
## 277             <NA>                NA            FALSE         NA
## 283             <NA>             FALSE            FALSE         NA
## 285             <NA>             FALSE            FALSE         NA
## 286             <NA>             FALSE            FALSE         NA
## 288             <NA>             FALSE            FALSE         NA
## 290             <NA>             FALSE            FALSE         NA
## 291             <NA>             FALSE            FALSE         NA
## 293             <NA>             FALSE            FALSE         NA
## 296             <NA>             FALSE            FALSE         NA
## 297             <NA>             FALSE            FALSE         NA
## 298             <NA>             FALSE            FALSE         NA
## 301             <NA>             FALSE            FALSE         NA
## 305             <NA>             FALSE            FALSE         NA
## 309             <NA>             FALSE            FALSE         NA
```

```r
# Used again in predict.data.new chunk
glb_analytics_diag_plots <- function(obs_df) {
    for (var in subset(glb_feats_df, importance > 0)$id) {
        plot_df <- melt(obs_df, id.vars=var, 
                        measure.vars=c(glb_rsp_var, glb_rsp_var_out))
#         if (var == "<feat_name>") print(myplot_scatter(plot_df, var, "value", 
#                                              facet_colcol_name="variable") + 
#                       geom_vline(xintercept=<divider_val>, linetype="dotted")) else     
            print(myplot_scatter(plot_df, var, "value", colorcol_name="variable",
                                 facet_colcol_name="variable", jitter=TRUE) + 
                      guides(color=FALSE))
    }
    
    if (glb_is_regression) {
#         plot_vars_df <- subset(glb_feats_df, importance > 
#                         glb_feats_df[glb_feats_df$id == ".rnorm", "importance"])
        plot_vars_df <- orderBy(~ -importance, glb_feats_df)
        if (nrow(plot_vars_df) == 0)
            warning("No important features in glb_fin_mdl") else
            print(myplot_prediction_regression(df=obs_df, 
                        feat_x=ifelse(nrow(plot_vars_df) > 1, plot_vars_df$id[2],
                                      ".rownames"), 
                                               feat_y=plot_vars_df$id[1],
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        id_vars=glb_id_vars)
    #               + facet_wrap(reformulate(plot_vars_df$id[2])) # if [1 or 2] is a factor                                                         
    #               + geom_point(aes_string(color="<col_name>.fctr")) #  to color the plot
                  )
    }    
    
    if (glb_is_classification) {
        if (nrow(plot_vars_df <- subset(glb_feats_df, importance > 0)) == 0)
            warning("No features in selected model are statistically important")
        else print(myplot_prediction_classification(df=obs_df, 
                feat_x=ifelse(nrow(plot_vars_df) > 1, plot_vars_df$id[2], 
                              ".rownames"),
                                               feat_y=plot_vars_df$id[1],
                     rsp_var=glb_rsp_var, 
                     rsp_var_out=glb_rsp_var_out, 
                     id_vars=glb_id_vars)
#               + geom_hline(yintercept=<divider_val>, linetype = "dotted")
                )
    }    
}
glb_analytics_diag_plots(obs_df=glb_trnent_df)
```

![](Apple_Tweets_files/figure-html/fit.data.training.all_1-1.png) ![](Apple_Tweets_files/figure-html/fit.data.training.all_1-2.png) ![](Apple_Tweets_files/figure-html/fit.data.training.all_1-3.png) ![](Apple_Tweets_files/figure-html/fit.data.training.all_1-4.png) ![](Apple_Tweets_files/figure-html/fit.data.training.all_1-5.png) ![](Apple_Tweets_files/figure-html/fit.data.training.all_1-6.png) ![](Apple_Tweets_files/figure-html/fit.data.training.all_1-7.png) ![](Apple_Tweets_files/figure-html/fit.data.training.all_1-8.png) 

```
##                                                                                                                                             Tweet
## 1                                           I have to say, Apple has by far the best customer care service I have ever received! @Apple @AppStore
## 4      Because @Apple has no idea how real people work, I have to erase my iphone & start over just for re-installing itunes after a new OS. #fun
## 5    Bc apple no longer impresses like it used to "@lmkunert: @Apple stock still on the decline after yesterday's iPhone Event. Not a good sign."
## 12    Okay so @apple apparently is making the  new iPhone which the EXACT SAME THING as the old iPhone but now it has colour, I'm going to murder
## 16                     why has it been 24 hours since i started restoring from icloud and my apps still arent downloaded??? @apple #darnyouiphone
## 17                                                                       .@tmcconnon You're the @Apple of my eye. Meaning you're a piece of crap.
## 18                                              Just spent a painful 20 minutes on the phone to iTunes support. Life's too short for this @apple.
## 38                                                                  @Apple I hate when they put out a new iPhone because then my iPhone feels old
## 200                                                                WHY CANT I freakING SEE PICTURES ON MY TL IM ANNOYED freak YOU @TWITTER @APPLE
## 853                                               Lmao i hate u RT @JZFan: When are y'all gonna come out with the iJordans? @Apple @MichaelJordan
## 1152                                                                            tbh i always love black or gold jesus fjaskdghdka hate you @apple
##       Avg
## 1     2.0
## 4    -1.0
## 5    -1.0
## 12   -1.0
## 16   -1.0
## 17   -1.0
## 18   -1.0
## 38   -1.0
## 200  -2.0
## 853  -0.2
## 1152 -0.8
##                                                                                                                                        Tweet.fctr
## 1                                           I have to say, Apple has by far the best customer care service I have ever received! @Apple @AppStore
## 4      Because @Apple has no idea how real people work, I have to erase my iphone & start over just for re-installing itunes after a new OS. #fun
## 5    Bc apple no longer impresses like it used to "@lmkunert: @Apple stock still on the decline after yesterday's iPhone Event. Not a good sign."
## 12    Okay so @apple apparently is making the  new iPhone which the EXACT SAME THING as the old iPhone but now it has colour, I'm going to murder
## 16                     why has it been 24 hours since i started restoring from icloud and my apps still arent downloaded??? @apple #darnyouiphone
## 17                                                                       .@tmcconnon You're the @Apple of my eye. Meaning you're a piece of crap.
## 18                                              Just spent a painful 20 minutes on the phone to iTunes support. Life's too short for this @apple.
## 38                                                                  @Apple I hate when they put out a new iPhone because then my iPhone feels old
## 200                                                                WHY CANT I freakING SEE PICTURES ON MY TL IM ANNOYED freak YOU @TWITTER @APPLE
## 853                                               Lmao i hate u RT @JZFan: When are y'all gonna come out with the iJordans? @Apple @MichaelJordan
## 1152                                                                            tbh i always love black or gold jesus fjaskdghdka hate you @apple
##          .rnorm Negative.fctr X244tsuyoponzu X7evenstarz actual add
## 1     0.7262803             N              0           0      0   0
## 4     0.4895881             Y              0           0      0   0
## 5     1.3209853             Y              0           0      0   0
## 12    0.3215156             Y              0           0      0   0
## 16    0.3227588             Y              0           0      0   0
## 17   -0.2609005             Y              0           0      0   0
## 18   -0.3096915             Y              0           0      0   0
## 38   -0.2439169             Y              0           0      0   0
## 200   0.7720553             Y              0           0      0   0
## 853   1.0031694             N              0           0      0   0
## 1152 -0.9875117             N              0           0      0   0
##      alreadi alway amaz amazon android announc anyon app appl appstor
## 1          0     0    0      0       0       0     0   0    0       1
## 4          0     0    0      0       0       0     0   0    0       0
## 5          0     0    0      0       0       0     0   0    0       0
## 12         0     0    0      0       0       0     0   0    0       0
## 16         0     0    0      0       0       0     0   1    0       0
## 17         0     0    0      0       0       0     0   0    0       0
## 18         0     0    0      0       0       0     0   0    0       0
## 38         0     0    0      0       0       0     0   0    0       0
## 200        0     0    0      0       0       0     0   0    0       0
## 853        0     0    0      0       0       0     0   0    0       0
## 1152       0     1    0      0       0       0     0   0    0       0
##      arent ask avail away awesom back batteri best better big bit black
## 1        0   0     0    0      0    0       0    1      0   0   0     0
## 4        0   0     0    0      0    0       0    0      0   0   0     0
## 5        0   0     0    0      0    0       0    0      0   0   0     0
## 12       0   0     0    0      0    0       0    0      0   0   0     0
## 16       1   0     0    0      0    0       0    0      0   0   0     0
## 17       0   0     0    0      0    0       0    0      0   0   0     0
## 18       0   0     0    0      0    0       0    0      0   0   0     0
## 38       0   0     0    0      0    0       0    0      0   0   0     0
## 200      0   0     0    0      0    0       0    0      0   0   0     0
## 853      0   0     0    0      0    0       0    0      0   0   0     0
## 1152     0   0     0    0      0    0       0    0      0   0   0     1
##      blackberri break. bring burberri busi buy call can cant carbon card
## 1             0      0     0        0    0   0    0   0    0      0    0
## 4             0      0     0        0    0   0    0   0    0      0    0
## 5             0      0     0        0    0   0    0   0    0      0    0
## 12            0      0     0        0    0   0    0   0    0      0    0
## 16            0      0     0        0    0   0    0   0    0      0    0
## 17            0      0     0        0    0   0    0   0    0      0    0
## 18            0      0     0        0    0   0    0   0    0      0    0
## 38            0      0     0        0    0   0    0   0    0      0    0
## 200           0      0     0        0    0   0    0   0    1      0    0
## 853           0      0     0        0    0   0    0   0    0      0    0
## 1152          0      0     0        0    0   0    0   0    0      0    0
##      care case cdp chang charg charger cheap china color colour come
## 1       1    0   0     0     0       0     0     0     0      0    0
## 4       0    0   0     0     0       0     0     0     0      0    0
## 5       0    0   0     0     0       0     0     0     0      0    0
## 12      0    0   0     0     0       0     0     0     0      1    0
## 16      0    0   0     0     0       0     0     0     0      0    0
## 17      0    0   0     0     0       0     0     0     0      0    0
## 18      0    0   0     0     0       0     0     0     0      0    0
## 38      0    0   0     0     0       0     0     0     0      0    0
## 200     0    0   0     0     0       0     0     0     0      0    0
## 853     0    0   0     0     0       0     0     0     0      0    1
## 1152    0    0   0     0     0       0     0     0     0      0    0
##      compani condescens condom copi crack creat custom darn data date day
## 1          0          0      0    0     0     0      1    0    0    0   0
## 4          0          0      0    0     0     0      0    0    0    0   0
## 5          0          0      0    0     0     0      0    0    0    0   0
## 12         0          0      0    0     0     0      0    0    0    0   0
## 16         0          0      0    0     0     0      0    0    0    0   0
## 17         0          0      0    0     0     0      0    0    0    0   0
## 18         0          0      0    0     0     0      0    0    0    0   0
## 38         0          0      0    0     0     0      0    0    0    0   0
## 200        0          0      0    0     0     0      0    0    0    0   0
## 853        0          0      0    0     0     0      0    0    0    0   0
## 1152       0          0      0    0     0     0      0    0    0    0   0
##      dear design develop devic didnt die differ disappoint discontinu
## 1       0      0       0     0     0   0      0          0          0
## 4       0      0       0     0     0   0      0          0          0
## 5       0      0       0     0     0   0      0          0          0
## 12      0      0       0     0     0   0      0          0          0
## 16      0      0       0     0     0   0      0          0          0
## 17      0      0       0     0     0   0      0          0          0
## 18      0      0       0     0     0   0      0          0          0
## 38      0      0       0     0     0   0      0          0          0
## 200     0      0       0     0     0   0      0          0          0
## 853     0      0       0     0     0   0      0          0          0
## 1152    0      0       0     0     0   0      0          0          0
##      divulg doesnt done dont download drop email emiss emoji even event
## 1         0      0    0    0        0    0     0     0     0    0     0
## 4         0      0    0    0        0    0     0     0     0    0     0
## 5         0      0    0    0        0    0     0     0     0    0     1
## 12        0      0    0    0        0    0     0     0     0    0     0
## 16        0      0    0    0        1    0     0     0     0    0     0
## 17        0      0    0    0        0    0     0     0     0    0     0
## 18        0      0    0    0        0    0     0     0     0    0     0
## 38        0      0    0    0        0    0     0     0     0    0     0
## 200       0      0    0    0        0    0     0     0     0    0     0
## 853       0      0    0    0        0    0     0     0     0    0     0
## 1152      0      0    0    0        0    0     0     0     0    0     0
##      ever everi everyth facebook fail featur feel femal figur final finger
## 1       1     0       0        0    0      0    0     0     0     0      0
## 4       0     0       0        0    0      0    0     0     0     0      0
## 5       0     0       0        0    0      0    0     0     0     0      0
## 12      0     0       0        0    0      0    0     0     0     0      0
## 16      0     0       0        0    0      0    0     0     0     0      0
## 17      0     0       0        0    0      0    0     0     0     0      0
## 18      0     0       0        0    0      0    0     0     0     0      0
## 38      0     0       0        0    0      0    1     0     0     0      0
## 200     0     0       0        0    0      0    0     0     0     0      0
## 853     0     0       0        0    0      0    0     0     0     0      0
## 1152    0     0       0        0    0      0    0     0     0     0      0
##      fingerprint fire first fix follow freak free fun generat genius get
## 1              0    0     0   0      0     0    0   0       0      0   0
## 4              0    0     0   0      0     0    0   1       0      0   0
## 5              0    0     0   0      0     0    0   0       0      0   0
## 12             0    0     0   0      0     0    0   0       0      0   0
## 16             0    0     0   0      0     0    0   0       0      0   0
## 17             0    0     0   0      0     0    0   0       0      0   0
## 18             0    0     0   0      0     0    0   0       0      0   0
## 38             0    0     0   0      0     0    0   0       0      0   0
## 200            0    0     0   0      0     2    0   0       0      0   0
## 853            0    0     0   0      0     0    0   0       0      0   0
## 1152           0    0     0   0      0     0    0   0       0      0   0
##      give gold gonna good googl got great guess guy happen happi hate help
## 1       0    0     0    0     0   0     0     0   0      0     0    0    0
## 4       0    0     0    0     0   0     0     0   0      0     0    0    0
## 5       0    0     0    1     0   0     0     0   0      0     0    0    0
## 12      0    0     0    0     0   0     0     0   0      0     0    0    0
## 16      0    0     0    0     0   0     0     0   0      0     0    0    0
## 17      0    0     0    0     0   0     0     0   0      0     0    0    0
## 18      0    0     0    0     0   0     0     0   0      0     0    0    0
## 38      0    0     0    0     0   0     0     0   0      0     0    1    0
## 200     0    0     0    0     0   0     0     0   0      0     0    0    0
## 853     0    0     1    0     0   0     0     0   0      0     0    1    0
## 1152    0    1     0    0     0   0     0     0   0      0     0    1    0
##      hey hope hour httpbitly18xc8dk ibrooklynb idea ill imessag impress
## 1      0    0    0                0          0    0   0       0       0
## 4      0    0    0                0          0    1   0       0       0
## 5      0    0    0                0          0    0   0       0       1
## 12     0    0    0                0          0    0   0       0       0
## 16     0    0    1                0          0    0   0       0       0
## 17     0    0    0                0          0    0   0       0       0
## 18     0    0    0                0          0    0   0       0       0
## 38     0    0    0                0          0    0   0       0       0
## 200    0    0    0                0          0    0   0       0       0
## 853    0    0    0                0          0    0   0       0       0
## 1152   0    0    0                0          0    0   0       0       0
##      improv innov instead internet ios7 ipad iphon iphone4 iphone5
## 1         0     0       0        0    0    0     0       0       0
## 4         0     0       0        0    0    0     1       0       0
## 5         0     0       0        0    0    0     1       0       0
## 12        0     0       0        0    0    0     2       0       0
## 16        0     0       0        0    0    0     0       0       0
## 17        0     0       0        0    0    0     0       0       0
## 18        0     0       0        0    0    0     0       0       0
## 38        0     0       0        0    0    0     2       0       0
## 200       0     0       0        0    0    0     0       0       0
## 853       0     0       0        0    0    0     0       0       0
## 1152      0     0       0        0    0    0     0       0       0
##      iphone5c iphoto ipod ipodplayerpromo isnt itun ive job just keynot
## 1           0      0    0               0    0    0   0   0    0      0
## 4           0      0    0               0    0    1   0   0    1      0
## 5           0      0    0               0    0    0   0   0    0      0
## 12          0      0    0               0    0    0   0   0    0      0
## 16          0      0    0               0    0    0   0   0    0      0
## 17          0      0    0               0    0    0   0   0    0      0
## 18          0      0    0               0    0    1   0   0    1      0
## 38          0      0    0               0    0    0   0   0    0      0
## 200         0      0    0               0    0    0   0   0    0      0
## 853         0      0    0               0    0    0   0   0    0      0
## 1152        0      0    0               0    0    0   0   0    0      0
##      know last launch let life like line lmao lock lol look los lost love
## 1       0    0      0   0    0    0    0    0    0   0    0   0    0    0
## 4       0    0      0   0    0    0    0    0    0   0    0   0    0    0
## 5       0    0      0   0    0    1    0    0    0   0    0   0    0    0
## 12      0    0      0   0    0    0    0    0    0   0    0   0    0    0
## 16      0    0      0   0    0    0    0    0    0   0    0   0    0    0
## 17      0    0      0   0    0    0    0    0    0   0    0   0    0    0
## 18      0    0      0   0    1    0    0    0    0   0    0   0    0    0
## 38      0    0      0   0    0    0    0    0    0   0    0   0    0    0
## 200     0    0      0   0    0    0    0    0    0   0    0   0    0    0
## 853     0    0      0   0    0    0    0    1    0   0    0   0    0    0
## 1152    0    0      0   0    0    0    0    0    0   0    0   0    0    1
##      mac macbook made make man mani market mayb mean microsoft mishiza
## 1      0       0    0    0   0    0      0    0    0         0       0
## 4      0       0    0    0   0    0      0    0    0         0       0
## 5      0       0    0    0   0    0      0    0    0         0       0
## 12     0       0    0    1   0    0      0    0    0         0       0
## 16     0       0    0    0   0    0      0    0    0         0       0
## 17     0       0    0    0   0    0      0    0    1         0       0
## 18     0       0    0    0   0    0      0    0    0         0       0
## 38     0       0    0    0   0    0      0    0    0         0       0
## 200    0       0    0    0   0    0      0    0    0         0       0
## 853    0       0    0    0   0    0      0    0    0         0       0
## 1152   0       0    0    0   0    0      0    0    0         0       0
##      miss mobil money motorola move much music natz0711 need never new
## 1       0     0     0        0    0    0     0        0    0     0   0
## 4       0     0     0        0    0    0     0        0    0     0   1
## 5       0     0     0        0    0    0     0        0    0     0   0
## 12      0     0     0        0    0    0     0        0    0     0   1
## 16      0     0     0        0    0    0     0        0    0     0   0
## 17      0     0     0        0    0    0     0        0    0     0   0
## 18      0     0     0        0    0    0     0        0    0     0   0
## 38      0     0     0        0    0    0     0        0    0     0   1
## 200     0     0     0        0    0    0     0        0    0     0   0
## 853     0     0     0        0    0    0     0        0    0     0   0
## 1152    0     0     0        0    0    0     0        0    0     0   0
##      news next. nfc nokia noth now nsa nuevo offer old one page para peopl
## 1       0     0   0     0    0   0   0     0     0   0   0    0    0     0
## 4       0     0   0     0    0   0   0     0     0   0   0    0    0     1
## 5       0     0   0     0    0   0   0     0     0   0   0    0    0     0
## 12      0     0   0     0    0   1   0     0     0   1   0    0    0     0
## 16      0     0   0     0    0   0   0     0     0   0   0    0    0     0
## 17      0     0   0     0    0   0   0     0     0   0   0    0    0     0
## 18      0     0   0     0    0   0   0     0     0   0   0    0    0     0
## 38      0     0   0     0    0   0   0     0     0   1   0    0    0     0
## 200     0     0   0     0    0   0   0     0     0   0   0    0    0     0
## 853     0     0   0     0    0   0   0     0     0   0   0    0    0     0
## 1152    0     0   0     0    0   0   0     0     0   0   0    0    0     0
##      perfect person phone photog photographi pictur plastic play pleas ppl
## 1          0      0     0      0           0      0       0    0     0   0
## 4          0      0     0      0           0      0       0    0     0   0
## 5          0      0     0      0           0      0       0    0     0   0
## 12         0      0     0      0           0      0       0    0     0   0
## 16         0      0     0      0           0      0       0    0     0   0
## 17         0      0     0      0           0      0       0    0     0   0
## 18         0      0     1      0           0      0       0    0     0   0
## 38         0      0     0      0           0      0       0    0     0   0
## 200        0      0     0      0           0      1       0    0     0   0
## 853        0      0     0      0           0      0       0    0     0   0
## 1152       0      0     0      0           0      0       0    0     0   0
##      preorder price print pro problem product promo promoipodplayerpromo
## 1           0     0     0   0       0       0     0                    0
## 4           0     0     0   0       0       0     0                    0
## 5           0     0     0   0       0       0     0                    0
## 12          0     0     0   0       0       0     0                    0
## 16          0     0     0   0       0       0     0                    0
## 17          0     0     0   0       0       0     0                    0
## 18          0     0     0   0       0       0     0                    0
## 38          0     0     0   0       0       0     0                    0
## 200         0     0     0   0       0       0     0                    0
## 853         0     0     0   0       0       0     0                    0
## 1152        0     0     0   0       0       0     0                    0
##      put que quiet read realli recommend refus releas right said samsung
## 1      0   0     0    0      0         0     0      0     0    0       0
## 4      0   0     0    0      0         0     0      0     0    0       0
## 5      0   0     0    0      0         0     0      0     0    0       0
## 12     0   0     0    0      0         0     0      0     0    0       0
## 16     0   0     0    0      0         0     0      0     0    0       0
## 17     0   0     0    0      0         0     0      0     0    0       0
## 18     0   0     0    0      0         0     0      0     0    0       0
## 38     1   0     0    0      0         0     0      0     0    0       0
## 200    0   0     0    0      0         0     0      0     0    0       0
## 853    0   0     0    0      0         0     0      0     0    0       0
## 1152   0   0     0    0      0         0     0      0     0    0       0
##      samsungsa say scanner screen secur see seem sell send servic shame
## 1            0   1       0      0     0   0    0    0    0      1     0
## 4            0   0       0      0     0   0    0    0    0      0     0
## 5            0   0       0      0     0   0    0    0    0      0     0
## 12           0   0       0      0     0   0    0    0    0      0     0
## 16           0   0       0      0     0   0    0    0    0      0     0
## 17           0   0       0      0     0   0    0    0    0      0     0
## 18           0   0       0      0     0   0    0    0    0      0     0
## 38           0   0       0      0     0   0    0    0    0      0     0
## 200          0   0       0      0     0   1    0    0    0      0     0
## 853          0   0       0      0     0   0    0    0    0      0     0
## 1152         0   0       0      0     0   0    0    0    0      0     0
##      share short show simpl sinc siri smart smartphon someth soon stand
## 1        0     0    0     0    0    0     0         0      0    0     0
## 4        0     0    0     0    0    0     0         0      0    0     0
## 5        0     0    0     0    0    0     0         0      0    0     0
## 12       0     0    0     0    0    0     0         0      0    0     0
## 16       0     0    0     0    1    0     0         0      0    0     0
## 17       0     0    0     0    0    0     0         0      0    0     0
## 18       0     1    0     0    0    0     0         0      0    0     0
## 38       0     0    0     0    0    0     0         0      0    0     0
## 200      0     0    0     0    0    0     0         0      0    0     0
## 853      0     0    0     0    0    0     0         0      0    0     0
## 1152     0     0    0     0    0    0     0         0      0    0     0
##      start steve still stop store stuff stupid suck support sure switch
## 1        0     0     0    0     0     0      0    0       0    0      0
## 4        1     0     0    0     0     0      0    0       0    0      0
## 5        0     0     1    0     0     0      0    0       0    0      0
## 12       0     0     0    0     0     0      0    0       0    0      0
## 16       1     0     1    0     0     0      0    0       0    0      0
## 17       0     0     0    0     0     0      0    0       0    0      0
## 18       0     0     0    0     0     0      0    0       1    0      0
## 38       0     0     0    0     0     0      0    0       0    0      0
## 200      0     0     0    0     0     0      0    0       0    0      0
## 853      0     0     0    0     0     0      0    0       0    0      0
## 1152     0     0     0    0     0     0      0    0       0    0      0
##      take talk team tech technolog tell text thank that theyr thing think
## 1       0    0    0    0         0    0    0     0    0     0     0     0
## 4       0    0    0    0         0    0    0     0    0     0     0     0
## 5       0    0    0    0         0    0    0     0    0     0     0     0
## 12      0    0    0    0         0    0    0     0    0     0     1     0
## 16      0    0    0    0         0    0    0     0    0     0     0     0
## 17      0    0    0    0         0    0    0     0    0     0     0     0
## 18      0    0    0    0         0    0    0     0    0     0     0     0
## 38      0    0    0    0         0    0    0     0    0     0     0     0
## 200     0    0    0    0         0    0    0     0    0     0     0     0
## 853     0    0    0    0         0    0    0     0    0     0     0     0
## 1152    0    0    0    0         0    0    0     0    0     0     0     0
##      tho thought time today togeth touch touchid tri true turn twitter two
## 1      0       0    0     0      0     0       0   0    0    0       0   0
## 4      0       0    0     0      0     0       0   0    0    0       0   0
## 5      0       0    0     0      0     0       0   0    0    0       0   0
## 12     0       0    0     0      0     0       0   0    0    0       0   0
## 16     0       0    0     0      0     0       0   0    0    0       0   0
## 17     0       0    0     0      0     0       0   0    0    0       0   0
## 18     0       0    0     0      0     0       0   0    0    0       0   0
## 38     0       0    0     0      0     0       0   0    0    0       0   0
## 200    0       0    0     0      0     0       0   0    0    0       1   0
## 853    0       0    0     0      0     0       0   0    0    0       0   0
## 1152   0       0    0     0      0     0       0   0    0    0       0   0
##      updat upgrad use user via video wait want watch way week well what
## 1        0      0   0    0   0     0    0    0     0   0    0    0    0
## 4        0      0   0    0   0     0    0    0     0   0    0    0    0
## 5        0      0   1    0   0     0    0    0     0   0    0    0    0
## 12       0      0   0    0   0     0    0    0     0   0    0    0    0
## 16       0      0   0    0   0     0    0    0     0   0    0    0    0
## 17       0      0   0    0   0     0    0    0     0   0    0    0    0
## 18       0      0   0    0   0     0    0    0     0   0    0    0    0
## 38       0      0   0    0   0     0    0    0     0   0    0    0    0
## 200      0      0   0    0   0     0    0    0     0   0    0    0    0
## 853      0      0   0    0   0     0    0    0     0   0    0    0    0
## 1152     0      0   0    0   0     0    0    0     0   0    0    0    0
##      white will windowsphon wish without wonder wont work world worst wow
## 1        0    0           0    0       0      0    0    0     0     0   0
## 4        0    0           0    0       0      0    0    1     0     0   0
## 5        0    0           0    0       0      0    0    0     0     0   0
## 12       0    0           0    0       0      0    0    0     0     0   0
## 16       0    0           0    0       0      0    0    0     0     0   0
## 17       0    0           0    0       0      0    0    0     0     0   0
## 18       0    0           0    0       0      0    0    0     0     0   0
## 38       0    0           0    0       0      0    0    0     0     0   0
## 200      0    0           0    0       0      0    0    0     0     0   0
## 853      0    0           0    0       0      0    0    0     0     0   0
## 1152     0    0           0    0       0      0    0    0     0     0   0
##      wtf yall year yes yet yooo your
## 1      0    0    0   0   0    0    0
## 4      0    0    0   0   0    0    0
## 5      0    0    0   0   0    0    0
## 12     0    0    0   0   0    0    0
## 16     0    0    0   0   0    0    0
## 17     0    0    0   0   0    0    2
## 18     0    0    0   0   0    0    0
## 38     0    0    0   0   0    0    0
## 200    0    0    0   0   0    0    0
## 853    0    1    0   0   0    0    0
## 1152   0    0    0   0   0    0    0
##      Negative.fctr.predict.Final.rpart.prob
## 1                                 0.1050584
## 4                                 0.1050584
## 5                                 0.1050584
## 12                                0.1050584
## 16                                0.1050584
## 17                                0.1050584
## 18                                0.1050584
## 38                                0.8000000
## 200                               0.8857143
## 853                               0.8000000
## 1152                              0.8000000
##      Negative.fctr.predict.Final.rpart
## 1                                    N
## 4                                    N
## 5                                    N
## 12                                   N
## 16                                   N
## 17                                   N
## 18                                   N
## 38                                   Y
## 200                                  Y
## 853                                  Y
## 1152                                 Y
##      Negative.fctr.predict.Final.rpart.accurate .label
## 1                                          TRUE     .1
## 4                                         FALSE     .4
## 5                                         FALSE     .5
## 12                                        FALSE    .12
## 16                                        FALSE    .16
## 17                                        FALSE    .17
## 18                                        FALSE    .18
## 38                                         TRUE    .38
## 200                                        TRUE   .200
## 853                                       FALSE   .853
## 1152                                      FALSE  .1152
```

![](Apple_Tweets_files/figure-html/fit.data.training.all_1-9.png) 

```r
replay.petrisim(pn=glb_analytics_pn, 
    replay.trans=(glb_analytics_avl_objs <- c(glb_analytics_avl_objs, 
        "data.training.all.prediction","model.final")), flip_coord=TRUE)
```

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0 
## 2.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction data.new.prediction 	firing:  model.selected 
## 3.0000 	 3 	 0 2 1 0 
## 3.0000 	multiple enabled transitions:  model.final data.training.all.prediction data.new.prediction 	firing:  data.training.all.prediction 
## 4.0000 	 5 	 0 1 1 1 
## 4.0000 	multiple enabled transitions:  model.final data.training.all.prediction data.new.prediction 	firing:  model.final 
## 5.0000 	 4 	 0 0 2 1
```

![](Apple_Tweets_files/figure-html/fit.data.training.all_1-10.png) 

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="predict.data.new", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                     chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed11 fit.data.training.all                6                1 319.780
## elapsed12      predict.data.new                7                0 326.243
```

## Step `7`: predict data.new

```r
# Compute final model predictions
glb_newent_df <- glb_get_predictions(glb_newent_df)
glb_analytics_diag_plots(obs_df=glb_newent_df)
```

![](Apple_Tweets_files/figure-html/predict.data.new-1.png) ![](Apple_Tweets_files/figure-html/predict.data.new-2.png) ![](Apple_Tweets_files/figure-html/predict.data.new-3.png) ![](Apple_Tweets_files/figure-html/predict.data.new-4.png) ![](Apple_Tweets_files/figure-html/predict.data.new-5.png) ![](Apple_Tweets_files/figure-html/predict.data.new-6.png) ![](Apple_Tweets_files/figure-html/predict.data.new-7.png) ![](Apple_Tweets_files/figure-html/predict.data.new-8.png) 

```
##                                                                                                                                            Tweet
## 6                                                              i want ios7 freak can it come out already im getting real tired of waiting @apple
## 8                                                                                                      lol someone @apple is asleep at the wheel
## 9                     iPhone battery was on 86% at 1am last night.. 53% when I woke up.. 20% an hour ago.. Now its on 2%. Cheers @Apple #Crapple
## 21  iOS 7 makes it 300% harder to change your wallpaper. Considering this is all the customization #iPhone users get, @Apple has some work to do
## 23                                                              sorry @apple, without @stevejobs, you're just another company run by businessmen
## 28                                                                             I don't like #iOS7 they're changing the look of everything @apple
## 31                                                                                                            @riveraoor good ol terrible @apple
## 82                                           @Punk240z I just hate @Apple not the phone. You ignored your obligation to defend the female condom
## 151                                                           Twitter and snapchat works but my iMessage always freaks up. What the freak @apple
## 666      @Microsoft Love what the design team has done with Metro but hate the recent jab @Apple.Why not just differentiate yourself from Apple?
## 997     @apple: love your products, hate inefficiencies of Genius Bar; why make an apt when I still need to wait 30mins?! http://4sq.com/1eOIu9d
##      Avg
## 6   -1.0
## 8   -1.0
## 9   -1.0
## 21  -1.0
## 23  -1.0
## 28  -1.0
## 31  -1.0
## 82  -1.0
## 151 -1.4
## 666  0.0
## 997 -0.4
##                                                                                                                                       Tweet.fctr
## 6                                                              i want ios7 freak can it come out already im getting real tired of waiting @apple
## 8                                                                                                      lol someone @apple is asleep at the wheel
## 9                     iPhone battery was on 86% at 1am last night.. 53% when I woke up.. 20% an hour ago.. Now its on 2%. Cheers @Apple #Crapple
## 21  iOS 7 makes it 300% harder to change your wallpaper. Considering this is all the customization #iPhone users get, @Apple has some work to do
## 23                                                              sorry @apple, without @stevejobs, you're just another company run by businessmen
## 28                                                                             I don't like #iOS7 they're changing the look of everything @apple
## 31                                                                                                            @riveraoor good ol terrible @apple
## 82                                           @Punk240z I just hate @Apple not the phone. You ignored your obligation to defend the female condom
## 151                                                           Twitter and snapchat works but my iMessage always freaks up. What the freak @apple
## 666      @Microsoft Love what the design team has done with Metro but hate the recent jab @Apple.Why not just differentiate yourself from Apple?
## 997     @apple: love your products, hate inefficiencies of Genius Bar; why make an apt when I still need to wait 30mins?! http://4sq.com/1eOIu9d
##         .rnorm Negative.fctr X244tsuyoponzu X7evenstarz actual add alreadi
## 6    0.7260443             Y              0           0      0   0       1
## 8   -0.8152816             Y              0           0      0   0       0
## 9    0.1064385             Y              0           0      0   0       0
## 21   0.2173778             Y              0           0      0   0       0
## 23   0.4496361             Y              0           0      0   0       0
## 28   0.0258875             Y              0           0      0   0       0
## 31   0.1693495             Y              0           0      0   0       0
## 82   0.3176902             Y              0           0      0   0       0
## 151  0.1712761             Y              0           0      0   0       0
## 666  2.0957052             N              0           0      0   0       0
## 997 -0.2398467             N              0           0      0   0       0
##     alway amaz amazon android announc anyon app appl appstor arent ask
## 6       0    0      0       0       0     0   0    0       0     0   0
## 8       0    0      0       0       0     0   0    0       0     0   0
## 9       0    0      0       0       0     0   0    0       0     0   0
## 21      0    0      0       0       0     0   0    0       0     0   0
## 23      0    0      0       0       0     0   0    0       0     0   0
## 28      0    0      0       0       0     0   0    0       0     0   0
## 31      0    0      0       0       0     0   0    0       0     0   0
## 82      0    0      0       0       0     0   0    0       0     0   0
## 151     1    0      0       0       0     0   0    0       0     0   0
## 666     0    0      0       0       0     0   0    0       0     0   0
## 997     0    0      0       0       0     0   0    0       0     0   0
##     avail away awesom back batteri best better big bit black blackberri
## 6       0    0      0    0       0    0      0   0   0     0          0
## 8       0    0      0    0       0    0      0   0   0     0          0
## 9       0    0      0    0       1    0      0   0   0     0          0
## 21      0    0      0    0       0    0      0   0   0     0          0
## 23      0    0      0    0       0    0      0   0   0     0          0
## 28      0    0      0    0       0    0      0   0   0     0          0
## 31      0    0      0    0       0    0      0   0   0     0          0
## 82      0    0      0    0       0    0      0   0   0     0          0
## 151     0    0      0    0       0    0      0   0   0     0          0
## 666     0    0      0    0       0    0      0   0   0     0          0
## 997     0    0      0    0       0    0      0   0   0     0          0
##     break. bring burberri busi buy call can cant carbon card care case cdp
## 6        0     0        0    0   0    0   1    0      0    0    0    0   0
## 8        0     0        0    0   0    0   0    0      0    0    0    0   0
## 9        0     0        0    0   0    0   0    0      0    0    0    0   0
## 21       0     0        0    0   0    0   0    0      0    0    0    0   0
## 23       0     0        0    0   0    0   0    0      0    0    0    0   0
## 28       0     0        0    0   0    0   0    0      0    0    0    0   0
## 31       0     0        0    0   0    0   0    0      0    0    0    0   0
## 82       0     0        0    0   0    0   0    0      0    0    0    0   0
## 151      0     0        0    0   0    0   0    0      0    0    0    0   0
## 666      0     0        0    0   0    0   0    0      0    0    0    0   0
## 997      0     0        0    0   0    0   0    0      0    0    0    0   0
##     chang charg charger cheap china color colour come compani condescens
## 6       0     0       0     0     0     0      0    1       0          0
## 8       0     0       0     0     0     0      0    0       0          0
## 9       0     0       0     0     0     0      0    0       0          0
## 21      1     0       0     0     0     0      0    0       0          0
## 23      0     0       0     0     0     0      0    0       1          0
## 28      1     0       0     0     0     0      0    0       0          0
## 31      0     0       0     0     0     0      0    0       0          0
## 82      0     0       0     0     0     0      0    0       0          0
## 151     0     0       0     0     0     0      0    0       0          0
## 666     0     0       0     0     0     0      0    0       0          0
## 997     0     0       0     0     0     0      0    0       0          0
##     condom copi crack creat custom darn data date day dear design develop
## 6        0    0     0     0      0    0    0    0   0    0      0       0
## 8        0    0     0     0      0    0    0    0   0    0      0       0
## 9        0    0     0     0      0    0    0    0   0    0      0       0
## 21       0    0     0     0      1    0    0    0   0    0      0       0
## 23       0    0     0     0      0    0    0    0   0    0      0       0
## 28       0    0     0     0      0    0    0    0   0    0      0       0
## 31       0    0     0     0      0    0    0    0   0    0      0       0
## 82       1    0     0     0      0    0    0    0   0    0      0       0
## 151      0    0     0     0      0    0    0    0   0    0      0       0
## 666      0    0     0     0      0    0    0    0   0    0      1       0
## 997      0    0     0     0      0    0    0    0   0    0      0       0
##     devic didnt die differ disappoint discontinu divulg doesnt done dont
## 6       0     0   0      0          0          0      0      0    0    0
## 8       0     0   0      0          0          0      0      0    0    0
## 9       0     0   0      0          0          0      0      0    0    0
## 21      0     0   0      0          0          0      0      0    0    0
## 23      0     0   0      0          0          0      0      0    0    0
## 28      0     0   0      0          0          0      0      0    0    1
## 31      0     0   0      0          0          0      0      0    0    0
## 82      0     0   0      0          0          0      0      0    0    0
## 151     0     0   0      0          0          0      0      0    0    0
## 666     0     0   0      0          0          0      0      0    1    0
## 997     0     0   0      0          0          0      0      0    0    0
##     download drop email emiss emoji even event ever everi everyth facebook
## 6          0    0     0     0     0    0     0    0     0       0        0
## 8          0    0     0     0     0    0     0    0     0       0        0
## 9          0    0     0     0     0    0     0    0     0       0        0
## 21         0    0     0     0     0    0     0    0     0       0        0
## 23         0    0     0     0     0    0     0    0     0       0        0
## 28         0    0     0     0     0    0     0    0     0       1        0
## 31         0    0     0     0     0    0     0    0     0       0        0
## 82         0    0     0     0     0    0     0    0     0       0        0
## 151        0    0     0     0     0    0     0    0     0       0        0
## 666        0    0     0     0     0    0     0    0     0       0        0
## 997        0    0     0     0     0    0     0    0     0       0        0
##     fail featur feel femal figur final finger fingerprint fire first fix
## 6      0      0    0     0     0     0      0           0    0     0   0
## 8      0      0    0     0     0     0      0           0    0     0   0
## 9      0      0    0     0     0     0      0           0    0     0   0
## 21     0      0    0     0     0     0      0           0    0     0   0
## 23     0      0    0     0     0     0      0           0    0     0   0
## 28     0      0    0     0     0     0      0           0    0     0   0
## 31     0      0    0     0     0     0      0           0    0     0   0
## 82     0      0    0     1     0     0      0           0    0     0   0
## 151    0      0    0     0     0     0      0           0    0     0   0
## 666    0      0    0     0     0     0      0           0    0     0   0
## 997    0      0    0     0     0     0      0           0    0     0   0
##     follow freak free fun generat genius get give gold gonna good googl
## 6        0     1    0   0       0      0   1    0    0     0    0     0
## 8        0     0    0   0       0      0   0    0    0     0    0     0
## 9        0     0    0   0       0      0   0    0    0     0    0     0
## 21       0     0    0   0       0      0   1    0    0     0    0     0
## 23       0     0    0   0       0      0   0    0    0     0    0     0
## 28       0     0    0   0       0      0   0    0    0     0    0     0
## 31       0     0    0   0       0      0   0    0    0     0    1     0
## 82       0     0    0   0       0      0   0    0    0     0    0     0
## 151      0     2    0   0       0      0   0    0    0     0    0     0
## 666      0     0    0   0       0      0   0    0    0     0    0     0
## 997      0     0    0   0       0      1   0    0    0     0    0     0
##     got great guess guy happen happi hate help hey hope hour
## 6     0     0     0   0      0     0    0    0   0    0    0
## 8     0     0     0   0      0     0    0    0   0    0    0
## 9     0     0     0   0      0     0    0    0   0    0    1
## 21    0     0     0   0      0     0    0    0   0    0    0
## 23    0     0     0   0      0     0    0    0   0    0    0
## 28    0     0     0   0      0     0    0    0   0    0    0
## 31    0     0     0   0      0     0    0    0   0    0    0
## 82    0     0     0   0      0     0    1    0   0    0    0
## 151   0     0     0   0      0     0    0    0   0    0    0
## 666   0     0     0   0      0     0    1    0   0    0    0
## 997   0     0     0   0      0     0    1    0   0    0    0
##     httpbitly18xc8dk ibrooklynb idea ill imessag impress improv innov
## 6                  0          0    0   0       0       0      0     0
## 8                  0          0    0   0       0       0      0     0
## 9                  0          0    0   0       0       0      0     0
## 21                 0          0    0   0       0       0      0     0
## 23                 0          0    0   0       0       0      0     0
## 28                 0          0    0   0       0       0      0     0
## 31                 0          0    0   0       0       0      0     0
## 82                 0          0    0   0       0       0      0     0
## 151                0          0    0   0       1       0      0     0
## 666                0          0    0   0       0       0      0     0
## 997                0          0    0   0       0       0      0     0
##     instead internet ios7 ipad iphon iphone4 iphone5 iphone5c iphoto ipod
## 6         0        0    1    0     0       0       0        0      0    0
## 8         0        0    0    0     0       0       0        0      0    0
## 9         0        0    0    0     1       0       0        0      0    0
## 21        0        0    0    0     1       0       0        0      0    0
## 23        0        0    0    0     0       0       0        0      0    0
## 28        0        0    1    0     0       0       0        0      0    0
## 31        0        0    0    0     0       0       0        0      0    0
## 82        0        0    0    0     0       0       0        0      0    0
## 151       0        0    0    0     0       0       0        0      0    0
## 666       0        0    0    0     0       0       0        0      0    0
## 997       0        0    0    0     0       0       0        0      0    0
##     ipodplayerpromo isnt itun ive job just keynot know last launch let
## 6                 0    0    0   0   0    0      0    0    0      0   0
## 8                 0    0    0   0   0    0      0    0    0      0   0
## 9                 0    0    0   0   0    0      0    0    1      0   0
## 21                0    0    0   0   0    0      0    0    0      0   0
## 23                0    0    0   0   0    1      0    0    0      0   0
## 28                0    0    0   0   0    0      0    0    0      0   0
## 31                0    0    0   0   0    0      0    0    0      0   0
## 82                0    0    0   0   0    1      0    0    0      0   0
## 151               0    0    0   0   0    0      0    0    0      0   0
## 666               0    0    0   0   0    1      0    0    0      0   0
## 997               0    0    0   0   0    0      0    0    0      0   0
##     life like line lmao lock lol look los lost love mac macbook made make
## 6      0    0    0    0    0   0    0   0    0    0   0       0    0    0
## 8      0    0    0    0    0   1    0   0    0    0   0       0    0    0
## 9      0    0    0    0    0   0    0   0    0    0   0       0    0    0
## 21     0    0    0    0    0   0    0   0    0    0   0       0    0    1
## 23     0    0    0    0    0   0    0   0    0    0   0       0    0    0
## 28     0    1    0    0    0   0    1   0    0    0   0       0    0    0
## 31     0    0    0    0    0   0    0   0    0    0   0       0    0    0
## 82     0    0    0    0    0   0    0   0    0    0   0       0    0    0
## 151    0    0    0    0    0   0    0   0    0    0   0       0    0    0
## 666    0    0    0    0    0   0    0   0    0    1   0       0    0    0
## 997    0    0    0    0    0   0    0   0    0    1   0       0    0    1
##     man mani market mayb mean microsoft mishiza miss mobil money motorola
## 6     0    0      0    0    0         0       0    0     0     0        0
## 8     0    0      0    0    0         0       0    0     0     0        0
## 9     0    0      0    0    0         0       0    0     0     0        0
## 21    0    0      0    0    0         0       0    0     0     0        0
## 23    0    0      0    0    0         0       0    0     0     0        0
## 28    0    0      0    0    0         0       0    0     0     0        0
## 31    0    0      0    0    0         0       0    0     0     0        0
## 82    0    0      0    0    0         0       0    0     0     0        0
## 151   0    0      0    0    0         0       0    0     0     0        0
## 666   0    0      0    0    0         1       0    0     0     0        0
## 997   0    0      0    0    0         0       0    0     0     0        0
##     move much music natz0711 need never new news next. nfc nokia noth now
## 6      0    0     0        0    0     0   0    0     0   0     0    0   0
## 8      0    0     0        0    0     0   0    0     0   0     0    0   0
## 9      0    0     0        0    0     0   0    0     0   0     0    0   1
## 21     0    0     0        0    0     0   0    0     0   0     0    0   0
## 23     0    0     0        0    0     0   0    0     0   0     0    0   0
## 28     0    0     0        0    0     0   0    0     0   0     0    0   0
## 31     0    0     0        0    0     0   0    0     0   0     0    0   0
## 82     0    0     0        0    0     0   0    0     0   0     0    0   0
## 151    0    0     0        0    0     0   0    0     0   0     0    0   0
## 666    0    0     0        0    0     0   0    0     0   0     0    0   0
## 997    0    0     0        0    1     0   0    0     0   0     0    0   0
##     nsa nuevo offer old one page para peopl perfect person phone photog
## 6     0     0     0   0   0    0    0     0       0      0     0      0
## 8     0     0     0   0   0    0    0     0       0      0     0      0
## 9     0     0     0   0   0    0    0     0       0      0     0      0
## 21    0     0     0   0   0    0    0     0       0      0     0      0
## 23    0     0     0   0   0    0    0     0       0      0     0      0
## 28    0     0     0   0   0    0    0     0       0      0     0      0
## 31    0     0     0   0   0    0    0     0       0      0     0      0
## 82    0     0     0   0   0    0    0     0       0      0     1      0
## 151   0     0     0   0   0    0    0     0       0      0     0      0
## 666   0     0     0   0   0    0    0     0       0      0     0      0
## 997   0     0     0   0   0    0    0     0       0      0     0      0
##     photographi pictur plastic play pleas ppl preorder price print pro
## 6             0      0       0    0     0   0        0     0     0   0
## 8             0      0       0    0     0   0        0     0     0   0
## 9             0      0       0    0     0   0        0     0     0   0
## 21            0      0       0    0     0   0        0     0     0   0
## 23            0      0       0    0     0   0        0     0     0   0
## 28            0      0       0    0     0   0        0     0     0   0
## 31            0      0       0    0     0   0        0     0     0   0
## 82            0      0       0    0     0   0        0     0     0   0
## 151           0      0       0    0     0   0        0     0     0   0
## 666           0      0       0    0     0   0        0     0     0   0
## 997           0      0       0    0     0   0        0     0     0   0
##     problem product promo promoipodplayerpromo put que quiet read realli
## 6         0       0     0                    0   0   0     0    0      0
## 8         0       0     0                    0   0   0     0    0      0
## 9         0       0     0                    0   0   0     0    0      0
## 21        0       0     0                    0   0   0     0    0      0
## 23        0       0     0                    0   0   0     0    0      0
## 28        0       0     0                    0   0   0     0    0      0
## 31        0       0     0                    0   0   0     0    0      0
## 82        0       0     0                    0   0   0     0    0      0
## 151       0       0     0                    0   0   0     0    0      0
## 666       0       0     0                    0   0   0     0    0      0
## 997       0       1     0                    0   0   0     0    0      0
##     recommend refus releas right said samsung samsungsa say scanner screen
## 6           0     0      0     0    0       0         0   0       0      0
## 8           0     0      0     0    0       0         0   0       0      0
## 9           0     0      0     0    0       0         0   0       0      0
## 21          0     0      0     0    0       0         0   0       0      0
## 23          0     0      0     0    0       0         0   0       0      0
## 28          0     0      0     0    0       0         0   0       0      0
## 31          0     0      0     0    0       0         0   0       0      0
## 82          0     0      0     0    0       0         0   0       0      0
## 151         0     0      0     0    0       0         0   0       0      0
## 666         0     0      0     0    0       0         0   0       0      0
## 997         0     0      0     0    0       0         0   0       0      0
##     secur see seem sell send servic shame share short show simpl sinc siri
## 6       0   0    0    0    0      0     0     0     0    0     0    0    0
## 8       0   0    0    0    0      0     0     0     0    0     0    0    0
## 9       0   0    0    0    0      0     0     0     0    0     0    0    0
## 21      0   0    0    0    0      0     0     0     0    0     0    0    0
## 23      0   0    0    0    0      0     0     0     0    0     0    0    0
## 28      0   0    0    0    0      0     0     0     0    0     0    0    0
## 31      0   0    0    0    0      0     0     0     0    0     0    0    0
## 82      0   0    0    0    0      0     0     0     0    0     0    0    0
## 151     0   0    0    0    0      0     0     0     0    0     0    0    0
## 666     0   0    0    0    0      0     0     0     0    0     0    0    0
## 997     0   0    0    0    0      0     0     0     0    0     0    0    0
##     smart smartphon someth soon stand start steve still stop store stuff
## 6       0         0      0    0     0     0     0     0    0     0     0
## 8       0         0      0    0     0     0     0     0    0     0     0
## 9       0         0      0    0     0     0     0     0    0     0     0
## 21      0         0      0    0     0     0     0     0    0     0     0
## 23      0         0      0    0     0     0     0     0    0     0     0
## 28      0         0      0    0     0     0     0     0    0     0     0
## 31      0         0      0    0     0     0     0     0    0     0     0
## 82      0         0      0    0     0     0     0     0    0     0     0
## 151     0         0      0    0     0     0     0     0    0     0     0
## 666     0         0      0    0     0     0     0     0    0     0     0
## 997     0         0      0    0     0     0     0     1    0     0     0
##     stupid suck support sure switch take talk team tech technolog tell
## 6        0    0       0    0      0    0    0    0    0         0    0
## 8        0    0       0    0      0    0    0    0    0         0    0
## 9        0    0       0    0      0    0    0    0    0         0    0
## 21       0    0       0    0      0    0    0    0    0         0    0
## 23       0    0       0    0      0    0    0    0    0         0    0
## 28       0    0       0    0      0    0    0    0    0         0    0
## 31       0    0       0    0      0    0    0    0    0         0    0
## 82       0    0       0    0      0    0    0    0    0         0    0
## 151      0    0       0    0      0    0    0    0    0         0    0
## 666      0    0       0    0      0    0    0    1    0         0    0
## 997      0    0       0    0      0    0    0    0    0         0    0
##     text thank that theyr thing think tho thought time today togeth touch
## 6      0     0    0     0     0     0   0       0    0     0      0     0
## 8      0     0    0     0     0     0   0       0    0     0      0     0
## 9      0     0    0     0     0     0   0       0    0     0      0     0
## 21     0     0    0     0     0     0   0       0    0     0      0     0
## 23     0     0    0     0     0     0   0       0    0     0      0     0
## 28     0     0    0     1     0     0   0       0    0     0      0     0
## 31     0     0    0     0     0     0   0       0    0     0      0     0
## 82     0     0    0     0     0     0   0       0    0     0      0     0
## 151    0     0    0     0     0     0   0       0    0     0      0     0
## 666    0     0    0     0     0     0   0       0    0     0      0     0
## 997    0     0    0     0     0     0   0       0    0     0      0     0
##     touchid tri true turn twitter two updat upgrad use user via video wait
## 6         0   0    0    0       0   0     0      0   0    0   0     0    1
## 8         0   0    0    0       0   0     0      0   0    0   0     0    0
## 9         0   0    0    0       0   0     0      0   0    0   0     0    0
## 21        0   0    0    0       0   0     0      0   0    1   0     0    0
## 23        0   0    0    0       0   0     0      0   0    0   0     0    0
## 28        0   0    0    0       0   0     0      0   0    0   0     0    0
## 31        0   0    0    0       0   0     0      0   0    0   0     0    0
## 82        0   0    0    0       0   0     0      0   0    0   0     0    0
## 151       0   0    0    0       1   0     0      0   0    0   0     0    0
## 666       0   0    0    0       0   0     0      0   0    0   0     0    0
## 997       0   0    0    0       0   0     0      0   0    0   0     0    1
##     want watch way week well what white will windowsphon wish without
## 6      1     0   0    0    0    0     0    0           0    0       0
## 8      0     0   0    0    0    0     0    0           0    0       0
## 9      0     0   0    0    0    0     0    0           0    0       0
## 21     0     0   0    0    0    0     0    0           0    0       0
## 23     0     0   0    0    0    0     0    0           0    0       1
## 28     0     0   0    0    0    0     0    0           0    0       0
## 31     0     0   0    0    0    0     0    0           0    0       0
## 82     0     0   0    0    0    0     0    0           0    0       0
## 151    0     0   0    0    0    0     0    0           0    0       0
## 666    0     0   0    0    0    0     0    0           0    0       0
## 997    0     0   0    0    0    0     0    0           0    0       0
##     wonder wont work world worst wow wtf yall year yes yet yooo your
## 6        0    0    0     0     0   0   0    0    0   0   0    0    0
## 8        0    0    0     0     0   0   0    0    0   0   0    0    0
## 9        0    0    0     0     0   0   0    0    0   0   0    0    0
## 21       0    0    1     0     0   0   0    0    0   0   0    0    0
## 23       0    0    0     0     0   0   0    0    0   0   0    0    1
## 28       0    0    0     0     0   0   0    0    0   0   0    0    0
## 31       0    0    0     0     0   0   0    0    0   0   0    0    0
## 82       0    0    0     0     0   0   0    0    0   0   0    0    0
## 151      0    0    1     0     0   0   0    0    0   0   0    0    0
## 666      0    0    0     0     0   0   0    0    0   0   0    0    0
## 997      0    0    0     0     0   0   0    0    0   0   0    0    0
##     Negative.fctr.predict.Final.rpart.prob
## 6                                0.8857143
## 8                                0.1050584
## 9                                0.1050584
## 21                               0.1050584
## 23                               0.1050584
## 28                               0.1050584
## 31                               0.1050584
## 82                               0.8000000
## 151                              0.8857143
## 666                              0.8000000
## 997                              0.8000000
##     Negative.fctr.predict.Final.rpart
## 6                                   Y
## 8                                   N
## 9                                   N
## 21                                  N
## 23                                  N
## 28                                  N
## 31                                  N
## 82                                  Y
## 151                                 Y
## 666                                 Y
## 997                                 Y
##     Negative.fctr.predict.Final.rpart.accurate .label
## 6                                         TRUE     .6
## 8                                        FALSE     .8
## 9                                        FALSE     .9
## 21                                       FALSE    .21
## 23                                       FALSE    .23
## 28                                       FALSE    .28
## 31                                       FALSE    .31
## 82                                        TRUE    .82
## 151                                       TRUE   .151
## 666                                      FALSE   .666
## 997                                      FALSE   .997
```

![](Apple_Tweets_files/figure-html/predict.data.new-9.png) 

```r
tmp_replay_lst <- replay.petrisim(pn=glb_analytics_pn, 
    replay.trans=(glb_analytics_avl_objs <- c(glb_analytics_avl_objs, 
        "data.new.prediction")), flip_coord=TRUE)
```

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0 
## 2.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction data.new.prediction 	firing:  model.selected 
## 3.0000 	 3 	 0 2 1 0 
## 3.0000 	multiple enabled transitions:  model.final data.training.all.prediction data.new.prediction 	firing:  data.training.all.prediction 
## 4.0000 	 5 	 0 1 1 1 
## 4.0000 	multiple enabled transitions:  model.final data.training.all.prediction data.new.prediction 	firing:  model.final 
## 5.0000 	 4 	 0 0 2 1 
## 6.0000 	 6 	 0 0 1 2
```

![](Apple_Tweets_files/figure-html/predict.data.new-10.png) 

```r
print(ggplot.petrinet(tmp_replay_lst[["pn"]]) + coord_flip())
```

![](Apple_Tweets_files/figure-html/predict.data.new-11.png) 

Null Hypothesis ($\sf{H_{0}}$): mpg is not impacted by am_fctr.  
The variance by am_fctr appears to be independent. 
#```{r q1, cache=FALSE}
# print(t.test(subset(cars_df, am_fctr == "automatic")$mpg, 
#              subset(cars_df, am_fctr == "manual")$mpg, 
#              var.equal=FALSE)$conf)
#```
We reject the null hypothesis i.e. we have evidence to conclude that am_fctr impacts mpg (95% confidence). Manual transmission is better for miles per gallon versus automatic transmission.


```
##                   chunk_label chunk_step_major chunk_step_minor elapsed
## 10                 fit.models                5                1 299.425
## 11      fit.data.training.all                6                0 313.567
## 9                  fit.models                5                0  24.668
## 7             select_features                4                0  14.222
## 13           predict.data.new                7                0 326.243
## 12      fit.data.training.all                6                1 319.780
## 6            extract_features                3                0   6.366
## 4         manage_missing_data                2                2   1.620
## 8  remove_correlated_features                4                1  15.261
## 5         encodeORretype.data                2                3   2.161
## 2                cleanse_data                2                0   0.503
## 3       inspectORexplore.data                2                1   0.538
## 1                 import_data                1                0   0.002
##    elapsed_diff
## 10      274.757
## 11       14.142
## 9         9.407
## 7         7.856
## 13        6.463
## 12        6.213
## 6         4.205
## 4         1.082
## 8         1.039
## 5         0.541
## 2         0.501
## 3         0.035
## 1         0.000
```

```
## [1] "Total Elapsed Time: 326.243 secs"
```

![](Apple_Tweets_files/figure-html/print_sessionInfo-1.png) 

```
## R version 3.1.3 (2015-03-09)
## Platform: x86_64-apple-darwin13.4.0 (64-bit)
## Running under: OS X 10.10.3 (Yosemite)
## 
## locale:
## [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8
## 
## attached base packages:
## [1] tcltk     grid      stats     graphics  grDevices utils     datasets 
## [8] methods   base     
## 
## other attached packages:
##  [1] randomForest_4.6-10 rpart.plot_1.5.2    rpart_4.1-9        
##  [4] ROCR_1.0-7          gplots_2.16.0       caret_6.0-41       
##  [7] lattice_0.20-31     tm_0.6              NLP_0.1-6          
## [10] sqldf_0.4-10        RSQLite_1.0.0       DBI_0.3.1          
## [13] gsubfn_0.6-6        proto_0.3-10        reshape2_1.4.1     
## [16] plyr_1.8.1          caTools_1.17.1      doBy_4.5-13        
## [19] survival_2.38-1     ggplot2_1.0.1      
## 
## loaded via a namespace (and not attached):
##  [1] bitops_1.0-6        BradleyTerry2_1.0-6 brglm_0.5-9        
##  [4] car_2.0-25          chron_2.3-45        class_7.3-12       
##  [7] codetools_0.2-11    colorspace_1.2-6    compiler_3.1.3     
## [10] digest_0.6.8        e1071_1.6-4         evaluate_0.5.5     
## [13] foreach_1.4.2       formatR_1.1         gdata_2.13.3       
## [16] gtable_0.1.2        gtools_3.4.1        htmltools_0.2.6    
## [19] iterators_1.0.7     KernSmooth_2.23-14  knitr_1.9          
## [22] labeling_0.3        lme4_1.1-7          MASS_7.3-40        
## [25] Matrix_1.2-0        mgcv_1.8-6          minqa_1.2.4        
## [28] munsell_0.4.2       nlme_3.1-120        nloptr_1.0.4       
## [31] nnet_7.3-9          parallel_3.1.3      pbkrtest_0.4-2     
## [34] quantreg_5.11       RColorBrewer_1.1-2  Rcpp_0.11.5        
## [37] rmarkdown_0.5.1     scales_0.2.4        slam_0.1-32        
## [40] SparseM_1.6         splines_3.1.3       stringr_0.6.2      
## [43] tools_3.1.3         yaml_2.1.13
```
