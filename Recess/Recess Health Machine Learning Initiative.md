---
Author: Eduardo Abdala Rivera
Title: Recess Health Machine Learning Initiative
---

# Recess Health Machine Learning Initiative

### Goals

- Explore which machine learning algorithms can be applied to Recess Health use cases
- Shift the company revenue cycle to the left (faster balance payment) and up (increase overall balance payment)
- Describe patient's propensity to pay in 120 days in terms of feature importance
- Implement data-driven policies that will improve patient satisfaction post-bill initiation



### Introduction

With the advent of machine learning in the last decade, machine learning is proving effective in providing business insights and solutions to problems that are more nuanced for humans to solve. Many algorithms have been designed for different research questions and types of data. Our goal is to provide a reproducible use case for the medical billing revenue cycle space that optimizes the allocation of company resources with regards to time spent calling and other considerations.

An algorithm of interest is the gradient boosted decision tree. This involves optimizing a loss function, using "weak learners" to train the model, and an additive model that adds more weak learners such that:
$$
\begin{align*}
F(x) = \sum_{m=1}^{M} \gamma_m h_m(x)
\end{align*}
$$
where $\ h_{m}(x)$ are the weak learners.



### Current Situation

Dialer qualification takes all of the info from active accounts and determines if we should call them based on [established criteria.](https://github.com/RecessHealth/Documentation/blob/master/current_scoring_criteria)

Preliminary data exploration found a few key metrics. A data set was imported to Jupyter notebook using the datascience package.

```python
from datascience import *
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
import numpy as np

bench = Tabe.read_table('Benchmark_metricDaystoPIF.csv')
bench.head(10)
```

We imported the data from our in-house database. The data was extracted using Microsoft SQL Server Management Studio and cleaned using the above python packages.

* ID = dummy list of ID's assigned to invoices
* fully_paid_amount = the full amount of the invoice
* pif_date = the day they paid their balance in full
* days_to_pif = the amount of days it took for patient to pay in full (pif)

![Screenshot from 2019-03-08 15_22_56](/home/edabdala/Pictures/Screenshot from 2019-03-08 15_22_56.png)

The distributions for days_to_pif is shown below. The first graph shows the entire set of data and the second shows a zoomed in version with more visibility on the majority of the data.

```python
bench.scatter('days_to_pif')
```

```python
plt.plot(x,y)
axes = plt.gca()
axes.set_xlim([0,120])
axes.set_ylim([0.4450])
```

![scatterbig.png](/home/edabdala/Pictures/scatterbig.png)![scattersmall.png](/home/edabdala/Pictures/scattersmall.png)

We found that most patients will PIF in 16 days with average at 67 days. This will be useful information used to compare any implemented machine learning-driven policy. We want to see if an optimized model will allow us to bring our days_to_pif number down, effectively shortening the amount of time it takes for us to close an account and free up company resources.

### Work with BigSquid

BigSquid's Kraken is an automated machine learning (AutoML) platform that aims to give non-technical users business insights

When data is imported into Kraken, it goes through automatic preprocessing, which includes normalizing, cleaning null values, and encoding of categorical data. Then the platform will partition the imported data to perfom cross-validation. This means using data fro the same source as the training set to test how accurate the given ML model is.

> The basic process is as follows:
>
> 1. Data is shuffled randomly
> 2. Data is partitioned into 5 groups
> 3. For each group:
>    - a test data set is created
>    - the rest is used for training the model
>    - model is fit to the training set to evaluate against test set
>    - evaluation score is recorded, validation model dissipates

Possible ML algorithms in Kraken:

​	Random Forest

​	Logistic Regression

​	XGBoost

​	Nearest Neighbors

​	Support Vector

​	Linear Regression

​	Stochastic Gradient Descent

Kraken automatically assigns a model to the algorithm that performs the best. The designation of “the best” is determined from either the F1 score for classification models or the R2. . The business question that Recess is answering through these initial Kraken Models is a classification question. Classification models are scored through their F1 score.  

When a dataset is loaded into Kraken, Kraken performs analysis on the data and consequently runs the dataset through each of the above defined algorithms. A list of 4-5 algorithms is given post analysis that highlights which algorithms performed the best. While Kraken automatically chooses the algorithm that performed best, there is still autonomy in that users can change which algorithm is used at any time.

Statistics shown inside of Kraken that help assist in determining which algorithm should be used and how well the model is performing:  

**F1 score:** the harmonic average between precision and recall

**Precision:** true positives/ no. of predicted positives

**Recall:** true positives/ no. of actual positives  

**AUC:** the area under the curve indicates how well the model can distinguish between the two classes in question

***\*the above four statistics are arguably the most important when determining the performance and accuracy of a model\****  

**Miss Rate:** false negatives/ no. of actual positives  

**Fallout:** false positives/ no. of actual negatives (opposite of recall)

**Specificity:** true negatives/ no. of actual negatives  

**Accuracy:** (true positives + false positives) / total population

**Negative Predictive Value:** true negatives / no. of predicted negative  

**Matthews Correlation Coefficient:** takes all of the cells of the Confusion Matrix into consideration in its formula. Gives a general sense of model performance.



**Kraken Scoring and Processing as it Relates to Recess’ Models:**

**First Model**

In the first model created for Recess by Kyle Jourdan, the highest scoring algorithm (in terms of an F1 score) was an XGBoost Classifier. This model was chosen from scores presented after unsupervised learning on the appended training set that was fed into Kraken. The information contained in the dataset used for this model can be found under the header “First Model Built”

TheXGBoost algorithm is a highly sophisticated implementation of gradient boosting algorithms. The advantages to using XGBoost to model are its regularization of data, and ability to handle missing values. The automated technology of Kraken automatically handles that optimization.  

**Second Model**  

In the second model created for Recess by Jordan Ganung, the highest performing algorithm was a Random Forest Classifier. The F1 score was an 84%, recall and precision were 81% and 86% respectively, and the AUC was .941. This model was created to eliminate the data leakage present in the first model (created by Kyle).  

The Random Forest Classifier was chosen from scores presented after learning on the appended training set that was fed into Kraken. Random Forest is a supervised learning algorithm that builds multiple decision trees that then merge together. The benefit of multiple trees that merge is that the predictions that stem from such algorithms tend to be more accurate and stable (even with variance among results). The information contained in the dataset used for this model can be found under the header “Model to Eliminate Data Leakage”

**Third Model**  

The third model built for Recess was built to predict at 30 days after assignment who would PIF in 120 days. *Statistics to follow: Jay is compiling this dataset and will run it through Kraken*.  

**First Model Built:**  

Kyle Jourdan had access to RWJ Barnabas data first and built an initial model. That model included the following features (i.e. columns/ data points):

![Screenshot from 2019-03-08 14_38_26](/home/edabdala/Downloads/Screenshot from 2019-03-08 14_38_26.png)

The SQL query that was used to create base table as defined above is as follows:  

CREATE OR REPLACE TABLE "KRAKEN_PIF_120_STAGE"

AS

SELECT "a"."account_id"

​    ,"a"."client_id"

​    ,"a"."patient_age"

​    ,"a"."zip_code"

​    ,COALESCE("b"."CITY_NAME",'UNKNOWN') AS "city_name"

​    ,COALESCE("b"."COUNTY_NAME",'UNKNOWN') AS "county_name"

​    ,COALESCE("b"."STATE_NAME",'UNKNOWN') AS "state_name"

​    ,COALESCE("b"."POPULATION",0) AS "population_estimate"

​    ,COALESCE("b"."DENSITY",0.00) AS "density_estimate"

​    ,COALESCE("b"."TIMEZONE",'UNKNOWN') AS "local_timezone"

​    ,"a"."financial_class"

​    ,"a"."patient_type"

​    ,"a"."med_service"

​    ,"a"."status_code"

​    ,"a"."previous_status_code"

​    ,"a"."employer_known"

​    ,"a"."is_employed"

​    ,"a"."assigned_date"

​    ,YEAR("a"."assigned_date") AS "assigned_year"

​    ,MONTH("a"."assigned_date") AS "assigned_month"

​    ,WEEK("a"."assigned_date") AS "assigned_week"

​    ,"a"."initial_balance"

​    ,"a"."cancelled_balance"

​    ,"a"."paid_in_30"

​    ,"a"."paid_in_60"

​    ,"a"."paid_in_90"

​    ,"a"."paid_in_120"

​    ,"a"."paid_in_180"

​    ,"a"."paid_in_365"

​    ,COALESCE("a"."transaction_total",0.00) AS "transaction_total"

​    ,"a"."transaction_count"

​    ,"a"."trans_in_30"

​    ,"a"."trans_in_60"

​    ,"a"."trans_in_90"

​    ,"a"."trans_in_120"

​    ,CASE WHEN CURRENT_DATE() < DATEADD('day',120,"a"."assigned_date") AND ("a"."paid_in_120" < "a"."initial_balance") THEN NULL ELSE

​        CASE WHEN "a"."initial_balance" = 0 THEN 'false' ELSE

​            CASE WHEN ("a"."paid_in_120" / "a"."initial_balance") >= 0.9 THEN 'true' ELSE 'false' END

​        END

​     END AS "pif_in_120"

FROM (

SELECT "a"."ACCOUNT_ID" AS "account_id"

​    ,"a"."CLIENT_ID" AS "client_id"

​    ,"a"."PATIENT_AGE" AS "patient_age"

​    ,LEFT(LPAD("a"."ZIP_CODE",5,'0'),5) AS "zip_code"

​    ,COALESCE(NULLIF(NULLIF("a"."FINANCIAL_CLASS",'NULL'),''),'UNKNOWN') AS "financial_class"

​    ,COALESCE(NULLIF(NULLIF("a"."PATIENT_TYPE",'NULL'),''),'UNKNOWN') AS "patient_type"

​    ,COALESCE(NULLIF(NULLIF("a"."MED_SERVICE",'NULL'),''),'UNKNOWN') AS "med_service"

​    ,"a"."STATUS_CODE" AS "status_code"

​    ,"a"."PREVIOUS_STATUS_CODE" AS "previous_status_code"

​    ,CASE WHEN "a"."EMPLOYER" IN ('NULL','UNKNOWN') THEN '0' ELSE '1' END AS "employer_known"

​    ,"a"."EMPLOYED" AS "is_employed"

​    ,"a"."ASSIGNED_DATE"::DATE AS "assigned_date"

​    ,"a"."LETTERS_SENT" AS "letter_sent"

​    ,"a"."INITIAL_BALANCE" AS "initial_balance"

​    ,"a"."CANCELLED_AMOUNT" AS "cancelled_balance"

​    ,COUNT(DISTINCT "b"."ID") AS "transaction_count"

​    ,COUNT(DISTINCT CASE WHEN "b"."TRANS_DATE"::DATE <= DATEADD('day',30,"a"."ASSIGNED_DATE"::DATE) THEN "b"."ID" END) AS "trans_in_30"

​    ,COUNT(DISTINCT CASE WHEN "b"."TRANS_DATE"::DATE <= DATEADD('day',60,"a"."ASSIGNED_DATE"::DATE) THEN "b"."ID" END) AS "trans_in_60"

​    ,COUNT(DISTINCT CASE WHEN "b"."TRANS_DATE"::DATE <= DATEADD('day',90,"a"."ASSIGNED_DATE"::DATE) THEN "b"."ID" END) AS "trans_in_90"

​    ,COUNT(DISTINCT CASE WHEN "b"."TRANS_DATE"::DATE <= DATEADD('day',120,"a"."ASSIGNED_DATE"::DATE) THEN "b"."ID" END) AS "trans_in_120"

​    ,SUM(CASE WHEN "b"."TRANS_DATE"::DATE <= DATEADD('day',30,"a"."ASSIGNED_DATE"::DATE) THEN "b"."AMOUNT" ELSE 0.00 END) AS "paid_in_30"

​    ,SUM(CASE WHEN "b"."TRANS_DATE"::DATE <= DATEADD('day',60,"a"."ASSIGNED_DATE"::DATE) THEN "b"."AMOUNT" ELSE 0.00 END) AS "paid_in_60"

​    ,SUM(CASE WHEN "b"."TRANS_DATE"::DATE <= DATEADD('day',90,"a"."ASSIGNED_DATE"::DATE) THEN "b"."AMOUNT" ELSE 0.00 END) AS "paid_in_90"

​    ,SUM(CASE WHEN "b"."TRANS_DATE"::DATE <= DATEADD('day',120,"a"."ASSIGNED_DATE"::DATE) THEN "b"."AMOUNT" ELSE 0.00 END) AS "paid_in_120"

​    ,SUM(CASE WHEN "b"."TRANS_DATE"::DATE <= DATEADD('day',180,"a"."ASSIGNED_DATE"::DATE) THEN "b"."AMOUNT" ELSE 0.00 END) AS "paid_in_180"

​    ,SUM(CASE WHEN "b"."TRANS_DATE"::DATE <= DATEADD('day',365,"a"."ASSIGNED_DATE"::DATE) THEN "b"."AMOUNT" ELSE 0.00 END) AS "paid_in_365"

​    ,COALESCE(SUM("b"."AMOUNT"),'0.00') AS "transaction_total"

FROM "SQUIDBITS"."PFS_GROUP"."ACCOUNTS" AS "a"

LEFT JOIN "SQUIDBITS"."PFS_GROUP"."TRANSACTION_RECORDS" AS "b" ON "a"."ACCOUNT_ID" = "b"."ACCOUNT_ID"

WHERE "a"."CLIENT_ID" LIKE 'BH%'

GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

) AS "a"

LEFT JOIN "SQUIDBITS"."PFS_GROUP"."ZIP_DETAILS" AS "b" ON "a"."zip_code" = LPAD("b"."ZIP_CODE",5,'0')

WHERE "a"."transaction_total" >= 0 /*AND "a"."assigned_date" < CURRENT_DATE()-140*/

The aforementioned SQL query was put into the cloud-based data warehouse platform Snowflake.   

When placed into Big Squid’s product, Kraken, this initial model was performing at an 82.8% accuracy and a .854 AUC.  

**Model to Eliminate Data Leakage**

After reviewing the first model created by Kyle Jourdan, Jordan Ganung and Haven Helm recognized some data leakage present in the first model. Data leakage is the machine learning concept that creates an overfit model. In other words, you use data used in the training and genesis of the model that otherwise wouldn’t be there when you’re applying the model to the real world.  

The business question that the first model, built by Kyle Jourdan, was intended to answer was “Based off of the time assignment, what is the clients propensity to pay in full in 120 days?”. Data leakage occurs when there is information that would not be known at the time the business question is actually asking that question. In this instance, nothing can be included in the input dataset that would only be known *after the time of assignment.*  

The columns that we had to take out of the model due to them causing data leakage are below:  

- If they paid in full by days 30, 60, 90, 120, 180, and 365
- Transaction total amounts at days 30, 60, 90,120,180, and 365


The new model was run with the following SQL query:  

SELECT  

​     "a"."account_id"

​    ,"a"."client_id"::VARCHAR AS "client_id"

​    ,"a"."patient_age"

​    ,"a"."zip_code"

​    ,COALESCE("b"."CITY_NAME",'UNKNOWN') AS "city_name"

​    ,COALESCE("b"."COUNTY_NAME",'UNKNOWN') AS "county_name"

​    ,COALESCE("b"."STATE_NAME",'UNKNOWN') AS "state_name"

​    ,COALESCE("b"."POPULATION",0) AS "population_estimate"

​    ,COALESCE("b"."DENSITY",0.00) AS "density_estimate"

​    ,COALESCE("b"."TIMEZONE",'UNKNOWN') AS "local_timezone"

​    ,"a"."financial_class"

​    ,"a"."patient_type"

​    ,"a"."med_service"

​    ,"a"."status_code"

​    ,"a"."previous_status_code"

​    ,"a"."employer_known"

​    ,"a"."is_employed"

​    ,"a"."assigned_date"

​    ,YEAR("a"."assigned_date") AS "assigned_year"

​    ,MONTH("a"."assigned_date") AS "assigned_month"

​    ,WEEK("a"."assigned_date") AS "assigned_week"

​    ,"a"."initial_balance"

​    ,"a"."cancelled_balance"

​    ,"a"."paid_in_30"

​    ,"a"."paid_in_60"

​    ,"a"."paid_in_90"

​    ,"a"."paid_in_120"

   -- ,"a"."paid_in_180"

   -- ,"a"."paid_in_365"

​    --,COALESCE("a"."transaction_total",0.00) AS "transaction_total"

​    --,"a"."transaction_count"

​    ,"a"."trans_in_30"

​    ,"a"."trans_in_60"

​    ,"a"."trans_in_90"

​    ,"a"."trans_in_120"

​    /*,CASE WHEN CURRENT_DATE() < DATEADD('day',121,"a"."assigned_date") AND ("a"."paid_in_120" < "a"."initial_balance") THEN NULL ELSE

​        CASE WHEN "a"."initial_balance" = 0 THEN 'false' ELSE

​            CASE WHEN ("a"."paid_in_120" / "a"."initial_balance") >= 0.9 THEN 'true' ELSE 'false' END

​        END

​     END AS "pif_in_120"*/

​     ,CASE WHEN CURRENT_DATE() < DATEADD('day',121,"a"."assigned_date") AND ("a"."paid_in_120" < "a"."initial_balance") THEN NULL ELSE

​        CASE WHEN "a"."initial_balance" = 0 THEN 'false' ELSE

​           -- CASE WHEN ("a"."paid_in_120" / "a"."initial_balance") >= 0.9 THEN 'true' ELSE 'false' END

​            CASE WHEN "a"."paid_in_120" >= "a"."initial_balance"  THEN 'true' ELSE 'false' END

​        END

​     END AS "pif_in_120"

FROM (

SELECT "a"."ACCOUNT_ID" AS "account_id"

​    ,"a"."CLIENT_ID" AS "client_id"

​    ,"a"."PATIENT_AGE" AS "patient_age"

​    ,LEFT(LPAD("a"."ZIP_CODE",5,'0'),5) AS "zip_code"

​    ,COALESCE(NULLIF(NULLIF("a"."FINANCIAL_CLASS",'NULL'),''),'UNKNOWN') AS "financial_class"

​    ,COALESCE(NULLIF(NULLIF("a"."PATIENT_TYPE",'NULL'),''),'UNKNOWN') AS "patient_type"

​    ,COALESCE(NULLIF(NULLIF("a"."MED_SERVICE",'NULL'),''),'UNKNOWN') AS "med_service"

​    ,"a"."STATUS_CODE" AS "status_code"

​    ,"a"."PREVIOUS_STATUS_CODE" AS "previous_status_code"

​    ,CASE WHEN "a"."EMPLOYER" IN ('NULL','UNKNOWN') THEN '0' ELSE '1' END AS "employer_known"

​    ,"a"."EMPLOYED" AS "is_employed"

​    ,"a"."ASSIGNED_DATE"::DATE AS "assigned_date"

​    ,"a"."LETTERS_SENT" AS "letter_sent"

​    ,"a"."INITIAL_BALANCE" AS "initial_balance"

​    ,"a"."CANCELLED_AMOUNT" AS "cancelled_balance"

​    --,COUNT(DISTINCT "b"."ID") AS "transaction_count"

​    ,COUNT(DISTINCT CASE WHEN "b"."TRANS_DATE"::DATE <= DATEADD('day',30,"a"."ASSIGNED_DATE"::DATE) THEN "b"."ID" END) AS "trans_in_30" -- Cant use as features

​    ,COUNT(DISTINCT CASE WHEN "b"."TRANS_DATE"::DATE <= DATEADD('day',60,"a"."ASSIGNED_DATE"::DATE) THEN "b"."ID" END) AS "trans_in_60" -- Cant use as features

​    ,COUNT(DISTINCT CASE WHEN "b"."TRANS_DATE"::DATE <= DATEADD('day',90,"a"."ASSIGNED_DATE"::DATE) THEN "b"."ID" END) AS "trans_in_90" -- Cant use as features

​    ,COUNT(DISTINCT CASE WHEN "b"."TRANS_DATE"::DATE <= DATEADD('day',120,"a"."ASSIGNED_DATE"::DATE) THEN "b"."ID" END) AS "trans_in_120" -- Cant use as features

​    ,SUM(CASE WHEN "b"."TRANS_DATE"::DATE <= DATEADD('day',30,"a"."ASSIGNED_DATE"::DATE) THEN "b"."AMOUNT" ELSE 0.00 END) AS "paid_in_30" -- Cant use as features

​    ,SUM(CASE WHEN "b"."TRANS_DATE"::DATE <= DATEADD('day',60,"a"."ASSIGNED_DATE"::DATE) THEN "b"."AMOUNT" ELSE 0.00 END) AS "paid_in_60" -- Cant use as features

​    ,SUM(CASE WHEN "b"."TRANS_DATE"::DATE <= DATEADD('day',90,"a"."ASSIGNED_DATE"::DATE) THEN "b"."AMOUNT" ELSE 0.00 END) AS "paid_in_90" -- Cant use as features

​    ,SUM(CASE WHEN "b"."TRANS_DATE"::DATE <= DATEADD('day',120,"a"."ASSIGNED_DATE"::DATE) THEN "b"."AMOUNT" ELSE 0.00 END) AS "paid_in_120" -- Cant use as features

​    --,SUM(CASE WHEN "b"."TRANS_DATE"::DATE <= DATEADD('day',180,"a"."ASSIGNED_DATE"::DATE) THEN "b"."AMOUNT" ELSE 0.00 END) AS "paid_in_180"

​    --,SUM(CASE WHEN "b"."TRANS_DATE"::DATE <= DATEADD('day',365,"a"."ASSIGNED_DATE"::DATE) THEN "b"."AMOUNT" ELSE 0.00 END) AS "paid_in_365"

​    --,COALESCE(SUM("b"."AMOUNT"),'0.00') AS "transaction_total"

FROM "SQUIDBITS"."PFS_GROUP"."ACCOUNTS" AS "a"

LEFT JOIN "SQUIDBITS"."PFS_GROUP"."TRANSACTION_RECORDS" AS "b" ON "a"."ACCOUNT_ID" = "b"."ACCOUNT_ID"

WHERE "a"."CLIENT_ID" LIKE 'BH%'

GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15) AS "a"

LEFT JOIN "SQUIDBITS"."PFS_GROUP"."ZIP_DETAILS" AS "b" ON "a"."zip_code" = LPAD("b"."ZIP_CODE",5,'0')

WHERE  

​    --"a"."transaction_total" >= 0 AND  

​    "a"."assigned_date" < CURRENT_DATE()-121

​    AND  "cancelled_balance" = 0

​    AND "initial_balance" >0

;

The table that is produced from the above SQL query should be the base table used in all remaining iterations of the model.  

When the output from the query above was pushed through Kraken, the Highest performing algorithm was a Random Forest Classifier. The F1 score was an 84%, recall and precision were 81% and 86% respectively and the AUC was .941.  

**Third Model for Prediction at 30 Days After Assignment**  

In order to ensure the highest predictive power and to create the most effective dial lists, it is suggested that Recess builds 4 models:

1. A prediction of who will PIF *at time of assignment*
2. A prediction *30 days after time of assignment* of who will PIF by 120 days
3. A prediction *60 days after time of assignment* of who will PIF by 120 days
4. A prediction *90 days after time of assignment* of who will PIF by 120 days

The latter three models will include behavioral data such as:  

-was the contact contacted through calls?  

-were those calls inbound or outbound?  

-did the contact answer the phone or did the dialer get a voicemail?

-were they sent mailers?

-how many payments were made by 30 days? (and 60 days and 90 days depending on the model)

-what was the paid amount by 30 days?(and 60 days and 90 days depending on the model)

### Software

Windows 10, Mac OS X, Linux ([Solus](https://github.com/solus-project/budgie-desktop))

Kraken (proprietary data science platform used by [BigSquid](https://bigsquid.com)

[Snowflake](https://snowflake.com)

[DataRobot](https://www.datarobot.com/)

Microsoft SQL Server Management Studio v. 12 ([SSMS](https://docs.microsoft.com/en-us/sql/ssms/download-sql-server-management-studio-ssms?view=sql-server-2017))

Python 3.6 (Packages: [Pandas](https://github.com/pandas-dev/pandas), [Sci-Kit Learn](https://scikit-learn.org/stable/user_guide.html), [Numpy](https://docs.scipy.org/doc/numpy/), [Matplotlib](https://matplotlib.org/), [datascience](http://data8.org/datascience/))

[R](https://www.r-project.org/) v. 3.5, RStudio (Packages: Tidyverse)

### Select Timelines

[February 11-18](https://github.com/RecessHealth/Documentation/blob/master/Feb_11-18_%20timeline)

### References

- J. Friedman, “Greedy Function Approximation: A Gradient Boosting Machine”,
  The Annals of Statistics, Vol. 29, No. 5, 2001.

- van de Greer, Ruben and Wang, Qingchen and Bhulai, Sandjai, "Data-Driven Consumer Debt Collection via Machine Learning and Approximate Dynamic Programming", September 17, 2018

- [Scikit-learn: Machine Learning in Python](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html), Pedregosa *et al.*, JMLR 12, pp. 2825-2830, 2011.

  

  
