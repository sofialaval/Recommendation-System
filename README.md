## Project Overview 
Ecolabels play a pivotal role in guiding consumers towards sustainable purchasing decisions amid the increasing importance of environmental, social, and governance (ESG) factors. The current manual process of certifying products for certifications is both time consuming and error-prone, posing risks of incorrect certifications and product recalls. We proposed an innovative approach leveraging Large Language Models (LLMs) to streamline and enhance the certification process. We will compare the LLM assessments with traditional Machine Learning (ML) classification algorithms to evaluate the efficacy of these
models as an Ecolabel recommendation engine.

Checkout the deployed recommendation system [here](https://esg-certification-recommender-system.streamlit).

## Methodology 
Given IT product and certification data, we utilized Natural Language Processing (NLP) methods to assess whether a product meets the criteria for an ESG certification

<p align="center">
  <img src=https://github.com/sofialaval/Kaggle_Competition-Prediction_of_Obesity_Risk/assets/159965979/12676a6c-2898-428c-940c-d502ee74d0d7>
</p>


A more detailed outline is shown below:     

<p align="center">
  <img src=https://github.com/sofialaval/Kaggle_Competition-Prediction_of_Obesity_Risk/assets/159965979/77b5ea03-db79-4eec-ab6f-b93777dce48a>
</p>

## Performance Results 
Testing results obtained by traditional ML models and LLMs: 

Six machine learning approaches were applied to the data. We measured their respective effectiveness in predicting whether a given product was Energy Star or TCO certified, allowing for prediction of outcomes with future products. The code can be found [here](https://github.com/sofialaval/Recommendation-System/blob/main/Models.ipynb). Below are the six models’ performance on the respective 500-sample testing sets for Energy Star: 

<p align="center">
  <img src=https://github.com/sofialaval/Recommendation-System/assets/159965979/e7a90ccb-3020-4d90-8feb-db2d166208f8>
</p>

<p align="center">
  <img src=https://github.com/sofialaval/Recommendation-System/assets/159965979/aa8bac18-1925-4641-89d3-f94b62276a82>
</p>

and GPT-3.5 on the same data set: 

<p align="center">
  <img src=https://github.com/sofialaval/Recommendation-System/assets/159965979/03f8e73a-cafc-4551-904b-1eded76434ec>
</p>



Further analysis is needed to understand whether the observed exceptional performance of the ML models can be generalized to other products, or if it is a result of our specific methodology and dataset. Continued approaches could include alternative forms of imputation, cross-validation, variable selection, more balanced data, integration of additional data resources, up to date status on current product Ecolabel certifications, and other machine learning algorithms. Many ML algorithms lack clarity around the reason for classification and often require extensive data cleaning. 

Even though the performance was not as high for the LLM, they demonstratetheir capability to calculate metrics accurately and provide transparent conclusions. For example, consider the following Energy Star mandate with an IT product "E15": 

<p align="center">
  <img src=https://github.com/sofialaval/Recommendation-System/assets/159965979/fc26ba42-fda1-41f5-8cde-b616316e35f3>
</p>

<p align="center">
  <img src=https://github.com/sofialaval/Recommendation-System/assets/159965979/3b629e06-2747-4682-a5cc-de27ae1c58b8>
</p>

As expected, GPT is not able to come to any definitive assessment for this mandate. To improve the effectiveness of the LLM assessments, the user should be able to provide additional data and information about the products as necessary. Another improvement to the LLM process would be using “few shot learning,” a method where model accuracy improves by including a small number of examples per class. In our process, we queried the LLMs using product data without providing examples of compliant products with which to compare. This is called “zero shot learning” and can cause the uncertainty observed in the LLM responses. With these improvements implemented, we believe that LLMs can be an excellent tool for aiding in the ESG
certification process. It should be noted that while the ML algorithms performed incredibly well on the product dataset, most enterprises do not have the luxury of 100k+ labeled datapoints to train.


Refer to the [powerpoint](https://github.com/sofialaval/Recommendation-System/blob/main/Presentation_ESG_Recommendation_Engine_OMSA_Team2.pdf) and [report](https://github.com/sofialaval/Recommendation-System/blob/main/Report_ESG_Recommendation_Engine_OMSA_Team2.pdf) for more detailed information. 

