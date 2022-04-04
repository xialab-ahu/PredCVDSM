#PredCVDSM
PredCVDSM is a disease-specific prediction method to identify synonymous mutations that cause cardiovascular disease.
<br />
#Abstract
Cardiovascular disease has long been an important factor affecting human life and health, seriously threatening the 
safety of human life.With the deepening of research, it is found that synonymous mutations play an important role in 
the occurrence and development of cardiovascular diseases through different mechanisms, so the research on cardiovascular
pathogenic synonymous mutations is very meaningful and necessary. Existing methods are broad-spectrum methods for the 
prediction of cardiovascular pathogenic synonymous mutations, which are not conducive to accurate prediction of 
cardiovascular pathogenic synonymous mutations. In addition, no specific method has been developed for cardiovascular
disease-causing synonymous mutation prediction, so the development of such specific methods is necessary. In this dissertation,
we present for the first time a machine learning-based method for predicting cardiovascular pathogenic synonymous mutations, 
named PredCVDSM. The method quantifies the data using 17-dimensional features related to splicing, sequence, and functional
score, while selecting the optimal RF classifier to build a predictive model. Compared with other related prediction 
methods on the independent test set, the results show that our method can more effectively identify cardiovascular 
pathogenic synonymous mutations, and the performance is better than other prediction methods.

#Installation
*numpy==1.20.1
*matplotlib==3.3.4
*mlxtend==0.18.0
*pandas==1.2.2
*joblib==1.0.1
*xgboost==1.3.3
*scikit_learn==0.24.1
*torchvision==0.9.1
