# ETRI_ML

**requirements**
- scikit-learn=1.2.2
- numpy=1.26.4
- pandas=2.1.4
- scipy=1.11.4
- statsnmodels=0.14.0

---
**codes**
* feature_extraction.py : 데이터로 부터 feature 추출하는 코드

* feature_selection.py : p_value가 높은 feature를 기준으로 AdaBooostClassifer, ClassiferChain을 통해 모델 훈련하는 코드

* fe_test.py : test dataset에서 feature extraction 진행하는 코드

* utils.py : seed 고정 및 score 계산을 위한 function
