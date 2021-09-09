## 실험 단위

1. Validation
2. Preprocessing
   1. Noisy Data
   2. Hidden Data
3. 오류 분석: 한 두 단원 벗어난 경우가 아닌 경우
4. Hyperparameter Tuning
5. 소단원 vs 중단원 기준 분류
6. Data Augmentation

### 1. Validation

- update stratified random sampled validation dataset
  - validation 데이터셋 랜덤 추출에 따른 성능 변화 요소를 통제하기 위해서 이를 하나로 표준화시켰음 (random seed가 컴퓨터 별로 다르기 때문에 csv파일로 추출시켜놓음).
  - 소단원과 중단원 별로 training set과 validation set를 랜덤추출해서 만듦.

### 2. Preprocessing

#### Noisy Data Dashboard

1. KoELECTRA: 0.5682

- Preprocess: X
- Validaiton: O
- Epoch: 31/40
- 12/01

-> Preprocessing 1.5v을 noisy dataset에 적용하여 valid set 없이 KoELECTRA로 train한 결과입니다. 0.78 정도로 전처리 없을 때보다 무려 0.2 점 높아졌어요! 오히려 validation 없는 게 더 성능이 좋은 건가요? 0.57 -> 0.78로 오른 건 경이롭네요 ㅎㅎ

**To-do**

- 본래 Training Dataset에서 vocabulary set를 만들고, 일치하지 않는 애들은 [Levenstein Distance](https://lovit.github.io/nlp/2018/08/28/levenshtein_hangle/) 를 이용해서 일정 threshold 이하인 애들은 같은 단어라고 판단을 해서 치환하면 될 것 같음.

#### Hidden Data Dashboard

- Update preprocess_for_kobert v 1.2: Change arguments and remove spaces next to math symbols
  - Argument의 dafult 값을 (korean=True, space=True)로 설정하였습니다.
  - space=True로 설정해도 더 이상 기호 '+, -'의 양 옆에 공백이 추가되지 않습니다. 기존 sin, cos등의 영문 수학 용어의 옆에만 공백이 추가됩니다. 기호 양 옆에 공백을 추가한 것은 기존 word2vec등에서 토큰화를 할 때 기호와 영어가 붙어버려서 입니다. 하지만 KoELECTRA 등을 실행할 때 오히려 글자 간 공백이 너무 많아져 악영향을 줄 수 있다고 판단하여 삭제하였습니다.

여기까지가 제일 성능이 좋음

0 . KoELECTRA: 8769 (BEST)

- Preprocess: v1.2
- Validation: O
- Optimization: X
- Epoch: 25
- 11/29
- loss 기준
- 중단원 label 사용

1 . KoELECTRA: 0.8746 (BEST)

- Preprocess: X
- Validation: X
- Optimization: X
- Epoch: 50
- 11/25

2.  KoELECTRA: 0.8741

- Preprocess: v1.2
- Validation: X
- Optimization: X
- Epoch: 50
- 11/28

3. KoELECTRA: 0.8598

- Preprocess: v1.2
- Validation: X
- Optimization: X
- Epoch: 50
- 소단원 예측 사용
- 11/30

- Update preprocess version 1.3
  - (단, ) 으로 시작하는 조건 부분을 제거할 수 있도록 하였습니다. Argument 중 condition=True로 두면 해당 부분을 제거하고, False로 두면 제거하지 않습니다. (단, )으로 되어있는 부분은 대부분의 경우 "a와 b는 상수이다." 등의 문제 유형과는 관련 없는 조건들이 대부분입니다
  - 정규분포 ’N(0,1)’, 이항분포 'B(n,p)' 등 분포를 나타낼 때 쓰이는 알파벳과 괄호를 전부 삭제하였습니다. 괄호의 기능을 최대한 줄이기 위해 진행하였습니다. 해당 표현이 등장하는 모든 경우에 '정규분포', '이항분포' 등의 실제 용어가 등장하는 것을 확인하였고, 이에 해당 표현을 삭제하더라도 성능에 이상이 없을 것으로 판단하였습니다.
  - 수학 용어의 공백 기능을 강화하였습니다. 이제 이미 양 옆에 공백이 있는 경우에는 수학 용어 옆에 공백이 생기지 않습니다. 기존 학습 데이터에서는 수학 용어가 띄어져 있는 경우도 있고, 붙여져 있는 경우도 있었습니다. 따라서 이미 띄워져 있는 경우에 전처리 module을 돌리게 되면, 공백이 양 옆에 추가되어 쓸데 없이 두 번 이상 띄어지는 경우가 잦게 되었습니다. 이를 해결하였습니다.
  - **KoBERT 0.84로 성능 하락**
- Update preprocess_for_kobert v 1.4
  - ㄱ. ㄴ. …, (가), (나), … 등 보기 삭제
  - Noisy dataset을 위한 전처리 함수 preprocess_noisy를 추가하였습니다. 해당 함수를 실행하면 noisy dataset의 Latex 문법을 원래 dataset과 동일하게 바꿔주고, 이어서 원래 preprocess까지 진행합니다. argument 또한 원래 prerpocess 함수와 동일하게 사용할 수 있습니다.
- Update preprocess_for_kobert v 1.5
  - pipe 기호 (|) 구분 및 전처리
  - **KoBERT 0.83으로 성능하락**
  - **KoELECTRA valid set 없이 돌린 일반 test data는 현재까지 preprocess 다 적용, epoch 40으로 하였을 때 0.857** 나오네요 ㅠ 네 그러니깐요.. noise 없는 데이터셋은 전처리로 성능 올리기엔 이제 한계가 있는 것 같아요

### 3. 오류 분석: 한 두 단원 벗어난 경우가 아닌 경우

**두 가지 경우를 중점적으로 봤습니다.**

- **대단원 기준으로 분류가 아예 틀린 경우**
- **중단원 기준으로 ±3단원 이상 벗어난 경우**
- Qtid: H1S1-07 vs Predicted: HSTA-01이 있다면 네 글자가 달라서 4가 출력되게 하는 간단한 방법입니다! Letter 단위에요 ㅎㅎ

**H1S2-07 vs HSTA**

- 전체 오류 112개 중에 12개, 즉 10%나 차지하는 걸 찾았습니다.
  경우의 수, 방법의 수, 조합 및 순열에 해당하는 단원 H1S2-07입니다.
  실제로는 HSTA(확률과 통계) 대단원에 속하는 문제인데, KoElectra가 H1S2-07으로 잘못 분류한 경우가 제일 많았습니다.
  근데 애초에 매스프레소가 “경우의 수, 방법의 수, 조합 및 순열” 내용을 어떤 이유로 확률과 통계 단원으로 종속을 안 시킨 건지 궁금하네요 ㅋㅋㅋ
  [image:8CC1AC43-B843-4790-A402-90F8B7A94B58-1029-000001295EBEAC99/image.png]

**색칠하는 경우의 수**
[image:A51C26D9-57A7-4C3C-8B22-9BE7FFC0738D-1029-0000013D7B401A6B/image.png]
[image:A24CF598-BA4D-41B5-AF06-3DABC0439FD8-1029-0000013467143F39/image.png]

'색칠하는 경우의 수(HSTA-01-02)' 문제와 '색칠하는 방법의 수(H1S2-07-08)' 문제를 살펴보았는데… 아무리 봐도 구분할 방법이 없는 것 같네요. 기존에 컴퓨터의 방침 대로 그나마 더 많은 H1S2-07로 labeling 하는 게 맞는 것 같습니다.

- HSTA-04, HSTA-03 단원 간에 제대로 구분하지 못하는 경우를 보다가, **’시행과 사건’**문제에서 **true(HSTA-03) -> predict(HSTA-04 or HSTA-05)**으로 잘못 예측하는 것을 발견했습니다! 혹시 개선할 여지가 있을지, 의논하면 좋을 것 같습니다.

### 4. Hyperparameter Tuning

- KoElectra hyperparameter tunning
  - 현재 KoElectra 하이퍼파라미터 튜닝 돌리는 중입니다! 파일은 [KoElectra_yuna_hyperparameter_tunning.ipynb](https://github.com/YooGunWook/MATHPRESSO_NLP_Project/blob/working/KoElectra_yuna_hyperparameter_tunning.ipynb) 에서 확인하실 수 있습니다. valid_f1이랑 valid_acc 각각 최적화 기준 두고 돌려보고 있는데 시간은 좀 걸릴 것 같습니다.
  - 사용한 framework는 Optuna인데 이걸 선택한 이유는 솔직하게 제가 딥러닝 하이퍼 파라미터 튜닝을 잘 몰라서 사용하기 쉬워서, 최신이어서(2019년) 사용했습니다 ㅎㅎ 그래서 구조 자체는 이해를 못했는데 기본적으로 Hyperopt과 유사하게 Bayesian Optimization TPE를 기반으로 하는 것 같습니다.

### 5. 소단원 vs 중단원 기준

- Run KoELECTRA model with small category
  - Validation O
  - 0.8483으로 하락
  - preprocess 1.2, drop_noise 적용, valid_f1 = 0.8483

### 6. Data Augmentation

- Make Augmentation Module
  - Random Deletion, Random Switch
  - 벤치마크가 0.85였는데, 제출점수 KoBERT 0.83으로 성능 하락
