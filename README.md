# MATHPRESSO_NLP_Project

👉 [YBIGTA-매쓰프레소 산학협력 최종 발표 PDF](./YBIGTA_매쓰프레소_몽데이크_Final.pdf)

수학 문제가 주어졌을 때 문제의 해당 단원을 예측하는 Classification Task입니다.

**Input Example.**

```
`x`에 대한 이차방정식 `x^2-6x+k+1=0`은 실근을 갖고, `x`에 대한 이차방정식 `kx^2-8x+5=0`은 서로 다른 두 허근을 갖도록 하는 자연수 `k`의 개수는?
```

**Output Example.**

`H1S1-05`

---

## Experiments Table

### Cleansed Dataset

| Test F1 | Model        | Preprocessing Version | Optimizer | Hyperparameter Tuning | Epoch | Validation | Date  | ETC          |
| ------- | ------------ | --------------------- | --------- | --------------------- | ----- | ---------- | ----- | ------------ |
| 0.8925  | KoELECTRA-v3 | v1.5                  | AdamW     | Bayesian-opt          | 17    | X          | 12/04 | -            |
| 0.8849  | KoELECTRA-v3 | -                     | AdamW     | Bayesian-opt          | 17    | X          | 12/04 | -            |
| 0.8769  | KoELECTRA-v3 | v1.2                  | Adam      | -                     | 25    | O          | 11/29 | -            |
| 0.8746  | KoELECTRA-v3 | v1.1                  | Adam      | -                     | 50    | X          | 11/25 | -            |
| 0.8741  | KoELECTRA-v3 | v1.1                  | Adam      | -                     | 50    | X          | 11/28 | -            |
| 0.8676  | KoELECTRA-v3 | v1.6                  | RAdam     | -                     | 25    | O          | 12/04 |              |
| 0.8628  | KoELECTRA-v3 | v1.2                  | Adam      | -                     | 21    | O          | 12/1  | -            |
| 0.8598  | KoELECTRA-v3 | v1.2                  | Adam      | -                     | 50    | X          | 11/30 | 소단원 기준  |
| 0.8534  | KoELECTRA-v3 | v1.5                  | RMSProp   | Optuna                | 25    | O          | 12/04 | -            |
| 0.85    | KoBERT       | v1.2                  | Adam      | -                     | 15    | -          | -     | -            |
| 0.8483  | KoELECTRA-v3 | v1.2                  | Adam      | -                     | 25    | O          | 11/29 | 소단원 기준  |
| 0.84    | KoBERT       | v1.3                  | Adam      | -                     | 15    | O          | 12/03 | -            |
| 0.83    | KoBERT       | v1.3                  | Adam      | -                     | 22    | O          | 12/02 | Augmentation |
| 0.83    | KoBERT       | v1.5                  | Adam      | -                     | 15    | O          | 12/02 | -            |

### Noisy Dataset from OCR

| Test F1 | Model     | Preprocessing Version | Optimizer | Hyperparameter Tuning | Epoch | Validation | Date  | ETC |
| ------- | --------- | --------------------- | --------- | --------------------- | ----- | ---------- | ----- | --- |
| 0.7792  | KoELECTRA | v1.4                  | AdamW     | -                     | 40    | -          | 12/05 |     |
| 0.7787  | KoELECTRA | v1.4                  | Adam      | -                     | 40    | -          | 12/04 | -   |
| 0.7571  | KoELECTRA | v1.4                  | AdamW     | Bayesian-opt          | 17    | -          | 12/04 |     |
| 0.7364  | KoELECTRA | v1.6                  | RAdam     | -                     | 25    | O          | 12/04 |     |
| 0.7349  | KoELECTRA | v1.6                  | AdamW     | Bayesian-opt          | 17    | -          | 12/04 | -   |
| 0.6956  | KoELECTRA | v1.6                  | AdamW     | Bayesian-opt          | 17    | -          | 12/04 | -   |
| 0.5682  | KoELECTRA | -                     | Adam      | -                     | 31    | O          | 12/01 | -   |

## Participants

- [김지환](https://github.com/sopogen)
- [김희진](https://github.com/gimmizz)
- [✋ 안영진](https://github.com/snoop2head)
- [유건욱](https://github.com/YooGunWook)
- [최정윤](https://github.com/yuna-102)

## Feedbacks from Mathpresso Senior Engineers

- [📣 1st Feedback](./Documents/1st_FEEDBACK.md)
- [📣 2nd Feedback](./Documents/2nd_FEEDBACK.md)
