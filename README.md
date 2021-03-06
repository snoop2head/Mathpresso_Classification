# MATHPRESSO_NLP_Project

π [YBIGTA-λ§€μ°νλ μ μ°ννλ ₯ μ΅μ’ λ°ν PDF](./YBIGTA_λ§€μ°νλ μ_λͺ½λ°μ΄ν¬_Final.pdf)

μν λ¬Έμ κ° μ£Όμ΄μ‘μ λ λ¬Έμ μ ν΄λΉ λ¨μμ μμΈ‘νλ Classification Taskμλλ€.

**Input Example.**

```
`x`μ λν μ΄μ°¨λ°©μ μ `x^2-6x+k+1=0`μ μ€κ·Όμ κ°κ³ , `x`μ λν μ΄μ°¨λ°©μ μ `kx^2-8x+5=0`μ μλ‘ λ€λ₯Έ λ νκ·Όμ κ°λλ‘ νλ μμ°μ `k`μ κ°μλ?
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
| 0.8598  | KoELECTRA-v3 | v1.2                  | Adam      | -                     | 50    | X          | 11/30 | μλ¨μ κΈ°μ€  |
| 0.8534  | KoELECTRA-v3 | v1.5                  | RMSProp   | Optuna                | 25    | O          | 12/04 | -            |
| 0.85    | KoBERT       | v1.2                  | Adam      | -                     | 15    | -          | -     | -            |
| 0.8483  | KoELECTRA-v3 | v1.2                  | Adam      | -                     | 25    | O          | 11/29 | μλ¨μ κΈ°μ€  |
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

- [κΉμ§ν](https://github.com/sopogen)
- [κΉν¬μ§](https://github.com/gimmizz)
- [β μμμ§](https://github.com/snoop2head)
- [μ κ±΄μ±](https://github.com/YooGunWook)
- [μ΅μ μ€](https://github.com/yuna-102)

## Feedbacks from Mathpresso Senior Engineers

- [π£ 1st Feedback](./Documents/1st_FEEDBACK.md)
- [π£ 2nd Feedback](./Documents/2nd_FEEDBACK.md)
