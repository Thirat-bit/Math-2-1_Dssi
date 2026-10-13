# Math for Data Science — โน้ตบุ๊กฝึกหัด Week 1–14

วิชา **1145201 คณิตศาสตร์สำหรับวิทยาศาสตร์ข้อมูล** — โน้ตบุ๊กแบบฝึกหัดครบ **14 สัปดาห์ / 140 ข้อ**
ทุกข้อทำคะแนนได้ **100%** (ตรวจกับเฉลยในหน้าเว็บทางการแล้วทุกข้อ)

| | |
|---|---|
| โน้ตบุ๊ก | `week1_practice.ipynb` – `week14_practice.ipynb` |
| ข้อมูล | `data/week01/` – `data/week14/` |
| สรุปประกอบการอ่าน | `notes/week1.md` – `notes/week14.md` |
| เว็บออฟไลน์ | `web/index.html` |
| ไลบรารี | `requirements.txt` |
| แหล่งที่มา | [`link.md`](link.md) |

---

## ติดตั้ง

ต้องใช้ **Python 3.13**

```bash
pip install -r requirements.txt
```

ไลบรารีที่ใช้ (5 ตัว)

```text
numpy==2.4.6
pandas==3.0.5
scipy==1.17.1
scikit-learn==1.9.0
statsmodels==0.15.0
ipykernel==7.3.0
```

> ⚠️ **scikit-learn ต้องเป็น `1.9.0` เท่านั้น**
> ทดสอบแล้ว `1.9.1` ทำให้ข้อ 12.3 ได้ `0.2737` (เฉลย `0.2735`) และข้อ 12.10 ได้ `0.6910` (เฉลย `0.6907`)
> ซึ่งเกินความคลาดเคลื่อนที่เว็บยอมรับ
> ส่วน numpy / pandas / scipy ที่ใหม่กว่านี้ได้ผลเหมือนเดิมทุกข้อ

## เปิดใช้งาน

**JupyterLab / VS Code**

```bash
pip install jupyterlab
jupyter lab
```

เปิดไฟล์ `.ipynb` แล้วเลือก kernel **`Python 3.13 (ipykernel)`**

**รันทั้งเล่มจากบรรทัดคำสั่ง**

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python313 week1_practice.ipynb
```

---

## โครงสร้างโปรเจกต์

```text
├── week1_practice.ipynb  –  week14_practice.ipynb   โน้ตบุ๊ก 14 เล่ม (เล่มละ 24 cells)
├── notes/                                          สรุปประกอบการอ่าน 14 ไฟล์
│   └── week1.md  –  week14.md
├── data/
│   ├── week01/  –  week14/                         ข้อมูลแยกตามสัปดาห์
│   └── ...                                            ไฟล์ตรงกับต้นฉบับทุกไบต์
├── web/                                            เว็บสำเนาสำหรับเปิดออฟไลน์
│   ├── index.html                                   เปิดไฟล์นี้ได้เลย
│   └── data/                                        ไฟล์ CSV ทั้ง 13 ไฟล์
├── requirements.txt
└── link.md
```

### โครงสร้างภายในโน้ตบุ๊ก (เหมือนกันทุกเล่ม)

| ลำดับ | ประเภท | เนื้อหา |
|---|---|---|
| 1 | markdown | กติกาและจุดที่ต้องระวังของสัปดาห์นั้น |
| 2 | code | ดาวน์โหลดไฟล์ข้อมูล (ถ้ายังไม่มี) |
| 3 | code | ตรวจไฟล์และเวอร์ชันเครื่องมือ |
| 4–23 | markdown + code | คำอธิบายและโค้ดของข้อที่ 1–10 (สลับกัน) |
| 24 | markdown | ตารางสรุปคำตอบทั้ง 10 ข้อ |

> **ทุก cell ของข้อเป็น self-contained** — มี `import` และโหลดไฟล์ของตัวเองครบ
> คัดลอกไปรันเดี่ยว ๆ ได้โดยไม่ต้องรัน cell ก่อนหน้า

---

## เนื้อหาแต่ละสัปดาห์

| สัปดาห์ | หัวข้อ | ไลบรารีที่ใช้ |
|---|---|---|
| [1](notes/week1.md) | เวกเตอร์ NumPy — norm, cosine similarity, ระยะ Euclid | numpy, pandas |
| [2](notes/week2.md) | เมทริกซ์ ระบบสมการ `Ax = b` และ determinant | numpy, pandas |
| [3](notes/week3.md) | Rank, Null space, Projection, Least Squares, Condition number | numpy, pandas |
| [4](notes/week4.md) | Eigenvalue, SVD, Low-rank approximation, PCA | numpy, pandas |
| [5](notes/week5.md) | Statistical Learning — Linear vs KNN, Test MSE | numpy, pandas, scikit-learn |
| [6](notes/week6.md) | Bias-Variance Trade-Off | numpy, pandas, scikit-learn |
| [7](notes/week7.md) | EDA, Bootstrap CI, t-test, Chi-square | numpy, pandas, scipy |
| [8](notes/week8.md) | Simple Linear Regression — SE, RSE, t, p-value | numpy, pandas, scipy |
| [9](notes/week9.md) | Multiple Linear Regression — Adjusted R², AIC/BIC | numpy, pandas, statsmodels |
| [10](notes/week10.md) | Dummy, Interaction, Polynomial, Diagnostics, VIF | numpy, pandas, statsmodels |
| [11](notes/week11.md) | Logistic Regression — z, odds ratio, confusion matrix | numpy, pandas, scikit-learn, statsmodels |
| [12](notes/week12.md) | LDA, QDA, Gaussian Naive Bayes, KNN | numpy, pandas, scikit-learn |
| [13](notes/week13.md) | GLM (Poisson), ROC/AUC, class imbalance | numpy, pandas, scikit-learn, statsmodels |
| [14](notes/week14.md) | Cross-validation — K-fold, LOOCV, one-standard-error | numpy, pandas, scikit-learn |

---

## จุดที่คนพลาดบ่อย

| เรื่อง | สรุป |
|---|---|
| `LogisticRegression` ของ sklearn | มี **L2 regularization** เป็นค่าเริ่มต้น → โจทย์ที่บอก "MLE ไม่มี regularization" ต้องใช้ `smf.logit` |
| `LogitResults` ของ statsmodels | **ไม่มี `predict_proba()`** มีแค่ `predict()` — ต้องคำนวณความน่าจะเป็นเอง |
| `predict_proba()` | ต้องดึงคอลัมน์ที่ 1 → `predict_proba(X)[:, 1]` เสมอ |
| `roc_curve` | คืน threshold **จากมากไปน้อย** (ตัวแรกคือ `inf`) → "สูงสุดที่ยังผ่าน" = ดัชนีแรกที่ผ่าน |
| Cross-validation | **ห้ามสับข้อมูล** — ห้ามใส่ `shuffle` / `random_state` และห้ามใช้ `train_test_split` |
| standardize ใน CV | ต้อง fit กับ **train fold** เท่านั้น → ใช้ `make_pipeline(StandardScaler(), โมเดล())` |
| `PolynomialFeatures(d)` | ให้คอลัมน์ `d+1` ตัว → สัมประสิทธิ์ของ `xᵈ` อยู่ที่ `coef_[d]` ไม่ใช่ `coef_[d-1]` |
| ddof | `np.std` ใช้ `ddof=0` แต่ `pandas .std()` ใช้ `ddof=1` ส่วน `np.cov` ใช้ `n−1` |
| Interaction term | ความชันของกลุ่ม = `params['x']` **+** `params['x:C(g)[T.กลุ่ม]']` |
| สูตร polynomial ใน statsmodels | ต้องเขียน `I(age**2)` ไม่ใช่ `age**2` ตรง ๆ |

รายละเอียดเพิ่มเติมของแต่ละสัปดาห์อยู่ใน [`notes/`](notes/)

---

## เว็บสำหรับส่งคำตอบ

เว็บต้นทาง: <https://popotaroman-cloud.github.io/mathsForDSUBU/practice/practice_all.html>

ถ้าไม่มีเน็ต เปิด [`web/index.html`](web/index.html) ได้เลย
เป็นสำเนาแบบ byte-identical ฝังข้อมูลและเฉลยครบในไฟล์เดียว ไม่ต้องต่อเน็ต
แต่ละข้อมีปุ่ม **ส่ง** (ตรวจเฉลย) และ **ดูเฉลย**

---

## ผลการตรวจสอบ

```text
คำตอบ              140/140 ตรงเฉลย (ทุกข้ออยู่ใน tolerance ที่เว็บยอมรับ)
execution error    0 ทุกเล่ม
self-contained     ทุก cell ข้อมี import และโหลดข้อมูลเอง
โครงสร้าง          14 เล่ม × 24 cells
kernel             python313 ทั้งหมด
ไฟล์ข้อมูล        data/weekNN/ ตรงกับไฟล์ต้นฉบับทุกไบต์
```

วิธีตรวจของเว็บ: `|ค่าที่กรอก − เฉลย| ≤ ความคลาดเคลื่อนที่เว็บยอมรับ`

---

## หมายเหตุ

- ไฟล์ข้อมูลถูกดาวน์โหลดอัตโนมัติตอนรัน cell ที่ 2 ถ้ายังไม่มีใน `data/weekNN/`
  ต้องต่อเน็ตครั้งแรกเท่านั้น หลังจากนั้นรันออฟไลน์ได้
- ข้อมูลใน `web/data/` คือไฟล์ CSV ดิบ 13 ไฟล์ (แตกออกจาก base64 ที่ฝังในหน้าเว็บ)