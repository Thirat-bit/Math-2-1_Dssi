# Week 11 — Logistic Regression (z, odds ratio, ความน่าจะเป็น, confusion matrix, threshold)

**ไฟล์โน้ตบุ๊ก:** [`week11_practice.ipynb`](../week11_practice.ipynb)  
**ข้อมูล:** `../data/week11/`  
**ไลบรารี:** numpy, pandas, scikit-learn, statsmodels

> สรุปประกอบการอ่าน — รายละเอียดของแต่ละข้ออยู่ในโน้ตบุ๊ก

## สูตรที่ใช้

- **โอกาส (logistic)** — P(y=1|x) = 1 / (1 + exp(−(β₀ + β₁x₁ + β₂x₂)))
- **z-statistic** — = β̂ / SE(β̂) → `model.tvalues['x']`
- **odds ratio** — = exp(β̂ · k)  โดย k = จำนวนหน่วยที่เพิ่มขึ้น
- **แก้สมการย้อนกลับ** — `x = (logit(P) − β₀ − β₂·x₂) / β₁` โดย logit(0.4) = ln(0.4/0.6) ≈ −0.4055
- **confusion matrix** — `confusion_matrix(y_true, y_pred)` — ลำดับ: TN, FP, FN, TP
- **F1 / recall** — F1 = 2·TP/(2·TP+FP+FN) · recall = TP/(TP+FN)
- **class_weight balanced** — w_c = n_samples / (n_classes × n_c)

## จุดที่คนตกบ่อย

- **ต้องใช้ MLE ไม่มี regularization** — ใช้ `smf.logit(...).fit(disp=0)` หรือ `sm.GLM` — ❌ `LogisticRegression()` ของ sklearn มี L2 (C=1) เป็นค่าเริ่มต้น → ค่าไม่ตรงเฉลย
- **`LogitResults` ไม่มี `predict_proba()`** — มีแค่ `predict()` — ถ้าต้องความน่าจะเป็นให้คำนวณเอง: `P = 1/(1 + np.exp(-(b0 + b1*x1 + b2*x2)))`
- **เกณฑ์จำแนกคลาสใช้ `≥ threshold`** — ไม่ใช่ `>`
- **น้ำหนัก class คำนวณจาก train** — ไม่ใช่จากทั้งชุด
- **แบ่ง train/test ตามตำแหน่งแถว** — 960 / 243 ห้ามสับ

## สรุปคำตอบทั้ง 10 ข้อ

| ข้อ | หัวข้อ | คำตอบ | ปัดเศษ | ความคลาดเคลื่อนที่เว็บยอมรับ |
|---|---|---|---|---|
| 11.1 | z-statistic ของ email_opens | **6.2677** | 4 ตำแหน่ง | 0.00015 |
| 11.2 | ทำไมถึงใช้ logistic แทน linear regression | **126** | จำนวนเต็ม | 0.5 |
| 11.3 | F1 score ที่ threshold = 0.25 | **0.5739** | 4 ตำแหน่ง | 0.00015 |
| 11.4 | Odds ratio เมื่ออีเมลที่เปิดเพิ่ม 3 ฉบับ | **1.6930** | 4 ตำแหน่ง | 0.00015 |
| 11.5 | แก้สมการหาว่าต้องเปิดอีเมลกี่ฉบับถึงจะมีโอกาสตอบรับ 0.4 | **7.30** | 2 ตำแหน่ง | 0.015 |
| 11.6 | จำนวน True Positive ที่ threshold = 0.4 | **25** | จำนวนเต็ม | 0.5 |
| 11.7 | จะโทรหาลูกค้ากี่คนที่ความน่าจะเป็นการเลิกซื้อ ≥ 0.12 | **88** | จำนวนเต็ม | 0.5 |
| 11.8 | ถ่วงน้ำหนัก class แบบ balanced แล้ว recall เปลี่ยนไหม | **0.5714** | 4 ตำแหน่ง | 0.00015 |
| 11.9 | ความน่าจะเป็นที่โมเดลทำนายของแถวที่ 562 | **0.1508** | 4 ตำแหน่ง | 0.00015 |
| 11.10 | หา threshold ที่ให้ F1 สูงสุด | **0.5852** | 4 ตำแหน่ง | 0.00015 |

---

ติดตั้งไลบรารี: `pip install -r requirements.txt`
