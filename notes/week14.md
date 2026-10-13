# Week 14 — Cross-validation (validation set, LOOCV, K-fold, Stratified K-fold, one-standard-error)

**ไฟล์โน้ตบุ๊ก:** [`week14_practice.ipynb`](../week14_practice.ipynb)  
**ข้อมูล:** `../data/week14/`  
**ไลบรารี:** numpy, pandas, scikit-learn

> สรุปประกอบการอ่าน — รายละเอียดของแต่ละข้ออยู่ในโน้ตบุ๊ก

## สูตรที่ใช้

- **KFold(n_splits=K)** — แบ่งตามลำดับแถว (ไม่สับ)
- **StratifiedKFold** — รักษาสัดส่วนคลาสใกล้เคียงกัน → ต้องส่ง `split(X, y)`
- **LeaveOneOut()** — ตัดออกทีละ 1 คน
- **CV MSE** — = ค่าเฉลี่ยของ MSE ทุก fold (**ทุก fold น้ำหนักเท่ากัน**)
- **one-standard-error rule** — SE = SD(ddof=1) ของ 5 fold ÷ √5 แล้วเลือกดีกรี**ต่ำที่สุด** ที่ CV MSE ไม่เกิน (ต่ำสุด + SE)
- **Generalization gap** — = ค่าเฉลี่ย validation MSE − ค่าเฉลี่ย train MSE

## จุดที่คนตกบ่อย

- **ห้ามสับข้อมูลเด็ดขาด** — ห้ามใส่ `shuffle`, `random_state` และห้ามใช้ `train_test_split`
- **standardize ใหม่ในทุก fold โดยใช้เฉพาะ train fold** — ใช้ `make_pipeline(StandardScaler(), โมเดล())` — ถ้า scale ทั้งชุด = ข้อมูลรั่วจาก validation เข้าไปในการ fit
- **KNN classifier ต้องใช้ K เป็นเลขคี่** — เพราะผลทำนายคือเสียงข้างมาก
- **KFold(7) แบ่งเป็น 7 ก้อนเรียงตามลำดับแถว** — fold ลำดับที่ 7 คือก้อนสุดท้าย → `folds[6]`
- **“ค่า CV” ไม่ถ่วงตามขนาด fold** — ทุก fold น้ำหนักเท่ากัน

## สรุปคำตอบทั้ง 10 ข้อ

| ข้อ | หัวข้อ | คำตอบ | ปัดเศษ | ความคลาดเคลื่อนที่เว็บยอมรับ |
|---|---|---|---|---|
| 14.1 | Stratified 5-Fold CV: ค่าเฉลี่ย F1 ของ KNN (K=11) | **0.2545** | 4 ตำแหน่ง | 0.00015 |
| 14.2 | KFold(7) ที่ไม่สับ: ค่าเฉลี่ย income_k ของ validation fold ลำดับที่ 7 | **28.6515** | 4 ตำแหน่ง | 0.00015 |
| 14.3 | KFold(5) กับ Linear Regression: ค่า CV MSE | **2.3968** | 4 ตำแหน่ง | 0.00015 |
| 14.4 | Validation set approach: เลขคู่ = train, เลขคี่ = validation | **0.6928** | 4 ตำแหน่ง | 0.00015 |
| 14.5 | Leave-One-Out CV กับ KNN (K=9) บน 150 คนแรก | **0.2867** | 4 ตำแหน่ง | 0.00015 |
| 14.6 | one-standard-error rule | **2.4376** | 4 ตำแหน่ง | 0.00015 |
| 14.7 | ช่องว่างระหว่าง validation MSE กับ train MSE | **0.0117** | 4 ตำแหน่ง | 0.00015 |
| 14.8 | Stratified 5-Fold CV ของ LDA (ทำนายการเลิกซื้อ) | **0.6795** | 4 ตำแหน่ง | 0.00015 |
| 14.9 | KFold(5) กับ KNN Regression (K=15) | **0.6570** | 4 ตำแหน่ง | 0.00015 |
| 14.10 | เทียบ CV MSE กับ Test MSE ของการแบ่งครั้งเดียว | **0.0171** | 4 ตำแหน่ง | 0.00015 |

---

ติดตั้งไลบรารี: `pip install -r requirements.txt`
