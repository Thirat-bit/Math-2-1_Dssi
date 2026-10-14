# คู่มือสูตร + keyword Week 1–14

ค้นหาสูตรที่ต้องใช้ → ดูโค้ดตัวอย่าง → ดูว่าใช้จริงที่ข้อไหน

- เรียง**ตามหัวข้อ** ไม่ใช่ตามสัปดาห์ เพราะตอนสอบจำไม่ได้ว่า "ข้อ 3.6 อยู่ week ไหน"
  แต่จำได้ว่า "สัมประสิทธิ์ QR หายังไง"
- ตัวอย่างโค้ดทุกชิ้น**ตัดมาจากโน้ตบุ๊กจริง** ที่รันได้และได้คะแนนเต็ม
- ส่วน **กรณีผิด** คือความผิดพลาดที่เจอจริงตอนทำ 140 ข้อ

## สารบัญ

| หมวด | เนื้อหา | สัปดาห์ |
|---|---|---|
| [A](#a-คณิตศาสตร์-week-1-4) | เวกเตอร์ · เมทริกซ์ · rank · QR · SVD · PCA | 1–4 |
| [B](#b-สถิติศาสตร์-week-5-10) | Regression · t-test · chi-square · diagnostics | 5–10 |
| [C](#c-การจำแนกโญกลุ่ม-week-11-13) | Logistic · LDA/QDA · KNN · Poisson · ROC | 11–13 |
| [D](#d-การประเมินโมเดล-week-14) | K-fold · LOOCV · one-standard-error | 14 |
| [ดัชนี](#ดัชนี-keyword-ข้อ-week) | ตารางค้น keyword ทั้ง 71 ตัว | ทั้งหมด |

---
---

# A คณิตศาสตร์ Week 1-4

> ⚠️ สัปดาห์ 1–4 **ห้าม standardize** ใช้ค่าตามหน่วยในไฟล์ตรง ๆ

## A1 L2 norm ความยาวเวกเตอร์

**สูตร**  ‖x‖ = √(x₁² + x₂² + … + xₙ²)

**keyword**  `np.linalg.norm(x)`

**ตัวอย่างโค้ด**
```python
cols = ['tenure_months', 'web_visits', 'spend_k', 'orders_90d']
x = df.iloc[881][cols].to_numpy(dtype=float)   # เอาแถวเป็นเวกเตอร์
answer = np.linalg.norm(x)
```

**ทำไม**  คืนรากผลบวกกำลังสองของสมาชิกทุกตัว เท่ากับ `np.sqrt((x*x).sum())`

**ใช้ที่**  ข้อ 1.1 · ถ้าต้อง Frobenius norm ของเมทริกซ์ให้ใส่ `axis=None` ดู [A20](#a20-frobenius-norm-กับการตัด-rank)

---

## A2 cosine similarity

**สูตร**  cos(s, p) = (s·p) / (‖s‖‖p‖)

**keyword**  `s @ p / (norm(s) * norm(p))`

**ตัวอย่างโค้ด**
```python
cols = ['income_k', 'email_opens', 'basket_size', 'spend_k']
p = np.array([66, 9, 7, 8], dtype=float)                        # ลูกค้าต้นแบบ
s = df.iloc[898][cols].to_numpy(float) + df.iloc[1035][cols].to_numpy(float)   # ผลรวมทีละสมาชิก
answer = float(s @ p / (np.linalg.norm(s) * np.linalg.norm(p)))
```

**ทำไม**  วัดมุมระหว่างสองเวกเตอร์ ไม่สนขนาด — ใช้เป็น "ความคล้ายเชิงทิศทาง"
อยู่ในช่วง −1 ถึง 1

**ใช้ที่**  ข้อ 1.2, 1.4, 1.7, 1.8, 1.10

**กรณีผิด** ถ้าคูณด้วยค่าเฉลี่ยแทนที่จะบวกทีละสมาชิกจะได้คนละคำตอบ (ข้อ 1.2)

---

## A3 dot product

**สูตร**  s·p = Σ sᵢpᵢ

**keyword**  `s @ p` หรือ `np.dot(s, p)`

**ตัวอย่างโค้ด**
```python
cols = ['income_k', 'tenure_months', 'email_opens']
w = df[cols].to_numpy(float) @ np.array([0.4, 0.2, 0.3])   # คะแนนทุกแถวพร้อมกัน
top3 = float(np.sort(w)[-3:].mean())                        # เฉลี่ยของ 3 อันดับแรก
```

**ทำไม**  ถ้าสัมประสิทธิ์ของเวกเตอร์น้ำหนักเป็นบวกหมด ค่าที่สูงที่สุดคือแถวที่ "ครบทุกด้านที่สุด"

**ใช้ที่**  ข้อ 1.6, 1.7

---

## A4 มุมระหว่างเวกเตอร์

**สูตร**  θ = arccos(cos) → แปลงเป็นองศา

**keyword**  `np.degrees(np.arccos(c))`

**ตัวอย่างโค้ด**
```python
cols = ['age', 'income_k', 'email_opens', 'spend_k']
a = df.iloc[1059][cols].to_numpy(float)
b = df.iloc[509][cols].to_numpy(float)
cosab = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
answer = float(np.degrees(np.arccos(cosab)))
```

**ทำไม**  `arccos` คืน **เรเดียน** โจทย์ถามเป็น **องศา** ต้องคูณ 180/π

**ใช้ที่**  ข้อ 1.3

**กรณีผิด** ลืม `degrees` → ได้ค่าเรเดียน ไม่ตรงเฉลย

---

## A5 ระยะ Euclid

**สูตร**  d = √(Σ (sᵢ − pᵢ)²)

**keyword**  `np.linalg.norm(s - p)`

**ตัวอย่างโค้ด**
```python
cols = ['income_k', 'tenure_months', 'email_opens']
x = df.iloc[118][cols].to_numpy(float)
d = np.linalg.norm(df[cols].to_numpy(dtype=float) - x, axis=1)   # ระยะของทุกคน
d[118] = np.inf                       # ตัดตัวเองออก
answer = float(d.min())

# ข้อ 1.9 — นับคนที่อยู่ห่างน้อยกว่า 9.2
answer = int((d < 9.2).sum())
```

**ทำไม**  `axis=1` คือคิดภายในแต่ละแถว (ระยะของแต่ละคน) ถ้าไม่ใส่จะได้ค่าเดียวทั้งตาราง

**ใช้ที่**  ข้อ 1.5, 1.9

---

## A6 trace(Aᵏ) รอยของเมทริกซ์กำลัง

**สูตร**  trace(Aᵏ) = ผลรวมของ eigenvalue ทั้ง k ตัว ยกกำลัง k

**keyword**  `np.trace(A @ A @ A ...)` หรือ `np.linalg.matrix_power(A, k)`

**ตัวอย่างโค้ด**
```python
A5 = np.linalg.matrix_power(A, 5)
answer = float(np.trace(A5))
```

**ทำไม**  trace คือผลรวมเส้นทแยง ไม่ต้องวนธาตุทีละตัว

**ใช้ที่**  ข้อ 2.1

**กรณีผิด** ถ้าใช้ชนิด `int` จะ overflow ให้แปลงเป็น `float` ก่อนเสมอ

---

## A7 det(A) determinant

**สูตร**  det = 0 ⟹ A กลับด้านไม่ได้ (singular)

**keyword**  `np.linalg.det(A)`

**ตัวอย่างโค้ด**
```python
XTX = X.T @ X
answer = float(np.linalg.det(XTX))
# เทียบ: det(XTX) เท่ากับ det(X)**2 เสมอ
```

**ทำไม**  ใช้ตรวจว่า design matrix อิสระแค่ไหน — det เล็กมาก แปลว่าเกือบ collinear

**ใช้ที่**  ข้อ 2.6

---

## A8 แก้ระบบสมการ Ax = b

**สูตร**  x = A⁻¹b

**keyword**  `np.linalg.solve(A, b)`

**ตัวอย่างโค้ด**
```python
A = pd.read_csv(Path('data/week02') / 'pb02_Asys.csv').to_numpy(dtype=float)
b = pd.read_csv(Path('data/week02') / 'pb02_bsys.csv').to_numpy(dtype=float).ravel()
x = np.linalg.solve(A, b)
print('เช็ค A@x - b =', np.round(A @ x - b, 12), '(ต้องเป็น 0)')
```

**ทำไม**  `solve` ทำ LU decomposition ภายในให้ — เร็วกว่าและเลขนิ่งกว่าการหา inverse

**กรณีผิด** ❌ อย่าเขียน `np.linalg.inv(A) @ b`
การหา inverse แล้วคูณ ให้ผลลัพธ์ที่มีเลขทศนิยมเป็นเลขยาวและแม่นยำกว่าเท่าที่ควร

**ใช้ที่**  ข้อ 2.2, 2.4, 2.7, 2.8, 3.2, 3.4, 3.5, 3.9, 3.10 และ fit SLR ในข้อ 8.1–8.10

---

## A9 Normal equations / least squares

**สูตร**  (XᵀX)β = Xᵀy  →  β = (XᵀX)⁻¹Xᵀy

**keyword**  `np.linalg.solve(X.T @ X, X.T @ y)` หรือ `np.linalg.lstsq(X, y)`

**ตัวอย่างโค้ด**
```python
X = np.column_stack([np.ones(len(df)), df['x'].to_numpy(float)])   # ต้องมีคอลัมน์ 1 สำหรับ intercept
beta = np.linalg.solve(X.T @ X, X.T @ y)
```

**ทำไม**  เมื่อ X ไม่กลับด้านได้ (X ไม่เต็มอันดับ) ต้องใช้ `lstsq` แทน

**ใช้ที่**  ข้อ 2.7 · เป็นวิธี fit SLR มือลงในข้อ 8.1–8.10

---

## A10 forward elimination

**สูตร**  ทำให้เมทริกซ์เป็นแบบขั้นบน (upper triangular)

**keyword**  ลูป Python ล้วน ไม่มีฟังก์ชันสำเร็จ

**ตัวอย่างโค้ด**
```python
M = E.copy()
for i in range(n - 1):
    for j in range(i + 1, n):
        factor = M[j, i] / M[i, i]
        M[j] -= factor * M[i]          # ลบแถว j โดยไม่สลับแถว
```

**ทำไม**  ต้องใช้ในการหาตัวแปรของสมการจากระบบที่ยังไม่รู้ค่า (ข้อ 2.5)

**กรณีผิด** ⚠️ **ห้ามสลับแถว** ในโจทย์ Week 2 การสลับแถวเปลี่ยนคำตอบ

**ใช้ที่**  ข้อ 2.5

---

## A11 Markov chain เมทริกซ์เปลี่ยนสถานะ

**สูตร**  P(สถานะ b | เริ่มที่ a) = สมาชิก S[a, b] เมื่อผลรวมแต่ละแถว = 1

**keyword**  อ่านค่าจากเมทริกซ์ตรง ๆ (ไม่ต้องใช้ numpy พิเศษ)

**ตัวอย่างโค้ด**
```python
S = pd.read_csv(Path('data/week02') / 'pb02_A6.csv').to_numpy(dtype=float)
answer = float(S[0, 2])        # จากสถานะแถว 0 ไปสถานะคอลัมน์ 2
```

**ทำไม**  เมทริกซ์ที่ผลรวมแต่ละแถว = 1 คือตารางความน่าจะเป็นการเปลี่ยนสถานะ

**ใช้ที่**  ข้อ 2.9

---

## A12 rank และ dim(Null A)

**สูตร**  dim(Null A) = จำนวนคอลัมน์ − rank(A)   (A ขนาด m×n)

**keyword**  `np.linalg.matrix_rank(A)`

**ตัวอย่างโค้ด**
```python
n_cols = A.shape[1]
r = int(np.linalg.matrix_rank(A))
answer = n_cols - r          # มิติของ null space
```

**ทำไม**  null space คือเวกเตอร์ที่ส่งเข้า A แล้วได้ศูนย์ ขนาดเท่ากับช่องว่างที่คอลัมน์เสียไป

**ใช้ที่**  ข้อ 3.1

---

## A13 Projection การฉายลง column space

**สูตร**  P = X(XᵀX)⁻¹Xᵀ  และ  Pb = X(XᵀX)⁻¹Xᵀb

**keyword**  `np.linalg.inv(X.T @ X) @ X.T`

**ตัวอย่างโค้ด**
```python
XtX_inv = np.linalg.inv(X.T @ X)
Pb = X @ (XtX_inv @ (X.T @ b))       # เวกเตอร์ที่ฉายแล้ว
answer = float(np.linalg.norm(Pb))  # ความยาวของเวกเตอร์ที่ฉาย
```

**ทำไม**  ความยาวของ Pb คือส่วนที่ b มีอยู่ใน column space ของ X

**ใช้ที่**  ข้อ 3.3

**กรณีผิด** ถ้าโจทย์ถาม**ส่วนที่ตก** (error) ให้เอา `b − Pb` ไม่ใช่ Pb

---

## A14 QR decomposition

**สูตร**  X = QR → R ทแยงมุม → แก้ Rx = Qᵀy แบบ back substitution

**keyword**  `np.linalg.qr(X)`

**ตัวอย่างโค้ด**
```python
Q, R = np.linalg.qr(X)          # default = 'reduced'
answer = float(abs(R[3, 3]))    # |R[3,3]| คือรากที่ 4 ของสเปกตรัม
```

**ทำไม**  QR นิ่งกว่า SVD และคำนวณเร็วกว่า — ใช้แก้ least squares ได้โดยไม่ต้องหา inverse

**สำคัญ** สมาชิกทแยงของ R มีขนาดเท่า singular values เรียงตามลำดับ

**ใช้ที่**  ข้อ 3.6

---

## A15 condition number

**สูตร**  cond(X) = σ_max / σ_min

**keyword**  `np.linalg.cond(X, p=2)`

**ตัวอย่างโค้ด**
```python
answer = float(np.linalg.cond(X, 2))       # ต้องระบุ p=2
log10 = float(np.log10(answer))
```

**ทำไม** ยิ่งสูง = ระบบ ill-conditioned = ตัวเลขไวต่อความคลาดเคลื่อนมาก
> 1e8 = แย่มาก · 1e15 = ใช้ไม่ได้ในทางปฏิบัติ

**กรณีผิด** ค่าเริ่มต้นของ `np.linalg.cond` ใช้ p=2 แต่ถ้าเขียนโจทย์ระบุ norm ต้องระบุ `p=` ให้ชัด

**ใช้ที่**  ข้อ 3.7, 3.8

---

## A16 eigenvalue / eigenvector

**สูตร**  Av = λv

**keyword**  `np.linalg.eig(A)` / `np.linalg.eigvals(A)`

**ตัวอย่างโค้ด**
```python
A = pd.read_csv(Path('data/week04') / 'pb04_A.csv').to_numpy(dtype=float)
ev = np.linalg.eigvals(A)
desc = np.sort(ev.real)[::-1]        # เรียงเอง: มาก -> น้อย
answer = float(desc[1])              # อันดับสอง
```

**ทำไม** `eig` ไม่เรียงค่าให้ต้องเรียงเองเสมอ

**กรณีผิด** สองอย่าง
- เมทริกซ์ไม่สมมาตร → ใช้ `eig` / `eigvals` ไม่ใช่ `eigh`
- eigenvector เปลี่ยนเครื่องหมายได้ (v กับ −v) → โจทย์ที่ถามค่าที่แก้เครื่องหมายได้ให้ตอบเป็น**ค่าสัมบูรณ์**

**ใช้ที่**  ข้อ 4.1 · PC ที่ข้อ 4.2, 4.4, 4.7

---

## A17 PC explained variance

**สูตร**  PCk ratio = λk / Σλ

**keyword**  `PCA` หรือคำนวณเองจาก covariance

**ตัวอย่างโค้ด**
```python
raw = cust[cols].to_numpy(dtype=float)
Z = (raw - raw.mean(axis=0)) / raw.std(axis=0, ddof=0)
U, s, Vt = np.linalg.svd(Z, full_matrices=False)
ratio = s ** 2 / (s ** 2).sum()             # สัดส่วนที่แต่ละ PC อธิบาย
answer = float(ratio[0])
```

**ทำไม** `StandardScaler` ใช้ ddof=0 ดังนั้น z-score ต้องใช้ `ddof=0` ไม่ใช่ค่าเริ่มต้นของ pandas

**กรณีผิด** standardize กับ หักค่าเฉลี่ยอย่างเดียว ให้คนละผลลัพธ์มาก (ข้อ 4.2 vs 4.4)
- standardize → 0.3689 · หักค่าเฉลี่ยอย่างเดียว → 0.9246

**ใช้ที่**  ข้อ 4.2, 4.4

---

## A18 หา k ที่ cumsum ถึงเกณฑ์

**สูตร**  k = จำนวน PC ที่น้อยที่สุดที่ cumulative ≥ threshold

**keyword**  `np.cumsum`

**ตัวอย่างโค้ด**
```python
cum = np.cumsum(ratio)
k = int(np.searchsorted(cum, 0.60) + 1)     # +1 เพราะผลลัพธ์เป็น 0-based
```

**ทำไม** ต้องหา**ตัวเล็กที่สุด** ที่ผ่าน ไม่ใช่ตัวสุดท้ายที่ยังผ่าน

**ใช้ที่**  ข้อ 4.3

---

## A19 SVD และ low-rank approximation

**สูตร**  A = UΣVᵀ  →  ประมาณ rank-k:  Aₖ = Σᵢ₌₁..ₖ σᵢuᵢvᵢᵀ

**keyword**  `np.linalg.svd(A)`

**ตัวอย่างโค้ด**
```python
U, S, Vt = np.linalg.svd(A, full_matrices=False)
A3 = (U[:, :3] * S[:3]) @ Vt[:3]
answer = float(A3[8, 7])
```

**ทำไม** S คือ singular values เรียงจากมากไปน้อยอยู่แล้ว

**ใช้ที่**  ข้อ 4.8 · ตรวจ condition number ที่ 3.7

---

## A20 Frobenius norm กับการตัด rank

**สูตร**  ‖A − Aₖ‖_F = √(Σᵢ₌ₖ₊₁ σᵢ²)

**keyword**  `np.linalg.norm(E)` โดยไม่ใส่ axis = Frobenius

**ตัวอย่างโค้ด**
```python
E = A - A2                              # A2 = ประมาณ rank-2
abs_err = float(np.linalg.norm(E))      # Frobenius
rel = abs_err / float(np.linalg.norm(A))# ความผิดพลาดสัมพัทธ์
```

**ทำไม** ไม่ใส่ `axis` = รวมทุกธาตุเข้าด้วยกัน (ค่าเดียวทั้งเมทริกซ์)

**ใช้ที่**  ข้อ 4.5, 4.9

---

## A21 loading และ PC score

**สูตร**  loading = eigenvector · PC score = (x − x̄)·v

**keyword**  `np.cov` → eigenvector / `transform`

**ตัวอย่างโค้ด**
```python
C = np.cov(camp[['radio', 'email', 'sales']].to_numpy(dtype=float), rowvar=False)
vals, vecs = np.linalg.eigh(C)               # covariance ใช้ตัวหาร n-1
v1 = vecs[:, 0]                             # loading ของ PC1
loading = float(abs(v1[2]))                 # sales อยู่คอลัมน์ที่ 2

score = Z @ Vt[0]                           # คะแนน PC1 ของทุกคน
answer = float(abs(score[466]))             # ข้อ 4.7
```

**ทำไม** `np.cov` ใช้ตัวหาร **n−1** แต่ `StandardScaler` / z-score ใช้ **n**
→ ค่าความแปรปรวนต่างกันเล็กน้อย ถ้าโจทย์บอก ddof ให้ยึดตาม

**กรณีผิด** eigenvector เปลี่ยนเครื่องหมายได้ → ถ้าโจทย์ถามค่าสัมบูรณ์ให้ใส่ `abs()`

**ใช้ที่**  ข้อ 4.6, 4.7

---

## A22 covariance matrix

**keyword**  `np.cov(M, rowvar=False)`

**ตัวอย่างโค้ด**
```python
C = np.cov(M, rowvar=False)      # ใช้ n-1
C_std = np.cov(Z, rowvar=False)  # z-score แล้ว = correlation
```

**ทำไม** covariance ของข้อมูลที่ standardize แล้ว = ค่าสหสัมพันธ์ (correlation)

**ใช้ที่**  ข้อ 4.6 · กฎ 3 สัปดาห์ 4

---

## A23 z-score (standardize)

**สูตร**  z = (x − x̄) / std โดย std ใช้ **ddof=0**

**keyword**  `StandardScaler` หรือคำนวณเอง

**ตัวอย่างโค้ด**
```python
raw = cust[cols].to_numpy(dtype=float)
Z = (raw - raw.mean(axis=0)) / raw.std(axis=0, ddof=0)   # ddof=0 เหมือน StandardScaler
```

**กรณีผิด** ⚠️ `df['x'].std()` ของ **pandas ใช้ ddof=1** โดยค่าเริ่มต้น
ถ้าใช้ค่านั้นจะได้ SD ต่างจาก `StandardScaler` → คำตอบของ PCA ผิด

**ใช้ที่**  ข้อ 4.7, 4.10 · ปรับสเกลก่อน KNN ที่ 12.6, 12.7, 12.9, 14.1, 14.5, 14.9

---
---

# B สถิติศาสตร์ Week 5-10

> ⚠️ แบ่ง train/test **ตามตำแหน่งแถว** — ห้ามใช้ `train_test_split` ห้ามใส่ `random_state`
> มาตรฐานเครื่องที่ใช้: **train = 960 แถวแรก · test = 243 แถวที่เหลือ**

## B1 แบ่ง train/test ตามตำแหน่งแถว

**ตัวอย่างโค้ด**
```python
train, test = df.iloc[:960], df.iloc[960:]      # ครึ่งแรก / ครึ่งหลัง
```

**ทำไม** โจทย์กำหนดไว้ชัดเจน ไม่ต้องสุ่ม

**ใช้ที่**  ทุกข้อที่มี train/test (สัปดาห์ 5, 8, 9, 10, 11, 12, 13)

---

## B2 Test MSE

**สูตร**  MSE = (1/n) Σ(yᵢ − ŷᵢ)²

**keyword**  `mean_squared_error(y_true, y_pred)`

**ตัวอย่างโค้ด**
```python
model = LinearRegression().fit(Xtr, ytr)       # fit ด้วย train เท่านั้น
pred = model.predict(Xte)                      # ทำนาย test
answer = float(mean_squared_error(yte, pred))
```

**ทำไม** MSE มีหน่วยเป็นกำลังสองของตัวแปรตาม จึงเทียบข้ามโมเดลยาก

**ใช้ที่**  ข้อ 5.1, 5.5, 5.8, 6.1–6.3, 6.5, 6.10, 14.3, 14.4, 14.6, 14.7, 14.9, 14.10

---

## B3 R² และ Adjusted R²

**สูตร**  R² = 1 − RSS/TSS · Adjusted R² = 1 − (1−R²)(n−1)/(n−p−1)

**keyword**  `.rsquared` · `.rsquared_adj`

**ตัวอย่างโค้ด**
```python
m = smf.ols('spend_k ~ income_k + age + I(age**2)', data=df).fit()
answer = float(m.rsquared)        # R² ไม่ใช่ Adjusted
```

**กรณีผิด** โจทย์ถาม R² ให้ใช้ `.rsquared` · ถาม Adjusted ใช้ `.rsquared_adj` — อย่าสลับ

**ใช้ที่**  ข้อ 5.6, 9.4, 9.6, 10.4, 11.2

---

## B4 baseline model

**สูตร**  ทำนายว่าเป็น**ค่าเฉลี่ยของ train** ทุกคน

**ตัวอย่างโค้ด**
```python
baseline = np.full_like(yte, ytr.mean())       # ใช้ค่าเฉลี่ยของ TRAIN ไม่ใช่ test
base_mse = mean_squared_error(yte, baseline)
```

**ทำไม** ถ้าโมเดลไม่ดีกว่า baseline แปลว่าตัวแปรไม่มีประโยชน์

**ใช้ที่**  ข้อ 5.8

---

## B5 KNN การทำนายด้วยเพื่อนบ้าน

**keyword**  `KNeighborsClassifier` / `KNeighborsRegressor`

**ตัวอย่างโค้ด**
```python
knn = KNeighborsRegressor(n_neighbors=7).fit(Xtr, ytr)
pred = knn.predict(Xte)          # ค่าเฉลี่ย y ของ K เพื่อนบ้านที่ใกล้สุด
```

**ทำไม** classifier → เสียงข้างมาก · regressor → ค่าเฉลี่ย

**กรณีผิด** K ของ classifier **ต้องเป็นเลขคี่** กันการโยงเสมอ

**ใช้ที่**  ข้อ 5.5, 5.9, 5.10, 6.9, 12.6, 12.7, 12.9, 14.1, 14.5, 14.9

---

## B6 Pearson correlation

**keyword**  `df[a].corr(df[b])` หรือ `df[[a, b]].corr().iloc[0, 1]`

**ตัวอย่างโค้ด**
```python
q1 = camp[camp['season'] == 'Q1']            # กรองก่อน แล้วค่อยหา correlation
answer = float(q1['tv'].corr(q1['radio']))
```

**ทำไม** อยู่ในช่วง −1 ถึง 1 · r² คือสัดส่วนความแปรปรวนที่ซ้อนกัน

**กรณีผิด** ต้อง**กรองกลุ่มก่อน** แล้วค่อยคำนวณ (ข้อ 7.1, 7.4) ไม่ใช่คำนวณรวมแล้วมาหาร

**ใช้ที่**  ข้อ 5.4, 7.1, 9.2, 9.10

---

## B7 skewness

**keyword**  `df[col].skew()`

**ตัวอย่างโค้ด**
```python
answer = float(df['spend_k'].skew())       # ค่าเริ่มต้น (ไม่ใส่ bias=)
```

**ทำไม** ค่าบวก = หางยาวไปทางขวา (มีค่าผิดปกติสูง ๆ หลายคน)

**ใช้ที่**  ข้อ 7.2

---

## B8 t-test เทียบ 2 กลุ่ม

**สูตร**  H₁: μ₁ ≠ μ₂ (สองด้าน) หรือ μ₁ > μ₂ (ทางเดียว)

**keyword**  `stats.ttest_ind(a, b)`

**ตัวอย่างโค้ด**
```python
g1 = df.loc[df['responded'] == 1, 'orders_90d'].to_numpy()   # กลุ่มแรก
g0 = df.loc[df['responded'] == 0, 'orders_90d'].to_numpy()   # กลุ่มที่สอง
t, p = stats.ttest_ind(g1, g0)                              # ค่าเริ่มต้น = two-sided

# ทางเดียว: กลุ่มแรกต้องเป็นกลุ่มที่ "สูงกว่า"
t2, p2 = stats.ttest_ind(g1, g0, alternative='greater')
```

**กรณีผิด** ⚠️ **ลำดับกลุ่มสำคัญ**
- ทางเดียว → สลับกลุ่มแล้วคำตอบเปลี่ยนทิศทาง
- `equal_var=True` เป็นค่าเริ่มต้น (ข้อ 7.3)

**ใช้ที่**  ข้อ 7.3, 7.4

---

## B9 t-test เทียบกับค่าคงที่

**keyword**  `stats.ttest_1samp(a, popmean)`

**ตัวอย่างโค้ด**
```python
q2 = camp.loc[camp['season'] == 'Q2', 'sales'].to_numpy()
t, p = stats.ttest_1samp(q2, 18)         # 18 คือค่าที่โจทย์ให้
```

**ทำไม** ใช้เมื่อมีค่าที่เทียบแต่ตัวอย่างมาจากประชากรเดียว

**ใช้ที่**  ข้อ 7.5

---

## B10 Chi-square test ตารางไขว้

**keyword**  `pd.crosstab` → `stats.chi2_contingency`

**ตัวอย่างโค้ด**
```python
table = pd.crosstab(camp['season'], camp['channel'])
chi2, p, dof, expected = stats.chi2_contingency(table)
```

**ทำไม** ทดสอบว่า 2 หมวดหมวดนี้ **อิสระกัน** หรือไม่

**ใช้ที่**  ข้อ 7.7, 7.8, 7.10

---

## B11 expected count

**สูตร**  expected = (ผลรวมแถว × ผลรวมคอลัมน์) / ผลรวมทั้งตาราง

**keyword**  ตัวที่ 4 ที่คืนมาจาก `chi2_contingency`

**ตัวอย่างโค้ด**
```python
table = pd.crosstab(cust['loyalty_member'], cust['churn'])
chi2, p, dof, expected = stats.chi2_contingency(table)
answer = float(expected[1, 1])          # [แถว loyalty=1, คอลัมน์ churn=1]
```

**กรณีผิด** ดัชนีคือ **[แถว, คอลัมน์]** ไม่ใช่ [คอลัมน์, แถว]

**ใช้ที่**  ข้อ 7.8

---

## B12 กฎ IQR หาค่าผิดปกติ

**สูตร**  ผิดปกติถ้า < Q1 − 1.5·IQR หรือ > Q3 + 1.5·IQR

**keyword**  `.quantile()`

**ตัวอย่างโค้ด**
```python
q1 = cust['income_k'].quantile(0.25)
q3 = cust['income_k'].quantile(0.75)
iqr = q3 - q1
upper = q3 + 1.5 * iqr
answer = int((cust['income_k'] > upper).sum())
```

**ใช้ที่**  ข้อ 7.9

---

## B13 Bootstrap percentile CI

**สูตร**  CI = percentile(ค่าที่สุ่มได้, [5, 95]) สำหรับ 90%

**keyword**  `np.percentile`

**ตัวอย่างโค้ด**
```python
boot = pd.read_csv(Path('data/week07') / 'pb07_boot_diff.csv')   # ผลต่างค่าเฉลี่ย 1000 รอบ
vals = boot['boot_diff']
lo, hi = np.percentile(vals, [5, 95])          # 90% CI แบบ percentile
answer = float(hi)
```

**กรณีผิด** ไม่ใช่ `mean ± 1.645·SD` — ต้องเอา percentile ของ**ค่าที่สุ่มได้**

**ใช้ที่**  ข้อ 7.6

---

## B14 SLR fit ด้วยมือ

**สูตร**  β = (XᵀX)⁻¹Xᵀy โดย X มีคอลัมน์แรกเป็น 1

**ตัวอย่างโค้ด**
```python
def slr(x, y):
    X = np.column_stack([np.ones(len(x)), x])      # ← คอลัมน์ 1 สำหรับ intercept
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    resid = y - X @ beta
    return float(beta[0]), float(beta[1]), resid, float(resid @ resid), len(x)
```

**กรณีผิด** ⚠️ ลืมคอลัมน์ 1 → ได้โมเดลไม่มี intercept คำตอบเพี้ยนทั้งเล่ม

**ใช้ที่**  ข้อ 8.1–8.10 (ทั้งสัปดาห์ 8 ใช้วิธีนี้)

---

## B15 RSE Residual Standard Error

**สูตร**  RSE = √(RSS / (n − 2))

**ตัวอย่างโค้ด**
```python
rse_val = float(np.sqrt(rss / (n - 2)))     # df = n - 2 เพราะประมาณ 2 พารามิเตอร์
```

**กรณีผิด** หารด้วย `n` → ค่าเล็กเกิน ต้องใช้ n−2

**ใช้ที่**  ข้อ 8.2

---

## B16 SE(β̂₁) standard error ของความชัน

**สูตร**  SE(β̂₁) = RSE / √(Σ(xᵢ − x̄)²)

**ตัวอย่างโค้ด**
```python
se = rse_val / np.sqrt(((x - x.mean()) ** 2).sum())
```

**ทำไม** ตัวส่วนคือ RSE ตัวหารคือความกระจายของ x → x กระจายแคบ = SE ใหญ่

**ใช้ที่**  ข้อ 8.1

---

## B17 t-statistic และ p-value

**สูตร**  t = β̂₁ / SE(β̂₁), df = n − 2, p = 2·P(T > |t|)

**keyword**  `stats.t.sf(|t|, df)`

**ตัวอย่างโค้ด**
```python
t_stat = beta1 / se
p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 2))
```

**ใช้ที่**  ข้อ 8.5, 8.6

---

## B18 residual

**สูตร**  residual = y − ŷ

**ตัวอย่างโค้ด**
```python
resid = y - (beta0 + beta1 * x)
answer = float(resid[233])
```

**กรณีผิด** เป็น `y − ŷ` ไม่ใช่ `ŷ − y` (สลับเครื่องหมาย)

**ใช้ที่**  ข้อ 8.8

---

## B19 log-linear

**สูตร**  log(y) = β₀ + β₁x  →  แปลงกลับ ŷ = exp(β₀ + β₁x)

**ตัวอย่างโค้ด**
```python
y_log = np.log(y)
b0, b1, *_ = slr(x, y_log)
answer = float(np.exp(b0 + b1 * x_new))
```

**ทำไม** ใช้เมื่อความสัมพันธ์เป็นแบบยกกำลัง (โตแบบเปอร์เซ็นต์)

**ใช้ที่**  ข้อ 8.10

---

## B20 `smf.ols` Multiple Linear Regression

**สูตร**  y = β₀ + β₁x₁ + … + βₚxₚ

**keyword**  `smf.ols('y ~ x1 + x2', data=df).fit()`

**ตัวอย่างโค้ด**
```python
import statsmodels.formula.api as smf
m = smf.ols('spend_k ~ income_k + tenure_months + email_opens', data=cust).fit()
print(m.summary())
```

**ทำไม** statsmodels ให้ SE, t, p, CI, AIC/BIC ครบในตัว (sklearn ไม่ให้)

**ใช้ที่**  ข้อ 9.1–9.10, 10.1–10.10, 11.2

---

## B21 `.params` `.bse` `.tvalues` `.pvalues`

**ตัวอย่างโค้ด**
```python
beta = float(m.params['x'])          # สัมประสิทธิ์
se   = float(m.bse['x'])             # standard error
t    = float(m.tvalues['x'])         # t = beta / SE
p    = float(m.pvalues['x'])         # p-value
```

**ทำไม** statsmodels เก็บผลเป็น `Series` ชื่อคอลัมน์ → เลือกด้วยชื่อ

**ใช้ที่**  ข้อ 9.1, 9.3, 11.1, 13.4 และทุกข้อที่ถามสัมประสิทธิ์/SE

---

## B22 CI กับ PI

**สูตร**
- CI = ช่วงของค่าเฉลี่ย E[y|x] → แคบ
- PI = ช่วงของค่าของคนรายหนึ่ง → กว้างกว่า (มี Var(ε) เพิ่ม)

**ตัวอย่างโค้ด**
```python
m = smf.ols('spend_k ~ income_k + web_visits + tenure_months', data=cust).fit()
new = pd.DataFrame({'income_k': [25], 'web_visits': [9], 'tenure_months': [31]})
pred = m.get_prediction(new)
ci = np.asarray(pred.conf_int(alpha=0.05)).ravel()            # ของค่าเฉลี่ย
pi = np.asarray(pred.summary_frame(alpha=0.05)[
    ['obs_ci_lower', 'obs_ci_upper']].to_numpy()).ravel()    # ของค่าจริง
answer = float(ci[0])
```

**กรณีผิด** ใช้ `obs_ci` แทน `mean_ci` เมื่อโจทย์ถาม CI ของค่าเฉลี่ย → คำตอบกว้างเกิน

**ใช้ที่**  ข้อ 9.8 (CI ขอบล่าง), 9.9 (PI ขอบบน)

---

## B23 AIC / BIC เลือกโมเดล

**สูตร**  ยิ่ง**ต่ำยิ่งดี** · BIC = AIC + ln(n)·p (โทษตัวแปรมากกว่า)

**keyword**  `m.aic` · `m.bic`

**ตัวอย่างโค้ด**
```python
full = smf.ols('sales ~ tv + radio + social + total_budget', data=camp).fit()
small = smf.ols('sales ~ tv + radio + social', data=camp).fit()
delta = float(full.aic - small.aic)     # ลบ → ตัวไหนดีกว่า
```

**กรณีผิด** เปรียบเทียบผิดทิศ — โจทย์บางข้อถาม "โมเดลไหนดีกว่า" ให้ดูว่าค่าต่ำกว่า

**ใช้ที่**  ข้อ 9.5, 9.6, 10.9

---

## B24 Backward elimination

**ตัวอย่างโค้ด**
```python
current = ['income_k', 'loyalty_member', 'age', 'basket_size', 'email_opens', 'orders_90d']
while True:
    m = smf.ols('spend_k ~ ' + ' + '.join(current), data=cust).fit()
    pv = m.pvalues.drop('Intercept')          # ไม่สนใจ p ของ intercept
    worst = pv.idxmax()
    if float(pv.max()) > 0.05:
        current = [c for c in current if c != worst]
    else:
        break
```

**ทำไม** ตัดทีละตัวแปรที่ p-value สูงที่สุดจนกว่าจะได้โมเดลที่สมเหตุสมผล

**ใช้ที่**  ข้อ 9.7

---

## B25 dummy variable

**สูตร**  `C(x)` แปลงตัวแปรหมวดเป็น 0/1 อัตโนมัติ

**ตัวอย่างโค้ด**
```python
m = smf.ols('spend_k ~ income_k + C(channel)', data=cust).fit()
store = float(m.params['C(channel)[T.store]'])
```

**ทำไม** กลุ่มอ้างอิงคือ**ค่าแรกที่เจอ** โดย default คือ `online`

**กรณีผิด** สัมประสิทธิ์ของกลุ่มอ้างอิงเป็น 0 โดยอัตโนมัติ ไม่มีใน `params`

**ใช้ที่**  ข้อ 10.1

---

## B26 `pd.cut` แบ่งกลุ่มตัวแปรต่อเนื่อง

**ตัวอย่างโค้ด**
```python
camp['tier'] = pd.cut(camp['total_budget'],
                      bins=[-np.inf, 100, 200, np.inf],
                      labels=['Low', 'Mid', 'High'])
m = smf.ols('sales ~ C(tier)', data=camp).fit()
```

**ทำไม** ต้องใส่ `-np.inf` / `np.inf` ที่ปลาย ไม่งั้นค่าที่อยู่นอกช่วงจะกลายเป็น NaN

**ใช้ที่**  ข้อ 10.2

---

## B27 interaction

**สูตร**  `y ~ x + C(g) + x:C(g)` → ความชันในกลุ่ม g = β_x + β_(x:g)

**ตัวอย่างโค้ด**
```python
m = smf.ols('spend_k ~ income_k + C(channel) + income_k:C(channel)', data=cust).fit()
slope_store = float(m.params['income_k'] + m.params['income_k:C(channel)[T.store]'])
```

**กรณีผิด** ⚠️ ความชันของกลุ่ม = **ผลรวม** ของสองค่า
ไม่ใช่แค่ `params['income_k:C(channel)[T.store]']`

**ใช้ที่**  ข้อ 10.3

---

## B28 พหุนามในสูตร `I(x**2)`

**ตัวอย่างโค้ด**
```python
m = smf.ols('spend_k ~ income_k + age + I(age**2)', data=cust).fit()
```

**กรณีผิด** ⚠️ ต้องเขียน `I(age**2)`
เขียน `age**2` ตรง ๆ statsmodels จะตีความเป็น operator อื่น ไม่ใช่ยกกำลัง

**ใช้ที่**  ข้อ 10.4

---

## B29 VIF ตรวจ multicollinearity

**สูตร**  VIF = 1 / (1 − R²ⱼ) · > 5 หรือ > 10 ถือว่ามีปัญหา

**keyword**  `variance_inflation_factor`

**ตัวอย่างโค้ด**
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
X = sm.add_constant(camp[['tv', 'radio', 'email', 'total_budget']], has_constant='add')
vif = {c: float(variance_inflation_factor(X.values, X.columns.get_loc(c)))
       for c in X.columns if c != 'const'}
```

**ทำไม** multicollinearity ทำให้ SE โต → |t| เล็ก → p-value ใหญ่ (ทั้งที่ R² ยังสูง)

**ใช้ที่**  ข้อ 10.5, 10.6

---

## B30 Cook's distance

**keyword**  `model.get_influence().cooks_distance[0]`

**ตัวอย่างโค้ด**
```python
m = smf.ols('sales ~ tv + radio + social', data=camp).fit()
cd = m.get_influence().cooks_distance[0]
answer = float(cd.max())
```

**ทำไม** วัดว่าถ้าลบแถวนั้นทิ้ง สัมประสิทธิ์ทั้งโมเดลจะเปลี่ยนแค่ไหน

**ใช้ที่**  ข้อ 10.7

---

## B31 studentized residual

**keyword**  `model.get_influence().resid_studentized_external`

**ตัวอย่างโค้ด**
```python
se = m.get_influence().resid_studentized_external
idx = int(np.abs(se).argmax())          # แถวที่มีค่าสัมบูรณ์สูงสุด
```

**กรณีผิด** "externally" = standardized ด้วย SE ที่ไม่รวมแถวนั้นเอง
มีทั้งแบบ internally กับ externally ให้ดูว่าโจทย์บอกอะไร

**ใช้ที่**  ข้อ 10.8

---

## B32 `predict()` ต้องส่งคอลัมน์ครบ

**ตัวอย่างโค้ด**
```python
new = pd.DataFrame([[233, 33, 'Q4']], columns=['tv', 'radio', 'season'])  # ครบทุกคอลัมน์
answer = float(m.predict(new)[0])
```

**กรณีผิด** ⚠️ ถ้าโมเดลใช้ `tv, radio, tv:radio, C(season)`
ต้องส่ง DataFrame ที่มี **ทั้ง 4 อย่าง** รวม `season` ด้วย แม้จะทำนายแค่ค่าเดียว

**ใช้ที่**  ข้อ 10.10

---
---

# C การจำแนกโญกลุ่ม Week 11-13

> ⚠️ **คลาสบวก = 1** เสมอ · แบ่ง train/test 960/243 ไม่สับ

## C1 `smf.logit` Logistic Regression

**สูตร**  P(y=1|x) = 1 / (1 + exp(−(β₀ + β₁x₁ + β₂x₂)))

**keyword**  `smf.logit('y ~ x', data=df).fit(disp=0)`

**ตัวอย่างโค้ด**
```python
m = smf.logit('responded ~ email_opens + income_k + age', data=cust).fit(disp=0)
z = float(m.tvalues['email_opens'])
```

**ทำไม** `disp=0` กันข้อความ ConvergenceWarning รบกวน

**กรณีผิด** ⚠️ ❌ อย่าใช้ `LogisticRegression()` ของ sklearn
มันมี **L2 regularization (C=1)** เป็นค่าเริ่มต้น → ค่าไม่ตรงเฉลย
โจทย์บอก "Maximum Likelihood ไม่มี regularization" → ต้องใช้ `smf.logit` หรือ `sm.GLM`

**ใช้ที่**  ข้อ 11.1, 11.3–11.7, 11.9, 11.10, 13.8

---

## C2 odds ratio

**สูตร**  odds ratio = exp(β̂ · k)  โดย k = จำนวนหน่วยที่เพิ่มขึ้น

**ตัวอย่างโค้ด**
```python
beta = float(m.params['email_opens'])
answer = float(np.exp(3 * beta))      # เปิดอีเมลเพิ่ม 3 ฉบับ
```

**ทำไม** ต้องคูณ β ด้วย k ก่อน แล้วค่อย exp

**ใช้ที่**  ข้อ 11.4, 13.1, 13.5

---

## C3 แก้สมการ logistic ย้อนกลับ

**สูตร**  หา x จาก P:  x = (logit(P) − β₀ − β₂·x₂) / β₁

**ตัวอย่างโค้ด**
```python
P = 0.4
logit_P = np.log(P / (1 - P))            # ≈ -0.4055
x_needed = (logit_P - b0 - b2 * income) / b1
```

**กรณีผิด** ⚠️ logit(0.4) = **ln(0.4/0.6)** = −0.4055
ไม่ใช่ `np.log(0.4)` = −0.916 (ข้อ 11.5)

**ใช้ที่**  ข้อ 11.5

---

## C4 confusion matrix

**keyword**  `confusion_matrix(y_true, y_pred)`

**สูตร**  ลำดับที่คืนมา: **TN, FP, FN, TP**

**ตัวอย่างโค้ด**
```python
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
```

**ทำไม** sklearn คืนเป็นแถว 2×2 เรียงตาม **คลาสจริง = แถว**, **คลาสทำนาย = คอลัมน์**

**ใช้ที่**  ข้อ 11.3, 12.2, 12.4, 12.8, 13.9

---

## C5 F1 และ recall

**สูตร**  F1 = 2·TP/(2·TP+FP+FN) · recall = TP/(TP+FN)

**keyword**  `f1_score` · `recall_score`

**ตัวอย่างโค้ด**
```python
f1 = float(f1_score(y_test, pred, pos_label=1))
rec = float(recall_score(y_test, pred, pos_label=1))
```

**ใช้ที่**  ข้อ 11.3, 11.10, 12.2, 12.9, 14.1

---

## C6 เกณฑ์จำแนกคลาส

**สูตร**  ทำนายเป็นคลาส 1 เมื่อ **ความน่าจะเป็น ≥ threshold**

**ตัวอย่างโค้ด**
```python
pred = (proba >= 0.12).astype(int)       # ≥ ไม่ใช่ >
```

**กรณีผิด** ใช้ `>` แทน `≥` → ค่าที่ตรงเท่ากับ threshold พอดีจะหลุด

**ใช้ที่**  ข้อ 11.3, 11.6, 11.7, 12.8, 13.7, 13.9

---

## C7 GLM Binomial + `freq_weights` ถ่วงน้ำหนักคลาส

**ตัวอย่างโค้ด**
```python
ytr = train['churn'].to_numpy()
n_pos = int((ytr == 1).sum())
n_neg = int((ytr == 0).sum())
w_pos = len(ytr) / (2 * n_pos)          # น้ำหนักของคลาสเลิกซื้อ (คลาสหายาก)
w_neg = len(ytr) / (2 * n_neg)
m = sm.GLM(train['churn'], sm.add_constant(train[cols]),
           family=sm.families.Binomial(),
           freq_weights=np.where(ytr == 1, w_pos, w_neg)).fit()
```

**ทำไม** `freq_weights` ทำให้ได้ MLE ถ่วงน้ำหนักแบบเดียวกับ `class_weight='balanced'`
โดยไม่มี regularization

**ใช้ที่**  ข้อ 11.8

---

## C8 `class_weight='balanced'`

**สูตร**  w_c = n_samples / (n_classes × n_c)

**ตัวอย่างโค้ด**
```python
n_pos = int((ytr == 1).sum())
n_neg = int((ytr == 0).sum())
w_pos = len(ytr) / (2 * n_pos)          # <- คำตอบข้อ 13.6
w_neg = len(ytr) / (2 * n_neg)
```

**กรณีผิด** ⚠️ น้ำหนักคำนวณจาก **train** ไม่ใช่จากทั้งชุด (ข้อ 11.8, 13.6)

**ใช้ที่**  ข้อ 11.8, 13.6

---

## C9 LDA และ QDA

| | LDA | QDA |
|---|---|---|
| สมมติฐาน | covariance **เดียวกัน** ทุกคลาส | ต่างกันได้ |
| ขอบเขต | เส้นตรง | เส้นโค้ง |

**keyword**  `LinearDiscriminantAnalysis` · `QuadraticDiscriminantAnalysis`

**ตัวอย่างโค้ด**
```python
m = LinearDiscriminantAnalysis().fit(train[cols], train['responded'])
diff = int((m.predict(test[cols]) != qda.predict(test[cols])).sum())
```

**ทำไม** ถ้า QDA ไม่ดีกว่า LDA → ข้อมูลเกือบเป็น Gaussian รูปทรงเดียวกัน
LDA มีพารามิเตอร์น้อยกว่าจึงประมาณได้ดีกว่า

**ใช้ที่**  ข้อ 12.1, 12.3, 12.5, 12.8, 13.2, 13.8, 14.8

---

## C10 Gaussian Naive Bayes

**keyword**  `GaussianNB`

**ตัวอย่างโค้ด**
```python
gnb = GaussianNB().fit(train[cols], train['responded'])
```

**ทำไม** สมมติฐานตัวแปรอิสระกันในคลาส → ไม่มี interaction ในโมเดล

**ใช้ที่**  ข้อ 12.4, 13.8

---

## C11 `predict` กับ `predict_proba`

**keyword**  `predict_proba(X)[:, 1]`

**ตัวอย่างโค้ด**
```python
proba = gnb.predict_proba(new)[0]      # [ความน่าจะเป็นคลาส 0, ความน่าจะเป็นคลาส 1]
answer = float(proba[1])
pred_class = gnb.predict(new)[0]       # คลาสที่ prob สูงสุด
```

**กรณีผิด** ⚠️ ต้องดึง **`[:, 1]`** เสมอ
ถ้าใช้ `proba[0]` ได้ความน่าจะเป็นของ**คลาส 0**

**⚠️ ข้อสำคัญข้ามสัปดาห์** `LogitResults` ของ **statsmodels ไม่มี** `predict_proba()`
มีแค่ `predict()` — ถ้าต้องความน่าจะเป็นให้คำนวณเอง:
```python
P = 1 / (1 + np.exp(-(b0 + b1 * x1 + b2 * x2)))
```

**ใช้ที่**  ข้อ 12.3, 12.6, 12.8, 12.10, 13.2, 13.8, 14.8

---

## C12 KNN + `StandardScaler` (pipeline)

**ตัวอย่างโค้ด**
```python
pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=21))
pipe.fit(Xtr, ytr)          # scaler fit กับ train เท่านั้น โดยอัตโนมัติ
```

**ทำไม** KNN วัดระยะทาง → ตัวแปรสเกลใหญ่ครอบงำตัวอื่น
pipeline ทำให้ scaler ถูก fit เฉพาะ train โดยไม่ต้องคิดเอง

**ใช้ที่**  ข้อ 12.6, 12.7, 12.9, 14.1, 14.5, 14.9

---

## C13 Poisson GLM

**สูตร**  ใช้ **log link** → ค่าทำนายเป็นจำนวนนับเสมอ

**keyword**  `smf.glm(..., family=sm.families.Poisson())`

**ตัวอย่างโค้ด**
```python
m = smf.glm('orders_90d ~ loyalty_member + tenure_months + income_k',
            data=cust, family=sm.families.Poisson()).fit()
predicted = float(m.predict(new_df)[0])     # อยู่บนสเกลเดิม = จำนวนออเดอร์
```

**ทำไม** ใช้เมื่อตัวแปรตามเป็น**จำนวนนับ** (ไม่ติดลบ)

**ใช้ที่**  ข้อ 13.1, 13.3, 13.4, 13.5

---

## C14 IRR Incidence Rate Ratio

**สูตร**  IRR = e^β

**ตัวอย่างโค้ด**
```python
beta = float(m.params['loyalty_member'])
answer = float(np.exp(beta))
# ถ้าตัวแปรเพิ่มขึ้น 24 หน่วย
answer24 = float(np.exp(24 * m.params['tenure_months']))
```

**ทำไม** Poisson ใช้ log link → ตัวคูณจริงคือ exp(β·k)

**ใช้ที่**  ข้อ 13.1, 13.5

---

## C15 `roc_curve` เส้น ROC

**สูตร**  AUC = พื้นที่ใต้เส้น · 0.5 = เดาเสมอ, 1.0 = สมบูรณ์แบบ

**keyword**  `roc_curve` · `roc_auc_score`

**ตัวอย่างโค้ด**
```python
m = smf.logit('responded ~ age + email_opens + income_k', data=train).fit(disp=0)
p = m.predict(test)                    # statsmodels ใช้ predict() ไม่มี predict_proba
fpr, tpr, thr = roc_curve(test['responded'], p)
ok_idx = np.where(tpr >= 0.85)[0]
i = int(ok_idx.min())                  # ดัชนีแรก = threshold สูงสุดที่ยังผ่าน
answer = float(fpr[i])
```

**กรณีผิด** ⚠️⚠️ **`thr` เรียงจากมากไปน้อย** (ตัวแรกคือ `inf`)
- "สูงสุดที่ยังผ่าน" → `ok.min()`
- ใช้ `ok.max()` ได้ threshold ต่ำสุด → FPR = 1.0 → **พลาดข้อ 13.7**

**ใช้ที่**  ข้อ 13.7 · `roc_auc_score` ใช้ที่ 12.10, 13.2, 13.8, 13.10, 14.8

---

## C16 AUC คุณภาพการจัดอันดับ

**ตัวอย่างโค้ด**
```python
answer = float(roc_auc_score(y_test, model.predict_proba(Xtest)[:, 1]))
```

**ทำไม** AUC ไม่ต้องเลือก threshold — วัดว่าโมเดลจัดอันดับผู้ตอบรับไว้บนสุดได้แค่ไหน

**กรณีผิด** ถ้าใช้ `age` ดิบเป็นคะแนน AUC ก็ไม่ใช่ 0.5 เสมอ
เพราะกลุ่มที่ต่างกันอาจมีอายุเฉลี่ยต่างกันอยู่แล้ว (ข้อ 13.10)

**ใช้ที่**  ข้อ 12.10, 13.2, 13.8, 13.10, 14.8

---
---

# D การประเมินโมเดล Week 14

> ⚠️ **ห้ามสับข้อมูลเด็ดขาด** — ห้ามใส่ `shuffle` / `random_state` ห้ามใช้ `train_test_split`

## D1 `KFold` แบ่ง K ก้อนตามลำดับแถว

**ตัวอย่างโค้ด**
```python
folds = list(KFold(n_splits=7).split(X))     # ไม่สับ -> เรียงตามลำดับแถว
_, va7 = folds[6]                            # fold ลำดับที่ 7 = ก้อนสุดท้าย
answer = float(X[va7].mean())
```

**กรณีผิด** ⚠️ เมื่อไม่สับ fold จะเรียงตามลำดับแถว
**fold ลำดับที่ K = ก้อนสุดท้าย** = `folds[K-1]`

**ใช้ที่**  ข้อ 14.2, 14.3, 14.6, 14.7, 14.9, 14.10

---

## D2 `StratifiedKFold` รักษาสัดส่วนคลาส

**ตัวอย่างโค้ด**
```python
skf = StratifiedKFold(n_splits=5)
for tr, va in skf.split(X, y):      # ← ต้องส่ง y เป็นตัวแบ่งชั้น
    ...
```

**ทำไม** ใช้เมื่อคลาสหายาก เพื่อให้ทุก fold มีสัดส่วนคลาสใกล้เคียงกัน

**กรณีผิด** ลืมส่ง `y` → กลายเป็น KFold ธรรมดา ไม่ได้ stratify

**ใช้ที่**  ข้อ 14.1, 14.8

---

## D3 `LeaveOneOut` ตัดออกทีละคน

**ตัวอย่างโค้ด**
```python
errors = 0
for tr, va in LeaveOneOut().split(X):
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=9))
    pipe.fit(X[tr], y[tr])                  # fit ใหม่ทุกรอบ
    errors += int((pipe.predict(X[va]) != y[va]).sum())
answer = errors / len(y)                   # ผลรวม error ÷ จำนวนข้อมูล
```

**ทำไม** error rate = ผลรวม error ÷ n (เท่ากับค่าเฉลี่ยของ n ค่า)

**ใช้ที่**  ข้อ 14.5

---

## D4 `make_pipeline` รวมขั้นตอนให้ fit ถูกที่

**ตัวอย่างโค้ด**
```python
pipe = make_pipeline(StandardScaler(), LinearRegression())
pipe.fit(X[tr], y[tr])
pred = pipe.predict(X[va])        # scaler transform ด้วยค่าจาก train เท่านั้น
```

**กรณีผิด** ⚠️ ถ้าเขียน `StandardScaler().fit_transform(X)` ก่อนแล้วค่อยแบ่ง fold
= ข้อมูลจาก validation รั่วเข้าไปในการ fit (data leakage)

**ใช้ที่**  ข้อ 6.x, 12.6, 12.7, 12.9, ทุกข้อของสัปดาห์ 14

---

## D5 `PolynomialFeatures` + ดัชนีสัมประสิทธิ์

**สูตร**  โมเดลดีกรี d = ถดถอยเชิงเส้นด้วย [1, x, x², …, xᵈ] พร้อม intercept

**ตัวอย่างโค้ด**
```python
pipe = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(Xtr, ytr)
coef = pipe.named_steps['linearregression'].coef_      # ยาว d+1
x_d_coef = coef[d]      # ← สัมประสิทธิ์ของ xᵈ อยู่ที่ index d
```

**กรณีผิด** ⚠️ `PolynomialFeatures(d)` ให้คอลัมน์ **d+1** ตัว
สัมประสิทธิ์ของ xᵈ อยู่ที่ **`coef_[d]`** ไม่ใช่ `coef_[d-1]` → พลาดข้อ 6.4

**ทำไม** `np.polyfit` คืนสัมประสิทธิ์**กลับด้าน** (จากกำลังสูงสุดมาต่ำสุด) → `[0]` คือ xᵈ

**ใช้ที่**  ข้อ 6.1, 6.4, 6.5, 6.10, 14.4, 14.6, 14.7

---

## D6 ค่า CV = ค่าเฉลี่ยของทุก fold

**สูตร**  CV MSE = (1/K) Σ MSEᵢ  **ทุก fold น้ำหนักเท่ากัน**

**ตัวอย่างโค้ด**
```python
mses = []
for tr, va in KFold(n_splits=5).split(X):
    m = LinearRegression().fit(X[tr], y[tr])
    mses.append(mean_squared_error(y[va], m.predict(X[va])))
answer = float(np.mean(mses))          # ไม่ใช่ np.average(..., weights=n_va)
```

**กรณีผิด** ไม่ถ่วงตามขนาดของแต่ละ fold

**ใช้ที่**  ข้อ 14.1, 14.3, 14.8, 14.9, 14.10

---

## D7 one-standard-error rule

**สูตร**
1. หาดีกรีที่ CV MSE ต่ำที่สุด
2. `SE = SD ของ 5 fold (ddof=1) / √5`
3. เกณฑ์ = CV MSE ต่ำสุด + SE
4. เลือกดีกรี**ต่ำที่สุด** ที่ CV MSE ไม่เกินเกณฑ์

**ตัวอย่างโค้ด**
```python
best = min(cv, key=lambda d: cv[d].mean())
se = float(cv[best].std(ddof=1) / np.sqrt(len(cv[best])))   # ddof=1
cutoff = float(cv[best].mean()) + se
chosen = min(d for d in (1, 2, 3) if cv[d].mean() <= cutoff)
answer = float(cv[chosen].mean())
```

**ทำไม** ข้อนี้เลือกดีกรี 1 แม้ดีกรี 3 ดีที่สุด — เพราะทั้งสามถือว่าไม่ต่างกันอย่างมีนัยสำคัญ

**ใช้ที่**  ข้อ 14.6

---

## D8 generalization gap

**สูตร**  gap = ค่าเฉลี่ย validation MSE − ค่าเฉลี่ย train MSE

**ตัวอย่างโค้ด**
```python
va_mse, tr_mse = [], []
for tr, va in KFold(n_splits=5).split(X):
    m = make_pipeline(PolynomialFeatures(3), LinearRegression()).fit(X[tr], y[tr])
    va_mse.append(mean_squared_error(y[va], m.predict(X[va])))
    tr_mse.append(mean_squared_error(y[tr], m.predict(X[tr])))
answer = float(np.mean(va_mse) - np.mean(tr_mse))
```

**ทำไม** ใกล้ 0 = ไม่ได้จำซ้ำ · ใหญ่ = overfit

**ใช้ที่**  ข้อ 14.7

---

## D9 CV MSE เทียบ Test MSE ของการแบ่งครั้งเดียว

**ตัวอย่างโค้ด**
```python
m = LinearRegression().fit(X[:960], y[:960])          # วิธี A: 960/243
test_mse = float(mean_squared_error(y[960:], m.predict(X[960:])))

cv = [mean_squared_error(y[va], mm.predict(X[va]))
      for tr, va in KFold(n_splits=5).split(X)
      for mm in [LinearRegression().fit(X[tr], y[tr])]]  # วิธี B: 5-fold
answer = float(np.mean(cv)) - test_mse                # ติดลบได้
```

**ทำไม** บวก = การแบ่งครั้งเดียวทำให้เห็นภาพที่ดีเกินจริง

**ใช้ที่**  ข้อ 14.10

---
---

# ดัชนี keyword ข้อ week

ค้นด้วย `Ctrl+F` · สแกนจากโน้ตบุ๊กจริงทั้ง 14 เล่ม ตัดบรรทัด `import` ออกแล้ว

## numpy

| keyword | week | ข้อ |
|---|---|---|
| `np.linalg.norm` | 1, 3, 4 | 1.1–1.5, 1.7–1.10, 3.3, 4.5, 4.9 |
| `np.linalg.solve` | 2, 3, 8 | 2.2, 2.4, 2.7, 2.8, 3.2, 3.4, 3.5, 3.9, 3.10, 8.1–8.10 |
| `np.linalg.det` | 2 | 2.6 |
| `np.linalg.lstsq` | 2 | 2.7 |
| `np.linalg.inv` | 3 | 3.3 |
| `np.linalg.matrix_rank` | 3 | 3.1 |
| `np.linalg.qr` | 3 | 3.6 |
| `np.linalg.cond` | 3 | 3.7, 3.8 |
| `np.linalg.svd` | 3, 4 | 3.7, 4.2, 4.3, 4.4, 4.5, 4.7–4.10 |
| `np.linalg.eig` / `eigvals` | 4 | 4.1 |
| `np.polyfit` | 6 | 6.4 |
| `np.cos` · `np.arccos` · `np.degrees` | 1 | 1.2, 1.3 |
| `np.sin` | 6 | 6.2, 6.3, 6.6, 6.8 |
| `np.exp` | 8, 11, 13 | 8.10, 11.4, 11.5, 13.1, 13.5 |
| `np.log` | 8, 11 | 8.10, 11.5 |
| `np.log10` | 3 | 3.8 |
| `np.var` | 6 | 6.7, 6.8 |
| `np.std` · `np.sqrt` | 4, 8, 14 | 4.7, 4.10, 8.1–8.10, 14.6 |
| `np.cov` | 4 | 4.6 |
| `np.cumsum` | 4 | 4.3 |
| `np.argmax` | 4 | 4.6 |
| `np.percentile` · `np.quantile` | 7 | 7.6 |
| `np.unique` | 12 | 12.6 |
| `np.mean` | 14 | 14.1, 14.3, 14.7, 14.8, 14.9, 14.10 |

## pandas

| keyword | week | ข้อ |
|---|---|---|
| `pd.read_csv` | 1–14 | ทุกข้อ |
| `.corr()` | 5, 7, 9 | 5.4, 7.1, 9.2, 9.10 |
| `.skew()` | 7 | 7.2 |
| `.quantile()` | 7 | 7.9 |
| `pd.crosstab` | 7 | 7.7, 7.8, 7.10 |
| `pd.cut` | 7, 10 | 7.10, 10.2 |
| `.value_counts()` | 10 | 10.2 |

## scipy

| keyword | week | ข้อ |
|---|---|---|
| `stats.ttest_ind` | 7 | 7.3, 7.4 |
| `stats.ttest_1samp` | 7 | 7.5 |
| `stats.chi2_contingency` | 7 | 7.7, 7.8, 7.10 |
| `stats.t` | 8 | 8.6 |

## scikit-learn

| keyword | week | ข้อ |
|---|---|---|
| `LinearRegression` | 5, 6, 14 | 5.1, 5.6–5.8, 5.10, 6.1, 6.3–6.5, 6.10, 14.3, 14.4, 14.6, 14.7, 14.10 |
| `KNeighborsRegressor` | 5, 14 | 5.5, 5.9, 5.10, 14.9 |
| `KNeighborsClassifier` | 6, 12, 14 | 6.9, 12.6, 12.7, 12.9, 14.1, 14.5 |
| `LinearDiscriminantAnalysis` | 12, 13, 14 | 12.1, 12.2, 12.5, 12.8, 13.2, 13.8, 14.8 |
| `QuadraticDiscriminantAnalysis` | 12, 13 | 12.3, 12.5, 12.10, 13.8 |
| `GaussianNB` | 12, 13 | 12.4, 13.8 |
| `KFold` | 14 | 14.2, 14.3, 14.6, 14.7, 14.9, 14.10 |
| `StratifiedKFold` | 14 | 14.1, 14.8 |
| `LeaveOneOut` | 14 | 14.5 |
| `make_pipeline` | 6, 12, 14 | 6.1–6.5, 6.10, 12.6, 12.7, 12.9, 14.x |
| `PolynomialFeatures` | 6, 14 | 6.1–6.5, 6.10, 14.4, 14.6, 14.7 |
| `StandardScaler` | 4, 12, 14 | 4.7, 4.10, 12.6, 12.7, 12.9, 14.1, 14.5, 14.9 |
| `mean_squared_error` | 5, 6, 14 | 5.1, 5.5, 5.8, 6.1–6.3, 6.5, 6.10, 14.3, 14.4, 14.6, 14.7, 14.9, 14.10 |
| `accuracy_score` | 6 | 6.9 |
| `f1_score` | 11, 12, 14 | 11.3, 11.10, 12.2, 12.9, 14.1 |
| `recall_score` | 11, 12, 13 | 11.8, 12.4, 12.8, 13.9 |
| `confusion_matrix` | 11, 12, 13 | 11.3, 12.2, 12.4, 12.8, 13.9 |
| `roc_auc_score` | 12, 13, 14 | 12.10, 13.2, 13.8, 13.10, 14.8 |
| `roc_curve` | 13 | 13.7 |

## statsmodels

| keyword | week | ข้อ |
|---|---|---|
| `smf.ols` | 9, 10, 11 | 9.1–9.10, 10.1–10.10, 11.2 |
| `smf.logit` | 11, 13 | 11.1, 11.3–11.7, 11.9, 11.10, 13.7–13.9 |
| `smf.glm` · `sm.families.Poisson` | 13 | 13.1, 13.3, 13.4, 13.5 |
| `sm.GLM` · `sm.families.Binomial` | 11 | 11.8 |
| `sm.add_constant` | 10, 11 | 10.5, 11.8 |
| `variance_inflation_factor` | 10 | 10.5, 10.6 |
| `cooks_distance` | 10 | 10.7 |
| `resid_studentized_external` | 10 | 10.8 |
| `.params` | 9, 10, 11, 13 | 9.1, 9.3–9.5, 9.10, 10.1–10.4, 10.6, 10.10, 11.1, 11.4, 11.5, 13.1–13.5 |
| `.bse` | 9, 10, 11, 13 | 9.3, 9.10, 10.6, 11.1, 13.4 |
| `.tvalues` · `.pvalues` | 9, 10, 11, 13 | 9.3, 9.7, 9.10, 10.1, 10.2, 10.9, 11.1, 13.4 |
| `.conf_int()` | 9 | 9.8, 9.9 |
| `.rsquared` · `.rsquared_adj` | 9, 10, 11 | 9.4, 9.6, 10.4, 11.2 |
| `.aic` · `.bic` | 9, 10 | 9.5, 9.6, 9.7, 10.9 |

---

## ลิงก์ไปสัปดาห์แต่ละสัปดาห์

[1](week1.md) · [2](week2.md) · [3](week3.md) · [4](week4.md) · [5](week5.md) ·
[6](week6.md) · [7](week7.md) · [8](week8.md) · [9](week9.md) · [10](week10.md) ·
[11](week11.md) · [12](week12.md) · [13](week13.md) · [14](week14.md)

กลับสู่ [`../README.md`](../README.md)