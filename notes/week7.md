# Week 7 — EDA, Bootstrap CI, t-test และ Chi-square

**ไฟล์โน้ตบุ๊ก:** [`week7_practice.ipynb`](../week7_practice.ipynb)  
**ข้อมูล:** `../data/week07/`  
**ไลบรารี:** numpy, pandas, scipy

> สรุปประกอบการอ่าน — รายละเอียดของแต่ละข้ออยู่ในโน้ตบุ๊ก

## สูตรที่ใช้

- **Pearson r** — `df[a, b].corr()` (ddof ไม่มีผล)
- **skewness** — `df[col].skew()` (ค่าเริ่มต้น)
- **two-sample t-test** — `stats.ttest_ind(a, b)` · ทางเดียวเพิ่ม `alternative='greater'`
- **one-sample t-test** — `stats.ttest_1samp(a, popmean)`
- **Chi-square** — `stats.chi2_contingency(table)` → คืน chi2, p-value, dof, **expected**
- **กฎ IQR** — จุดผิดปกติ: < Q1 − 1.5·IQR หรือ > Q3 + 1.5·IQR
- **Bootstrap percentile CI** — = `np.percentile(boot, [5, 95])` — ไม่ใช่ค่าเฉลี่ย ± 1.645·SD

## จุดที่คนตกบ่อย

- **ลำดับกลุ่มสำคัญ** — เมื่อใช้ `alternative='greater'` → กลุ่มแรกต้องเป็นกลุ่มที่ “สูงกว่า”
- **`.iloc[741:1041]`** — เพื่อให้ได้แถว 741–1040 รวมทั้งสองปลาย
- **expected count** — มาจาก `chi2_contingency` **ตัวที่ 4** ไม่ใช่ตัวแรก
- **Bootstrap ต้องสุ่มซ้ำหลายรอบ** — แล้วใช้ percentile ของค่าที่ได้

## สรุปคำตอบทั้ง 10 ข้อ

| ข้อ | หัวข้อ | คำตอบ | ปัดเศษ | ความคลาดเคลื่อนที่เว็บยอมรับ |
|---|---|---|---|---|
| 7.1 | Pearson correlation เฉพาะสมาชิกสะสมแต้ม | **0.1369** | 4 ตำแหน่ง | 0.00015 |
| 7.2 | ความเบ้ (skewness) ของ spend_k | **0.3807** | 4 ตำแหน่ง | 0.00015 |
| 7.3 | two-sample t-test: จำนวนออเดอร์ของคนตอบโปรโมชัน | **2.822** | 4 ตำแหน่ง | 0.00015 |
| 7.4 | t-test ทางเดียว (H₁: μ₁ > μ₂) บนแถว 741–1040 | **0.0060** | 4 ตำแหน่ง | 0.00015 |
| 7.5 | one-sample t-test: ยอดขายแคมเปญไตรมาส Q2 เทียบ 18 | **0.0788** | 4 ตำแหน่ง | 0.00015 |
| 7.6 | ขอบบนของ 90% Bootstrap CI (percentile method) | **0.2768** | 4 ตำแหน่ง | 0.00015 |
| 7.7 | Chi-square test of independence: channel × responded | **7.1899** | 4 ตำแหน่ง | 0.00015 |
| 7.8 | Expected count ของช่อง loyalty = 1 และ churn = 1 | **53.848** | 3 ตำแหน่ง | 0.0015 |
| 7.9 | นับลูกค้าที่รายได้สูงผิดปกติด้วยกฎ IQR | **39** | จำนวนเต็ม | 0.5 |
| 7.10 | Chi-square: กลุ่มอายุ 3 ช่วง × churn | **11.3905** | 4 ตำแหน่ง | 0.00015 |

---

ติดตั้งไลบรารี: `pip install -r requirements.txt`
