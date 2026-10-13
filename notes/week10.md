# Week 10 — Dummy variable, Interaction, Polynomial, Diagnostics

**ไฟล์โน้ตบุ๊ก:** [`week10_practice.ipynb`](../week10_practice.ipynb)  
**ข้อมูล:** `../data/week10/`  
**ไลบรารี:** numpy, pandas, statsmodels

> สรุปประกอบการอ่าน — รายละเอียดของแต่ละข้ออยู่ในโน้ตบุ๊ก

## สูตรที่ใช้

- **Dummy หมวด** — `C(channel)` — กลุ่มอ้างอิงคือค่าแรกที่เจอ
- **Polynomial ในสูตร** — ต้องเขียน **`I(age**2)`** ไม่ใช่ `age**2` ตรง ๆ
- **Interaction** — `y ~ x + C(g) + x:C(g)` → ความชันในกลุ่ม g = β_x + β_(x:g)
- **VIF** — = 1/(1−R²ⱼ) → `variance_inflation_factor(X, i)` (> 5 หรือ > 10 ถือว่ามีปัญหา)
- **Cook's distance** — `model.get_influence().cooks_distance[0]`
- **Externally studentized residual** — `model.get_influence().resid_studentized_external`

## จุดที่คนตกบ่อย

- **`predict()` ต้องส่ง DataFrame ที่มีคอลัมน์ครบ** — ทุกตัวแปรที่โมเดลใช้ รวมถึงตัวแปรหมวด
- **interaction slope ต้องบวกกัน** — ไม่ใช่แค่อ่าน interaction term อย่างเดียว
- **R² ไม่ใช่ Adjusted R²** — ถ้าโจทย์ถาม R² ให้ใช้ `model.rsquared`

## สรุปคำตอบทั้ง 10 ข้อ

| ข้อ | หัวข้อ | คำตอบ | ปัดเศษ | ความคลาดเคลื่อนที่เว็บยอมรับ |
|---|---|---|---|---|
| 10.1 | สัมประสิทธิ์ของกลุ่ม store (dummy variable) | **-0.4874** | 4 ตำแหน่ง | 0.00015 |
| 10.2 | สัมประสิทธิ์ของกลุ่ม Long (แบ่งกลุ่มต่อเนื่องด้วย pd.cut) | **0.3998** | 4 ตำแหน่ง | 0.00015 |
| 10.3 | interaction: ความชันของ income_k **เฉพาะกลุ่ม store** | **0.0432** | 4 ตำแหน่ง | 0.00015 |
| 10.4 | R² ของโมเดลพหุนามดีกรี 2 | **0.2263** | 4 ตำแหน่ง | 0.00015 |
| 10.5 | VIF ของ total_budget (ตรวจ multicollinearity) | **24.38** | 2 ตำแหน่ง | 0.015 |
| 10.6 | SE(tv) โตขนาดเท่าไหร่เมื่อมี multicollinearity | **10.401** | 3 ตำแหน่ง | 0.0015 |
| 10.7 | Cook's distance ที่มากที่สุด | **0.0583** | 4 ตำแหน่ง | 0.00015 |
| 10.8 | externally studentized residual ที่มีค่าสัมบูรณ์สูงสุด | **1087** | จำนวนเต็ม | 0.5 |
| 10.9 | interaction ดีกว่าจริงไหม (เทียบ AIC) | **107.76** | 2 ตำแหน่ง | 0.015 |
| 10.10 | ทำนายยอดขายของแคมเปญ Q4 ที่ tv=233, radio=33 | **29.09** | 2 ตำแหน่ง | 0.015 |

---

ติดตั้งไลบรารี: `pip install -r requirements.txt`
