# SPOTME AI Scout

نظام تعرف اصطناعي (AI) لاكتشاف اللاعبين الرياضيين، يعتمد على نموذج لغوي عبر منصة **Groq** لإجراء مقابلة ذكية مع اللاعب، بناء ملف تعريفي (Player Profile) منظّم، تشغيل عدة "وكلاء" تقييم (Scout Agents)، ثم إصدار تقرير نهائي بصيغتي **PDF** و **JSON**.

المشروع متوفر بصيغتين:
- `main.py`: تطبيق **FastAPI** يعرض الوظائف كـ API.
- `AI_Scout.ipynb`: نفس المنطق داخل Jupyter Notebook للتجربة والتشغيل التفاعلي.

## المميزات

- **مقابلة ذكية تكيّفية**: يطرح النموذج سؤالاً واحداً في كل مرة، ويُعدّل الأسئلة التالية بناءً على إجابات اللاعب، دون قائمة أسئلة ثابتة.
- **الرياضات المدعومة**: كرة القدم، كرة السلة، الكرة الطائرة، كرة اليد (Football, Basketball, Volleyball, Handball).
- **استخراج بيانات منظّمة**: تحويل نص المقابلة إلى ملف تعريف (Profile) بحقول محددة (الاسم، العمر، المركز، الطول، الوزن، سنوات الخبرة... إلخ).
- **التحقق من البيانات (Verification)**: كشف الحقول الناقصة أو القيم غير المنطقية (مثل عمر أو طول غير واقعي)، مع محاولة استكمال المعلومات الناقصة عبر أسئلة إضافية.
- **وكلاء تقييم متعددون (Scout Agents)**:
  - Experience Scout
  - Mentality Scout
  - Achievement Scout
  - Development Scout
  - Head Scout (التقييم النهائي والتوصية)
  - Explainability Agent (شرح أسباب التقييم)
- **تصدير التقارير**: إنشاء تقرير PDF منسّق بالكامل، وملف JSON يحتوي كل البيانات والتقارير.
- **واجهة API** (في `main.py`) لتشغيل المسار كاملاً واسترجاع التقارير.

## المتطلبات

- Python 3.9 فأعلى
- مفتاح API من [Groq](https://console.groq.com) (`GROQ_API_KEY`)

## التثبيت

```bash
pip install -r requirements.txt
```

## الإعداد

عيّن مفتاح Groq كمتغير بيئة قبل التشغيل (أو سيُطلب منك إدخاله عند التشغيل):

```bash
export GROQ_API_KEY="your_api_key_here"
```

يمكن أيضاً تحديد نموذج Groq المستخدم (اختياري، الافتراضي `llama-3.3-70b-versatile`):

```bash
export GROQ_MODEL="llama-3.3-70b-versatile"
```

## طريقة التشغيل

### 1) عبر FastAPI (`main.py`)

```bash
python main.py
```

سيعمل السيرفر على: `http://0.0.0.0:8000`

**أهم المسارات (Endpoints):**

| Method | Path | الوصف |
|---|---|---|
| GET | `/health` | فحص حالة الخدمة والنموذج والرياضات المدعومة |
| POST | `/scout` | تشغيل خط أنابيب الاستكشاف الكامل لرياضة معيّنة (يُجري المقابلة عبر الطرفية/الإدخال القياسي) |
| GET | `/reports` | عرض قائمة التقارير المُولّدة سابقاً |
| GET | `/report/{file_name}` | تنزيل تقرير PDF أو JSON محدد |

مثال على طلب `/scout`:

```bash
curl -X POST http://localhost:8000/scout \
  -H "Content-Type: application/json" \
  -d '{"sport": "Football", "player_name": "Ahmed"}'
```

> ملاحظة: المقابلة تتم عبر الإدخال القياسي (`input()`)، لذا تشغيل `/scout` يتطلب تفاعلاً من الطرفية التي يعمل عليها السيرفر.

### 2) عبر Jupyter Notebook (`AI_Scout.ipynb`)

```bash
jupyter notebook AI_Scout.ipynb
```

نفّذ الخلايا بالترتيب؛ ستُطالَب بإدخال إجابات المقابلة مباشرة داخل الـ Notebook.

## المخرجات

تُحفظ التقارير في مجلد `spotme_output/`:
- `<اسم_اللاعب>_scouting_report.pdf`
- `<اسم_اللاعب>_scouting_data.json`

## بنية المشروع

```
.
├── main.py              # تطبيق FastAPI الكامل
├── AI_Scout.ipynb       # نفس المنطق كنوتبوك تفاعلي
├── requirements.txt     # متطلبات التشغيل
└── spotme_output/        # مجلد التقارير الناتجة (يُنشأ تلقائياً)
```
