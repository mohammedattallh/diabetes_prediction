# استخدام صورة Python 3.9 slim كالصورة الأساسية
FROM python:3.9-slim

# تعيين مجلد العمل داخل الحاوية
WORKDIR /app

# نسخ جميع الملفات من المجلد المحلي إلى الحاوية
COPY . /app/

# تثبيت الحزم اللازمة بما في ذلك scikit-learn
RUN pip install --no-cache-dir fastapi uvicorn joblib numpy pydantic nest-asyncio scikit-learn

# تحديد الأمر الذي سيتم تنفيذه عند تشغيل الحاوية
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8030"]
