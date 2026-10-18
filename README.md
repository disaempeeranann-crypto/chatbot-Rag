# 🌦️ พยากรณ์อากาศ จ.พระนครศรีอยุธยา (ML + Streamlit)

พยากรณ์ล่วงหน้า 7 วันด้วย HistGradientBoosting: อุณหภูมิสูงสุด/ต่ำสุด, โอกาสฝน, ปริมาณฝนรายวันและฝนสะสม, PM2.5
ข้อมูลจาก [Open-Meteo](https://open-meteo.com) (Weather + Air Quality/CAMS) ไม่ต้องใช้ API key

## รันในเครื่อง
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ผ่าน GitHub + Streamlit Community Cloud
```bash
git init && git add . && git commit -m "Ayutthaya weather ML"
git branch -M main
git remote add origin https://github.com/<username>/<repo>.git
git push -u origin main
```
จากนั้นไปที่ https://share.streamlit.io → **Create app** → เลือก repo, branch `main`, ไฟล์ `app.py` → **Deploy**
