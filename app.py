"""
พยากรณ์อากาศ จ.พระนครศรีอยุธยา ด้วย Machine Learning (Streamlit)
- ข้อมูล: Open-Meteo (Weather Archive/Forecast + Air Quality API จาก CAMS)
- โมเดล: HistGradientBoosting (direct multi-horizon 1-7 วัน)
- ผลลัพธ์: อุณหภูมิสูงสุด/ต่ำสุด, โอกาสฝน, ปริมาณฝนรายวัน + สะสม, PM2.5
"""
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

st.set_page_config(page_title="พยากรณ์อากาศ อยุธยา", page_icon="🌦️", layout="wide")

# พิกัดโดยประมาณของแต่ละอำเภอ (ข้อมูลโมเดลสภาพอากาศมีความละเอียดราว 10 กม.)
DISTRICTS = {
    "อ.พระนครศรีอยุธยา (ในเมือง)": (14.3532, 100.5689),
    "อ.บางปะอิน": (14.2300, 100.5800),
    "อ.วังน้อย": (14.2000, 100.7170),
    "อ.เสนา": (14.3200, 100.4200),
    "อ.บางไทร": (14.1833, 100.4833),
    "อ.ท่าเรือ": (14.5500, 100.7170),
    "อ.ผักไห่": (14.4500, 100.3500),
    "อ.ภาชี": (14.4333, 100.7333),
    "อ.นครหลวง": (14.5500, 100.6200),
    "อ.บางปะหัน": (14.4500, 100.5800),
}
WX = [
    "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
    "wind_speed_10m_max", "relative_humidity_2m_mean", "surface_pressure_mean",
]
ALL_VARS = WX + ["pm25"]
HORIZONS = range(1, 8)
RAIN_MM = 1.0
AQ_START = date(2022, 8, 1)  # CAMS global มีข้อมูลย้อนหลังตั้งแต่กลางปี 2022
TZ = ZoneInfo("Asia/Bangkok")

REG_TARGETS = ["tmax", "tmin", "precip", "pm25"]
BASELINE_COL = {
    "tmax": "temperature_2m_max_lag0", "tmin": "temperature_2m_min_lag0",
    "precip": "precipitation_sum_lag0", "pm25": "pm25_lag0",
}
LABEL = {"tmax": "อุณหภูมิสูงสุด (°C)", "tmin": "อุณหภูมิต่ำสุด (°C)",
         "precip": "ฝนรายวัน (มม.)", "pm25": "PM2.5 (µg/m³)"}


def today_th() -> date:
    return datetime.now(TZ).date()  # Streamlit Cloud ใช้เวลา UTC จึงต้องระบุเขตเวลา


# ---------- ดึงข้อมูล ----------
def _weather_df(js: dict) -> pd.DataFrame:
    df = pd.DataFrame(js["daily"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time").sort_index()


def _pm25_daily(js: dict) -> pd.Series:
    h = pd.DataFrame(js["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    h = h.set_index("time")["pm2_5"]
    g = h.resample("D")
    return g.mean().where(g.count() >= 18).rename("pm25")  # ต้องมีข้อมูลอย่างน้อย 18 ชม./วัน


def _empty_pm() -> pd.Series:
    return pd.Series(dtype=float, name="pm25", index=pd.DatetimeIndex([]))


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_history(lat: float, lon: float, years: int) -> pd.DataFrame:
    end = today_th() - timedelta(days=7)
    start = end - timedelta(days=365 * years)
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={"latitude": lat, "longitude": lon, "start_date": start.isoformat(),
                "end_date": end.isoformat(), "daily": ",".join(WX), "timezone": "Asia/Bangkok"},
        timeout=60,
    )
    r.raise_for_status()
    df = _weather_df(r.json())
    try:  # PM2.5 ล้มเหลวได้โดยแอปยังทำงานต่อ
        ra = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={"latitude": lat, "longitude": lon, "hourly": "pm2_5",
                    "start_date": max(start, AQ_START).isoformat(), "end_date": end.isoformat(),
                    "timezone": "Asia/Bangkok"},
            timeout=60,
        )
        ra.raise_for_status()
        pm = _pm25_daily(ra.json())
    except Exception:
        pm = _empty_pm()
    return df.join(pm)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_recent(lat: float, lon: float):
    """คืน (ข้อมูลย้อนหลัง 14 วัน, ค่า PM2.5 จาก CAMS รวมวันข้างหน้าไว้อ้างอิง)"""
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon, "daily": ",".join(WX),
                "past_days": 14, "forecast_days": 1, "timezone": "Asia/Bangkok"},
        timeout=60,
    )
    r.raise_for_status()
    df = _weather_df(r.json())
    df = df[df.index < pd.Timestamp(today_th())]
    try:
        ra = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={"latitude": lat, "longitude": lon, "hourly": "pm2_5",
                    "past_days": 14, "forecast_days": 5, "timezone": "Asia/Bangkok"},
            timeout=60,
        )
        ra.raise_for_status()
        aq = _pm25_daily(ra.json())
    except Exception:
        aq = _empty_pm()
    df = df.join(aq[aq.index < pd.Timestamp(today_th())])
    if "pm25" not in df:
        df["pm25"] = np.nan
    return df, aq


# ---------- Feature engineering ----------
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df[ALL_VARS].copy().ffill(limit=3)
    f = {}
    for c in ALL_VARS:
        for lag in range(7):
            f[f"{c}_lag{lag}"] = df[c].shift(lag)
    for c in ["temperature_2m_max", "precipitation_sum", "relative_humidity_2m_mean", "pm25"]:
        f[f"{c}_roll3"] = df[c].rolling(3, min_periods=2).mean()
        f[f"{c}_roll7"] = df[c].rolling(7, min_periods=4).mean()
    f["pressure_change"] = df["surface_pressure_mean"].diff()
    doy = df.index.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return pd.DataFrame(f, index=df.index)


def make_targets(df: pd.DataFrame, h: int) -> pd.DataFrame:
    p = df["precipitation_sum"].shift(-h)
    return pd.DataFrame({
        "tmax": df["temperature_2m_max"].shift(-h),
        "tmin": df["temperature_2m_min"].shift(-h),
        "precip": p,
        "rain": (p >= RAIN_MM).astype(float).where(p.notna()),
        "pm25": df["pm25"].shift(-h),
    }, index=df.index)


def wx_cols(cols):
    """คอลัมน์ฟีเจอร์ที่ต้องไม่ว่าง (PM2.5 ปล่อยว่างได้ เพราะโมเดลรองรับ NaN)"""
    return [c for c in cols if not c.startswith("pm25")]


def _reg():
    return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.06, max_depth=5, random_state=42)


def _clf():
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, max_depth=4, random_state=42)


def _fit(t, d, cols):
    d = d[d[t].notna()]
    if len(d) < 200:
        return None
    if t == "rain":
        return _clf().fit(d[cols], d[t].astype(int))
    return _reg().fit(d[cols], d[t])


# ---------- เทรนและประเมินผล ----------
@st.cache_resource(show_spinner=False)
def train_models(lat: float, lon: float, years: int):
    hist = fetch_history(lat, lon, years)
    X_all = make_features(hist)
    cols = list(X_all.columns)
    models, metrics = {}, []

    for h in HORIZONS:
        data = pd.concat([X_all, make_targets(hist, h)], axis=1).dropna(subset=wx_cols(cols))
        split = int(len(data) * 0.8)  # แบ่งตามเวลา ไม่สุ่ม เพื่อไม่ให้ข้อมูลอนาคตรั่ว
        tr, te = data.iloc[:split], data.iloc[split:]
        row = {"วันล่วงหน้า": h}

        for t in REG_TARGETS:
            m = _fit(t, tr, cols)
            sub = te[te[t].notna() & te[BASELINE_COL[t]].notna()]
            if m is not None and len(sub):
                row[f"MAE {t}"] = mean_absolute_error(sub[t], m.predict(sub[cols]))
                row[f"Base {t}"] = mean_absolute_error(sub[t], sub[BASELINE_COL[t]])
            else:
                row[f"MAE {t}"] = row[f"Base {t}"] = np.nan
        c = _fit("rain", tr, cols)
        sub = te[te["rain"].notna()]
        if c is not None and len(sub):
            y = sub["rain"].astype(int)
            row["Acc ฝน"] = accuracy_score(y, c.predict(sub[cols]))
            row["Base ฝน"] = max(y.mean(), 1 - y.mean())

        metrics.append(row)
        models[h] = {t: _fit(t, data, cols) for t in REG_TARGETS + ["rain"]}  # เทรนซ้ำด้วยข้อมูลทั้งหมด

    return models, pd.DataFrame(metrics).set_index("วันล่วงหน้า"), len(hist), cols


def predict(models, cols, recent: pd.DataFrame, aq: pd.Series):
    X = make_features(recent)[cols].dropna(subset=wx_cols(cols)).iloc[[-1]]
    last_day = X.index[0]
    rows = []
    for h in HORIZONS:
        m = models[h]
        d = pd.Timestamp(last_day) + timedelta(days=h)
        rows.append({
            "วันที่": d.date(),
            "สูงสุด (°C)": float(m["tmax"].predict(X)[0]),
            "ต่ำสุด (°C)": float(m["tmin"].predict(X)[0]),
            "โอกาสฝน (%)": float(m["rain"].predict_proba(X)[0][1] * 100),
            "ฝนคาดการณ์ (มม.)": max(0.0, float(m["precip"].predict(X)[0])),
            "PM2.5 (µg/m³)": max(0.0, float(m["pm25"].predict(X)[0])) if m["pm25"] else np.nan,
            "PM2.5 CAMS (อ้างอิง)": float(aq.get(d, np.nan)),
        })
    out = pd.DataFrame(rows)
    out["ฝนสะสมพยากรณ์ (มม.)"] = out["ฝนคาดการณ์ (มม.)"].cumsum()
    return out, last_day


# ---------- ฝนสะสม ----------
def window_sum(s: pd.Series, start, end) -> float:
    n = (end - start).days + 1
    w = s.loc[start:end]
    return float(w.sum()) if w.notna().sum() >= 0.95 * n else np.nan


def season_start_year(today: date) -> int:
    return today.year if today > date(today.year, 5, 1) else today.year - 1  # ฤดูฝนเริ่ม 1 พ.ค.


def rain_stats(precip: pd.Series, today: date) -> dict:
    end = pd.Timestamp(today) - pd.Timedelta(days=1)
    windows = {
        "7 วันที่ผ่านมา": end - pd.Timedelta(days=6),
        "30 วันที่ผ่านมา": end - pd.Timedelta(days=29),
        "ตั้งแต่ 1 พ.ค. (ฤดูฝน)": pd.Timestamp(season_start_year(today), 5, 1),
    }
    out = {}
    for name, start in windows.items():
        cur = window_sum(precip, start, end)
        normals = []
        for k in range(1, 11):
            s2, e2 = start - pd.DateOffset(years=k), end - pd.DateOffset(years=k)
            if s2 < precip.index.min():
                break
            v = window_sum(precip, s2, e2)
            if not np.isnan(v):
                normals.append(v)
        out[name] = (cur, float(np.mean(normals)) if normals else np.nan)
    return out


def season_cumulative(precip: pd.Series, today: date) -> pd.DataFrame:
    y0 = season_start_year(today)
    end = pd.Timestamp(today) - pd.Timedelta(days=1)
    start = pd.Timestamp(y0, 5, 1)
    n = (end - start).days + 1
    if n < 2:
        return pd.DataFrame()

    def cum(year):
        s = precip.loc[pd.Timestamp(year, 5, 1): pd.Timestamp(year, 5, 1) + pd.Timedelta(days=n - 1)]
        return s.cumsum().reset_index(drop=True) if len(s) == n and s.notna().all() else None

    cur = cum(y0)
    past = [c for c in (cum(y) for y in range(y0 - 10, y0)) if c is not None]
    if cur is None:
        return pd.DataFrame()
    df = pd.DataFrame({f"ปี {y0}": cur})
    if past:
        df["ค่าเฉลี่ยในอดีต"] = pd.concat(past, axis=1).mean(axis=1)
    df.index = df.index + 1
    df.index.name = "วันนับจาก 1 พ.ค."
    return df


def pm_level(v: float) -> str:
    """เกณฑ์ดัชนีคุณภาพอากาศ PM2.5 เฉลี่ย 24 ชม. ของกรมควบคุมมลพิษ (ปรับปรุง 2566)"""
    if v is None or np.isnan(v):
        return "ไม่มีข้อมูล"
    if v <= 15:
        return "🔵 ดีมาก"
    if v <= 25:
        return "🟢 ดี"
    if v <= 37.5:
        return "🟡 ปานกลาง"
    if v <= 75:
        return "🟠 เริ่มมีผลกระทบต่อสุขภาพ"
    return "🔴 มีผลกระทบต่อสุขภาพ"


# ---------- UI ----------
st.title("🌦️ พยากรณ์อากาศ จ.พระนครศรีอยุธยา")
st.caption("ข้อมูลจาก Open-Meteo • โมเดล Gradient Boosting • พยากรณ์ล่วงหน้า 7 วัน รวมปริมาณฝนสะสมและ PM2.5")

with st.sidebar:
    st.header("ตั้งค่า")
    place = st.selectbox("เลือกอำเภอ", list(DISTRICTS))
    years = st.slider("ข้อมูลย้อนหลัง (ปี)", 3, 10, 5)
    st.info("การเทรนครั้งแรกใช้เวลาประมาณ 30–90 วินาที หลังจากนั้นจะถูกแคชไว้")
    st.caption("ข้อมูล PM2.5 ย้อนหลังมีตั้งแต่ ส.ค. 2565")

lat, lon = DISTRICTS[place]
today = today_th()

try:
    with st.spinner("กำลังดึงข้อมูลและเทรนโมเดล..."):
        models, metrics, n_rows, cols = train_models(lat, lon, years)
        hist = fetch_history(lat, lon, years)
        recent, aq = fetch_recent(lat, lon)
        forecast, last_day = predict(models, cols, recent, aq)
except requests.RequestException as e:
    st.error(f"ดึงข้อมูลจาก Open-Meteo ไม่สำเร็จ: {e}")
    st.stop()

precip = pd.concat([hist["precipitation_sum"], recent["precipitation_sum"]])
precip = precip[~precip.index.duplicated(keep="last")].sort_index()
pm_all = pd.concat([hist["pm25"], recent["pm25"]])
pm_all = pm_all[~pm_all.index.duplicated(keep="last")].sort_index().dropna()

t0 = forecast.iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("สูงสุดวันนี้", f"{t0['สูงสุด (°C)']:.1f} °C")
c2.metric("ต่ำสุดวันนี้", f"{t0['ต่ำสุด (°C)']:.1f} °C")
c3.metric("โอกาสฝน / ปริมาณ", f"{t0['โอกาสฝน (%)']:.0f} %", f"{t0['ฝนคาดการณ์ (มม.)']:.1f} มม.", delta_color="off")
c4.metric("PM2.5 วันนี้", "-" if np.isnan(t0["PM2.5 (µg/m³)"]) else f"{t0['PM2.5 (µg/m³)']:.0f} µg/m³",
          pm_level(t0["PM2.5 (µg/m³)"]), delta_color="off")

tab1, tab2, tab3, tab4 = st.tabs(["📅 พยากรณ์ 7 วัน", "🌧️ ฝนสะสม", "😷 PM2.5", "📊 ประสิทธิภาพโมเดล"])

with tab1:
    show = forecast[["วันที่", "สูงสุด (°C)", "ต่ำสุด (°C)", "โอกาสฝน (%)", "ฝนคาดการณ์ (มม.)",
                     "ฝนสะสมพยากรณ์ (มม.)", "PM2.5 (µg/m³)"]]
    st.dataframe(show.style.format({c: "{:.1f}" for c in show.columns if c != "วันที่"}),
                 use_container_width=True, hide_index=True)
    ch = forecast.set_index("วันที่")
    a, b = st.columns(2)
    a.markdown("**อุณหภูมิ (°C)**")
    a.line_chart(ch[["สูงสุด (°C)", "ต่ำสุด (°C)"]])
    b.markdown("**โอกาสฝนตก (%)**")
    b.bar_chart(ch[["โอกาสฝน (%)"]])

with tab2:
    st.subheader("ปริมาณฝนสะสมที่ตรวจวัดแล้ว (เทียบค่าเฉลี่ยช่วงเดียวกันในอดีต)")
    stats = rain_stats(precip, today)
    cols_ = st.columns(len(stats))
    for col, (name, (cur, normal)) in zip(cols_, stats.items()):
        if np.isnan(cur):
            col.metric(name, "-")
        elif np.isnan(normal) or normal == 0:
            col.metric(name, f"{cur:.0f} มม.")
        else:
            col.metric(name, f"{cur:.0f} มม.", f"{cur / normal * 100:.0f}% ของค่าเฉลี่ย ({normal:.0f} มม.)",
                       delta_color="off")
    st.caption("คำนวณจากข้อมูลจุดกริดของ Open-Meteo ไม่ใช่เครื่องวัดฝนจริง อาจต่างจากสถานีของกรมอุตุนิยมวิทยาหรือกรมชลประทาน")

    l, r = st.columns(2)
    l.markdown("**ฝนรายวัน 30 วันล่าสุด (มม.)**")
    l.bar_chart(precip.iloc[-30:].rename("ฝน (มม.)"))
    sc = season_cumulative(precip, today)
    r.markdown("**ฝนสะสมในฤดูฝน เทียบค่าเฉลี่ย (มม.)**")
    if sc.empty:
        r.info("ยังไม่มีข้อมูลเพียงพอ (ฤดูฝนเริ่ม 1 พ.ค.)")
    else:
        r.line_chart(sc)

    st.subheader("ฝนสะสมที่พยากรณ์ 7 วันข้างหน้า")
    st.line_chart(forecast.set_index("วันที่")[["ฝนสะสมพยากรณ์ (มม.)"]])
    st.metric("คาดว่าฝนรวม 7 วัน", f"{forecast['ฝนคาดการณ์ (มม.)'].sum():.0f} มม.")
    st.caption("ปริมาณฝนรายวันเป็นค่าคาดหมายเฉลี่ย โดยทั่วไปทำนายได้ยากกว่า 'ฝนตกหรือไม่' โปรดดู MAE ในแท็บประสิทธิภาพ")

with tab3:
    if forecast["PM2.5 (µg/m³)"].isna().all():
        st.warning("ไม่สามารถพยากรณ์ PM2.5 ได้ (ดึงข้อมูลคุณภาพอากาศไม่สำเร็จหรือข้อมูลไม่พอ)")
    else:
        pm = forecast[["วันที่", "PM2.5 (µg/m³)", "PM2.5 CAMS (อ้างอิง)"]].copy()
        pm["ระดับ (เกณฑ์ กรมควบคุมมลพิษ)"] = pm["PM2.5 (µg/m³)"].map(pm_level)
        st.dataframe(pm.style.format({"PM2.5 (µg/m³)": "{:.1f}", "PM2.5 CAMS (อ้างอิง)": "{:.1f}"}),
                     use_container_width=True, hide_index=True)
        st.markdown("**PM2.5 พยากรณ์เทียบ CAMS (µg/m³)**")
        st.line_chart(pm.set_index("วันที่")[["PM2.5 (µg/m³)", "PM2.5 CAMS (อ้างอิง)"]])
        st.markdown("**PM2.5 เฉลี่ยรายวัน 90 วันล่าสุด (µg/m³)**")
        st.line_chart(pm_all.iloc[-90:].rename("PM2.5"))
    st.caption(
        "ค่า PM2.5 มาจากโมเดลบรรยากาศ CAMS (กริดหลายกิโลเมตร) จึงอาจต่ำ/สูงกว่าสถานีตรวจวัดจริง "
        "ปลายปีถึงต้นปีมักมีฝุ่นสูงกว่าช่วงฤดูฝน ตรวจสอบค่าทางการได้ที่ Air4Thai (กรมควบคุมมลพิษ)"
    )

with tab4:
    st.write(f"ใช้ข้อมูลฝึก {n_rows:,} วัน • ทดสอบกับข้อมูล 20% ล่าสุด • Base = ใช้ค่าวันนี้เป็นคำตอบของวันข้างหน้า "
             "(ฝน: ทายคลาสที่พบบ่อยสุด) • MAE ยิ่งน้อยยิ่งดี")
    st.dataframe(metrics.style.format("{:.2f}"), use_container_width=True)
    st.caption("ถ้า MAE ของโมเดลไม่ต่ำกว่า Base แปลว่าโมเดลยังไม่ช่วยอะไรสำหรับวันล่วงหน้านั้น")

st.caption(f"ข้อมูลล่าสุดที่ใช้พยากรณ์: {last_day.date()} • โปรเจกต์ตัวอย่างเพื่อการเรียนรู้ "
           "ไม่ควรใช้แทนการพยากรณ์/เตือนภัยอย่างเป็นทางการ")
