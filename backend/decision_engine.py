"""
Smart Co-Pilot — Hybrid Decision Engine
========================================
Mimari: Rule-Based (Phase 1) → ML Clustering (Phase 2) → Gemini XAI (Phase 3)
Çıktı: FastAPI-ready JSON — Dashboard ve gerçek donanımla plug-and-play uyumlu.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────
# 1. ENUM & DATACLASS TANIMLARI
# ──────────────────────────────────────────────

class AlertLevel(str, Enum):
    OK       = "OK"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class ScenarioType(str, Enum):
    NORMAL   = "NORMAL"
    HEATWAVE = "HEATWAVE"
    DROUGHT  = "DROUGHT"
    HARDWARE_FAULT = "HARDWARE_FAULT"


@dataclass
class SensorReading:
    """Ham sensör verisi — IoT cihazından veya simülasyondan gelir."""
    timestamp: str
    temperature: float          # °C
    humidity: float             # %
    water_level: float          # % (0-100)
    nitrogen: int               # ppm (N)
    phosphorus: int             # ppm (P)
    potassium: int              # ppm (K)
    fan_on: int = 0             # 0/1
    watering_pump_on: int = 0   # 0/1
    water_pump_on: int = 0      # 0/1


@dataclass
class ActuatorCommand:
    """Sistemin verdiği aktüatör komutu."""
    fan: bool = False
    watering_pump: bool = False
    water_pump: bool = False

    def to_dict(self):
        return {
            "fan": self.fan,
            "watering_pump": self.watering_pump,
            "water_pump": self.water_pump,
        }


@dataclass
class DecisionResult:
    """Decision Engine'in tam çıktısı — FastAPI'den dönen JSON budur."""
    timestamp: str
    alert_level: AlertLevel
    scenario: ScenarioType
    sensor_data: dict
    actuator_command: dict
    rule_based_flags: list[str]
    ml_cluster: Optional[int]
    ml_trend: Optional[dict]
    xai_explanation: Optional[str]
    literature_refs: list[str]
    user_action_required: bool
    confidence_score: float         # 0.0 – 1.0

    def to_json(self):
        d = asdict(self)
        d["alert_level"] = self.alert_level.value
        d["scenario"] = self.scenario.value
        return json.dumps(d, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# 2. PHASE 1 — RULE-BASED ENGINE
# ──────────────────────────────────────────────

# Bitki eşik değerleri (dikey tarım, yaprak yeşillikleri için genel aralıklar)
# Kaynak: Pereira et al. (2025), Kozai et al. (2022)
THRESHOLDS = {
    "temp_ok":          (15.0, 30.0),   # °C — optimal büyüme bandı
    "temp_warn_high":   30.0,           # °C — uyarı başlangıcı
    "temp_critical":    38.0,           # °C — kritik stres
    "temp_emergency":   42.0,           # °C — ölüm riski
    "temp_warn_low":    10.0,           # °C — soğuk stresi
    "temp_critical_low": 5.0,           # °C — dondurma riski

    "humidity_ok":      (50.0, 75.0),   # % — VPD optimumu
    "humidity_warn_low": 40.0,
    "humidity_critical_low": 20.0,
    "humidity_warn_high": 80.0,

    "water_level_ok":   50.0,           # % — minimum güvenli seviye
    "water_level_warn": 30.0,
    "water_level_critical": 10.0,
    "water_level_empty": 2.0,           # % — pompa koruma eşiği
}

LITERATURE = {
    "heatwave": [
        "Kozai et al. (2022) — Plant Factory: An Indoor Vertical Farming System, s.142",
        "Pereira et al. (2025) — Hybrid DSS for Smart Agriculture, MDPI Sensors",
        "ASHRAE (2021) — Thermal Guidelines for Controlled Environment Agriculture",
    ],
    "drought": [
        "Jones (2014) — Plants and Microclimate, Cambridge UP, s.88",
        "Pereira et al. (2025) — Hybrid DSS for Smart Agriculture, MDPI Sensors",
        "Savvas & Gruda (2018) — Hydroponic and Aeroponic Systems, Scientia Horticulturae",
    ],
    "hardware_fault": [
        "Gubbi et al. (2013) — IoT for Smart Cities: Vision & Challenges, Future Generation CS",
        "Pereira et al. (2025) — Hybrid DSS for Smart Agriculture, MDPI Sensors",
    ],
    "normal": [
        "Kozai et al. (2022) — Plant Factory: An Indoor Vertical Farming System",
    ],
}


def run_rule_based(reading: SensorReading) -> tuple[AlertLevel, ScenarioType, ActuatorCommand, list[str], float]:
    """
    Deterministik kural motoru.
    Dönüş: (alert_level, scenario, actuator_command, flags, confidence)
    """
    flags = []
    alert = AlertLevel.OK
    scenario = ScenarioType.NORMAL
    cmd = ActuatorCommand()
    confidence = 1.0  # Rule-based her zaman %100 güvenilir

    t  = reading.temperature
    h  = reading.humidity
    wl = reading.water_level

    # ── SICAKLIK KURALLARI ──────────────────────────
    if t >= THRESHOLDS["temp_emergency"]:
        flags.append(f"EMERGENCY: Sıcaklık {t}°C — Bitki ölüm riski!")
        alert = AlertLevel.EMERGENCY
        scenario = ScenarioType.HEATWAVE
        cmd.fan = True

    elif t >= THRESHOLDS["temp_critical"]:
        flags.append(f"CRITICAL: Sıcaklık {t}°C — Kritik ısı stresi")
        alert = AlertLevel.CRITICAL
        scenario = ScenarioType.HEATWAVE
        cmd.fan = True

    elif t >= THRESHOLDS["temp_warn_high"]:
        flags.append(f"WARNING: Sıcaklık {t}°C — Optimal aralık aşıldı (15-30°C)")
        alert = AlertLevel.WARNING
        cmd.fan = True

    elif t <= THRESHOLDS["temp_critical_low"]:
        flags.append(f"CRITICAL: Sıcaklık {t}°C — Dondurma riski!")
        alert = AlertLevel.CRITICAL
        scenario = ScenarioType.DROUGHT  # Soğuk-kuru senaryo

    elif t <= THRESHOLDS["temp_warn_low"]:
        flags.append(f"WARNING: Sıcaklık {t}°C — Soğuk stresi başlıyor")
        if alert == AlertLevel.OK:
            alert = AlertLevel.WARNING

    # ── NEM KURALLARI ───────────────────────────────
    if h <= THRESHOLDS["humidity_critical_low"]:
        flags.append(f"CRITICAL: Nem %{h} — Ciddi kuraklık stresi (VPD çok yüksek)")
        if alert.value < AlertLevel.CRITICAL.value:
            alert = AlertLevel.CRITICAL
        scenario = ScenarioType.DROUGHT
        cmd.watering_pump = True

    elif h <= THRESHOLDS["humidity_warn_low"]:
        flags.append(f"WARNING: Nem %{h} — Düşük nem, VPD optimal değil")
        if alert == AlertLevel.OK:
            alert = AlertLevel.WARNING
        cmd.watering_pump = True

    elif h >= THRESHOLDS["humidity_warn_high"]:
        flags.append(f"WARNING: Nem %{h} — Aşırı nem, hastalık riski")
        if alert == AlertLevel.OK:
            alert = AlertLevel.WARNING
        cmd.fan = True  # Fanla nem düşür

    # ── SU SEVİYESİ KURALLARI ───────────────────────
    if wl <= THRESHOLDS["water_level_empty"]:
        flags.append(f"EMERGENCY: Su seviyesi %{wl} — Pompa koruması devrede, sulama DURDU")
        alert = AlertLevel.EMERGENCY
        scenario = ScenarioType.DROUGHT
        cmd.watering_pump = False   # Kuru çalışmayı önle
        cmd.water_pump = False
        confidence = 1.0

    elif wl <= THRESHOLDS["water_level_critical"]:
        flags.append(f"CRITICAL: Su seviyesi %{wl} — Tank neredeyse boş")
        if alert.value not in [AlertLevel.EMERGENCY.value]:
            alert = AlertLevel.CRITICAL
        scenario = ScenarioType.DROUGHT
        cmd.water_pump = True   # Harici su kaynağından doldur

    elif wl <= THRESHOLDS["water_level_warn"]:
        flags.append(f"WARNING: Su seviyesi %{wl} — Yenileme önerisi")
        if alert == AlertLevel.OK:
            alert = AlertLevel.WARNING

    # ── DONANIM ARIZASI TESPİTİ ─────────────────────
    # Sulama pompası açık ama nem artmıyorsa → Arıza
    if reading.watering_pump_on == 1 and h <= THRESHOLDS["humidity_warn_low"]:
        flags.append("FAULT DETECTED: Sulama pompası açık ama nem artmıyor — Olası boru/pompa arızası!")
        scenario = ScenarioType.HARDWARE_FAULT
        if alert.value not in [AlertLevel.EMERGENCY.value]:
            alert = AlertLevel.CRITICAL
        confidence = 0.75  # Belirsizlik var

    if not flags:
        flags.append("OK: Tüm parametreler normal aralıkta")

    return alert, scenario, cmd, flags, confidence


# ──────────────────────────────────────────────
# 3. PHASE 2 — ML ENGINE (K-Means + Regresyon)
# ──────────────────────────────────────────────

class MLEngine:
    """
    K-Means ile operasyonel durum kümeleme.
    Linear Regression ile kısa vadeli trend tahmini.
    """
    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.regressors: dict[str, LinearRegression] = {}
        self.fitted = False
        self.cluster_labels = {
            0: "Optimal Koşullar",
            1: "Isı Stresi",
            2: "Kuraklık Stresi",
            3: "Kritik Durum",
        }

    def fit(self, df: pd.DataFrame):
        """Geçmiş veriyle modeli eğit."""
        features = df[["temperature", "humidity", "water_level"]].values
        scaled = self.scaler.fit_transform(features)
        self.kmeans.fit(scaled)

        # Her sensör için bağımsız trend regresörü
        for col in ["temperature", "humidity", "water_level"]:
            X = np.arange(len(df)).reshape(-1, 1)
            y = df[col].values
            reg = LinearRegression()
            reg.fit(X, y)
            self.regressors[col] = reg

        self.fitted = True
        self._assign_cluster_semantics(df)

    def _assign_cluster_semantics(self, df: pd.DataFrame):
        """Küme merkezlerini analiz ederek anlamlı isim ata."""
        features = df[["temperature", "humidity", "water_level"]].values
        scaled = self.scaler.transform(features)
        centers = self.scaler.inverse_transform(self.kmeans.cluster_centers_)

        # Merkezlere göre otomatik etiketleme
        for i, center in enumerate(centers):
            temp, hum, wl = center[0], center[1], center[2]
            if temp > 38:
                self.cluster_labels[i] = "Isı Stresi"
            elif hum < 30 or wl < 20:
                self.cluster_labels[i] = "Kuraklık Stresi"
            elif temp < 10:
                self.cluster_labels[i] = "Soğuk Stresi"
            elif 15 <= temp <= 30 and 50 <= hum <= 75 and wl >= 50:
                self.cluster_labels[i] = "Optimal Koşullar"
            else:
                self.cluster_labels[i] = "Stresli / Geçiş"

    def predict(self, reading: SensorReading, history_window: int = 10) -> dict:
        """Mevcut okuma için küme ve 30 dakikalık trend tahmin et."""
        if not self.fitted:
            return {"cluster": None, "cluster_label": "Model eğitilmedi", "trend": None}

        features = np.array([[reading.temperature, reading.humidity, reading.water_level]])
        scaled = self.scaler.transform(features)
        cluster_id = int(self.kmeans.predict(scaled)[0])

        # Trend: 30 dk sonra tahmini değerler (6 okuma ilerisi, 5 dk aralıklı varsayımı)
        trend = {}
        for col, reg in self.regressors.items():
            next_step = np.array([[history_window + 6]])
            trend[col] = round(float(reg.predict(next_step)[0]), 2)

        return {
            "cluster": cluster_id,
            "cluster_label": self.cluster_labels.get(cluster_id, "Bilinmiyor"),
            "trend_30min": trend,
        }


# ──────────────────────────────────────────────
# 4. PHASE 3 — GEMINI XAI KATMANI
# ──────────────────────────────────────────────

def build_gemini_prompt(reading: SensorReading, alert: AlertLevel,
                         scenario: ScenarioType, flags: list[str],
                         actuator: ActuatorCommand, trend: Optional[dict],
                         literature: list[str]) -> str:
    """Gemini API için yapılandırılmış Agronomist Persona prompt'u oluşturur."""

    refs_text = "\n".join(f"  - {r}" for r in literature)
    trend_text = json.dumps(trend, ensure_ascii=False) if trend else "Trend verisi mevcut değil"
    actuator_text = json.dumps(actuator.to_dict(), ensure_ascii=False)

    return f"""Sen, kentsel dikey tarım konusunda uzmanlaşmış bir agronomistsın (tarım bilimcisi).
Görevin, teknik sensör verilerini çiftçi/kullanıcı için açık, güven veren ve akademik temelli bir dille açıklamak.
Kullanıcı sistemin neden bu kararı aldığını anlamalı — sadece ne yapacağını değil.

## MEVCUT SENSÖR DURUMU
- Sıcaklık: {reading.temperature}°C
- Nem: {reading.humidity}%
- Su Seviyesi: {reading.water_level}%
- Azot (N): {reading.nitrogen} | Fosfor (P): {reading.phosphorus} | Potasyum (K): {reading.potassium}

## UYARI DURUMU: {alert.value}
## SENARYO: {scenario.value}

## TESPIT EDİLEN SORUNLAR
{chr(10).join(f'• {f}' for f in flags)}

## ÖNERİLEN AKTÜATÖR KOMUTLARI
{actuator_text}

## 30 DAKİKA SONRASI TAHMİN
{trend_text}

## KULLANILACAK LİTERATÜR
{refs_text}

## GÖREVİN
1. Mevcut durumu 2-3 cümleyle sade Türkçeyle açıkla (teknik jargon yok).
2. Neden bu aktüatör komutlarının verildiğini, yukarıdaki literatüre atıfla açıkla.
3. 30 dakikalık trende göre kullanıcının dikkat etmesi gereken bir sonraki adımı belirt.
4. Sonunda "⚠️ Kullanıcı Onayı Gerekli:" başlığıyla kısa bir özet sun.

Yanıtın maksimum 200 kelime olsun. Samimi, güven veren, ama asla panik yaratmayan bir ton kullan."""


async def call_gemini_xai(prompt: str, api_key: str) -> str:
    """
    Gemini 1.5 Flash API çağrısı.
    api_key: Ortam değişkeninden (GEMINI_API_KEY) okunur.
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except ImportError:
        return "[XAI] google-generativeai paketi yüklü değil. pip install google-generativeai"
    except Exception as e:
        return f"[XAI] Gemini API hatası: {str(e)}"


# ──────────────────────────────────────────────
# 5. ANA ORKESTRATÖR — HybridDecisionEngine
# ──────────────────────────────────────────────

class HybridDecisionEngine:
    """
    3 katmanlı karar motoru orkestratörü.

    Kullanım:
        engine = HybridDecisionEngine(gemini_api_key="...")
        engine.fit(historical_df)
        result = engine.decide(sensor_reading)
        print(result.to_json())
    """

    def __init__(self, gemini_api_key: Optional[str] = None, use_xai: bool = True):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self.use_xai = use_xai and bool(self.gemini_api_key)
        self.ml_engine = MLEngine(n_clusters=4)
        self._history: list[SensorReading] = []

    def fit(self, df: pd.DataFrame):
        """
        Geçmiş veriyle ML katmanını eğit.
        df kolonları: temperature, humidity, water_level (zorunlu)
        """
        # Kolon ismi uyumluluğu (orijinal datasetteki 'tempreature' typo'su)
        col_map = {"tempreature": "temperature"}
        df = df.rename(columns=col_map)

        required = ["temperature", "humidity", "water_level"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Eksik kolonlar: {missing}")

        self.ml_engine.fit(df.dropna(subset=required))
        print(f"[ML] Model {len(df)} satır veriyle eğitildi. Kümeler: {self.ml_engine.cluster_labels}")

    def decide(self, reading: SensorReading, xai_mode: bool = True) -> DecisionResult:
        """
        Senkron karar ver (Gemini XAI olmadan).
        Dashboard için hızlı yanıt (< 50ms).
        """
        self._history.append(reading)

        # PHASE 1: Rule-Based
        alert, scenario, actuator, flags, confidence = run_rule_based(reading)

        # PHASE 2: ML
        ml_result = self.ml_engine.predict(reading, len(self._history))

        # Literatür seçimi
        literature = LITERATURE.get(scenario.value.lower(), LITERATURE["normal"])

        # XAI prompt hazırla (asenkron çağrı için)
        xai_prompt = build_gemini_prompt(
            reading=reading,
            alert=alert,
            scenario=scenario,
            flags=flags,
            actuator=actuator,
            trend=ml_result.get("trend_30min"),
            literature=literature,
        )

        return DecisionResult(
            timestamp=reading.timestamp,
            alert_level=alert,
            scenario=scenario,
            sensor_data={
                "temperature": reading.temperature,
                "humidity": reading.humidity,
                "water_level": reading.water_level,
                "N": reading.nitrogen,
                "P": reading.phosphorus,
                "K": reading.potassium,
            },
            actuator_command=actuator.to_dict(),
            rule_based_flags=flags,
            ml_cluster=ml_result.get("cluster"),
            ml_trend=ml_result.get("trend_30min"),
            xai_explanation=None,   # async decide_with_xai() ile doldurulur
            literature_refs=literature,
            user_action_required=(alert in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]),
            confidence_score=confidence,
        ), xai_prompt  # prompt'u FastAPI'ye bırak, orada async çağır

    async def decide_with_xai(self, reading: SensorReading) -> DecisionResult:
        """XAI dahil tam karar (async). Dashboard detay paneli için."""
        result, xai_prompt = self.decide(reading)
        if self.use_xai:
            result.xai_explanation = await call_gemini_xai(xai_prompt, self.gemini_api_key)
        else:
            result.xai_explanation = "[XAI devre dışı — GEMINI_API_KEY ayarlanmamış]"
        return result


# ──────────────────────────────────────────────
# 6. VERİ YÜKLEYICI (Simülasyon & Test)
# ──────────────────────────────────────────────

def load_dataset(path: str) -> tuple[pd.DataFrame, list[SensorReading]]:
    """Excel/CSV dataseti SensorReading listesine dönüştür."""
    df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
    df = df.rename(columns={"tempreature": "temperature"})  # typo düzeltme

    readings = []
    for _, row in df.iterrows():
        r = SensorReading(
            timestamp=str(row.get("date", datetime.now().isoformat())),
            temperature=float(row.get("temperature", 25.0)),
            humidity=float(row.get("humidity", 60.0)),
            water_level=float(row.get("water_level", 80.0)),
            nitrogen=int(row.get("N", 150)),
            phosphorus=int(row.get("P", 50)),
            potassium=int(row.get("K", 200)),
            fan_on=int(row.get("Fan_actuator_ON", 0)),
            watering_pump_on=int(row.get("Watering_plant_pump_ON", 0)),
            water_pump_on=int(row.get("Water_pump_actuator_ON", 0)),
        )
        readings.append(r)
    return df, readings


# ──────────────────────────────────────────────
# 7. TEST — Simülasyon Çalıştırıcı
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    print("=" * 60)
    print("  Smart Co-Pilot — Hybrid Decision Engine TEST")
    print("=" * 60)

    # Veri yükle
    heatwave_df, heatwave_readings = load_dataset(
        "/mnt/user-data/uploads/synthetic_heatwave_scenario.xlsx"
    )
    drought_df, drought_readings = load_dataset(
        "/mnt/user-data/uploads/synthetic_drought_scenario.xlsx"
    )

    # Combined training data
    combined_df = pd.concat([heatwave_df, drought_df], ignore_index=True)
    combined_df = combined_df.rename(columns={"tempreature": "temperature"})

    # Engine oluştur ve eğit (XAI kapalı — API key yok)
    engine = HybridDecisionEngine(use_xai=False)
    engine.fit(combined_df)

    print("\n" + "─" * 60)
    print("SENARYO 1: HEATWAVE — İlk 3 okuma")
    print("─" * 60)
    for reading in heatwave_readings[:3]:
        result, _ = engine.decide(reading)
        print(f"\n[{result.timestamp}]")
        print(f"  Alert   : {result.alert_level.value}")
        print(f"  Senaryo : {result.scenario.value}")
        print(f"  Cluster : {result.ml_cluster} — {engine.ml_engine.cluster_labels.get(result.ml_cluster, '?')}")
        print(f"  Aktüatör: {result.actuator_command}")
        print(f"  Flags   : {result.rule_based_flags}")
        print(f"  Trend   : {result.ml_trend}")
        print(f"  Güven   : {result.confidence_score}")

    print("\n" + "─" * 60)
    print("SENARYO 2: DROUGHT — İlk 3 okuma")
    print("─" * 60)
    engine2 = HybridDecisionEngine(use_xai=False)
    engine2.fit(combined_df)
    for reading in drought_readings[:3]:
        result, _ = engine2.decide(reading)
        print(f"\n[{result.timestamp}]")
        print(f"  Alert   : {result.alert_level.value}")
        print(f"  Senaryo : {result.scenario.value}")
        print(f"  Cluster : {result.ml_cluster} — {engine2.ml_engine.cluster_labels.get(result.ml_cluster, '?')}")
        print(f"  Aktüatör: {result.actuator_command}")
        print(f"  Flags   : {result.rule_based_flags}")
        print(f"  Trend   : {result.ml_trend}")

    print("\n" + "─" * 60)
    print("SENARYO 3: DONANIM ARIZASI")
    print("─" * 60)
    fault_reading = SensorReading(
        timestamp=datetime.now().isoformat(),
        temperature=28.0,
        humidity=18.0,       # nem düşük
        water_level=65.0,
        nitrogen=150,
        phosphorus=50,
        potassium=200,
        fan_on=0,
        watering_pump_on=1,  # pompa açık ama nem artmıyor
        water_pump_on=0,
    )
    engine3 = HybridDecisionEngine(use_xai=False)
    engine3.fit(combined_df)
    result, prompt = engine3.decide(fault_reading)
    print(f"\n  Alert   : {result.alert_level.value}")
    print(f"  Senaryo : {result.scenario.value}")
    print(f"  Aktüatör: {result.actuator_command}")
    print(f"  Flags   : {result.rule_based_flags}")
    print(f"  Güven   : {result.confidence_score}")
    print(f"\n  [XAI PROMPT ÖRNEK — Gemini'ye gidecek metin]")
    print(f"  {prompt[:300]}...")

    print("\n" + "─" * 60)
    print("TAM JSON ÇIKTI ÖRNEĞİ (FastAPI response):")
    print("─" * 60)
    result.xai_explanation = "[Gemini XAI çıktısı buraya gelecek]"
    print(result.to_json())
