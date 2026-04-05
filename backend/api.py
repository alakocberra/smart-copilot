"""
Smart Co-Pilot — FastAPI Gateway
==================================
Endpoint'ler:
  POST /decide        → Hızlı karar (XAI yok, < 50ms)
  POST /decide/xai    → Tam karar (Gemini XAI dahil, async)
  POST /simulate      → Dataset'ten toplu simülasyon
  GET  /health        → Servis durumu
  GET  /thresholds    → Mevcut eşik değerleri
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from decision_engine import (
    HybridDecisionEngine,
    SensorReading,
    load_dataset,
    THRESHOLDS,
    LITERATURE,
)


# ── Başlangıç: Modeli Yükle ──────────────────
engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = HybridDecisionEngine(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        use_xai=bool(os.getenv("GEMINI_API_KEY")),
    )
    # Varsa geçmiş veriyle eğit
    try:
        df1 = pd.read_excel("data/synthetic_heatwave_scenario.xlsx")
        df2 = pd.read_excel("data/synthetic_drought_scenario.xlsx")
        combined = pd.concat([df1, df2], ignore_index=True)
        engine.fit(combined)
        print("[Startup] ML modeli eğitildi.")
    except Exception as e:
        print(f"[Startup] Eğitim verisi yüklenemedi: {e} — Engine hazır ama ML pasif.")
    yield
    print("[Shutdown] Engine kapatıldı.")


app = FastAPI(
    title="Smart Co-Pilot Decision Engine",
    description="IoT tabanlı Akıllı Tarım Karar Destek Sistemi — Hybrid Rule-Based + ML + XAI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Production'da Next.js domain'ini yaz
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Modelleri ───────────────

class SensorPayload(BaseModel):
    """IoT cihazından veya simülasyondan gelen sensör verisi."""
    timestamp: Optional[str] = None
    temperature: float
    humidity: float
    water_level: float
    nitrogen: int = 150
    phosphorus: int = 50
    potassium: int = 200
    fan_on: int = 0
    watering_pump_on: int = 0
    water_pump_on: int = 0

    model_config = {"json_schema_extra": {
        "example": {
            "temperature": 41.5,
            "humidity": 22.0,
            "water_level": 75.0,
            "nitrogen": 255,
            "phosphorus": 255,
            "potassium": 255,
        }
    }}


# ── Endpoints ────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "ml_fitted": engine.ml_engine.fitted if engine else False,
        "xai_enabled": engine.use_xai if engine else False,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/thresholds")
def get_thresholds():
    return {"thresholds": THRESHOLDS, "literature": LITERATURE}


@app.post("/decide")
def decide_fast(payload: SensorPayload):
    """
    Hızlı karar endpoint'i (< 50ms).
    XAI açıklaması yok — dashboard ana kartları için.
    """
    if not engine:
        raise HTTPException(503, "Engine başlatılmamış")

    reading = SensorReading(
        timestamp=payload.timestamp or datetime.now().isoformat(),
        temperature=payload.temperature,
        humidity=payload.humidity,
        water_level=payload.water_level,
        nitrogen=payload.nitrogen,
        phosphorus=payload.phosphorus,
        potassium=payload.potassium,
        fan_on=payload.fan_on,
        watering_pump_on=payload.watering_pump_on,
        water_pump_on=payload.water_pump_on,
    )

    result, _ = engine.decide(reading)
    return {
        "timestamp": result.timestamp,
        "alert_level": result.alert_level.value,
        "scenario": result.scenario.value,
        "sensor_data": result.sensor_data,
        "actuator_command": result.actuator_command,
        "rule_based_flags": result.rule_based_flags,
        "ml_cluster": result.ml_cluster,
        "ml_trend": result.ml_trend,
        "literature_refs": result.literature_refs,
        "user_action_required": result.user_action_required,
        "confidence_score": result.confidence_score,
    }


@app.post("/decide/xai")
async def decide_with_xai(payload: SensorPayload):
    """
    XAI dahil tam karar (Gemini API çağrısı — ~1-3 sn).
    Dashboard detay paneli / "Neden?" butonu için.
    """
    if not engine:
        raise HTTPException(503, "Engine başlatılmamış")

    reading = SensorReading(
        timestamp=payload.timestamp or datetime.now().isoformat(),
        temperature=payload.temperature,
        humidity=payload.humidity,
        water_level=payload.water_level,
        nitrogen=payload.nitrogen,
        phosphorus=payload.phosphorus,
        potassium=payload.potassium,
        fan_on=payload.fan_on,
        watering_pump_on=payload.watering_pump_on,
        water_pump_on=payload.water_pump_on,
    )

    result = await engine.decide_with_xai(reading)
    import json, dataclasses
    return json.loads(result.to_json())


@app.post("/simulate")
def simulate_scenario(scenario: str = "heatwave", rows: int = 10):
    """
    Mevcut datasetten satır çekip simüle et.
    scenario: 'heatwave' | 'drought'
    """
    paths = {
        "heatwave": "data/synthetic_heatwave_scenario.xlsx",
        "drought":  "data/synthetic_drought_scenario.xlsx",
    }
    if scenario not in paths:
        raise HTTPException(400, f"Geçersiz senaryo. Seçenekler: {list(paths.keys())}")

    try:
        _, readings = load_dataset(paths[scenario])
    except FileNotFoundError:
        raise HTTPException(404, "Veri dosyası bulunamadı")

    results = []
    sim_engine = HybridDecisionEngine(use_xai=False)
    # Training için her iki dataseti yükle
    try:
        df1 = pd.read_excel("data/synthetic_heatwave_scenario.xlsx")
        df2 = pd.read_excel("data/synthetic_drought_scenario.xlsx")
        sim_engine.fit(pd.concat([df1, df2], ignore_index=True))
    except Exception:
        pass

    for reading in readings[:min(rows, len(readings))]:
        result, _ = sim_engine.decide(reading)
        results.append({
            "timestamp": result.timestamp,
            "alert_level": result.alert_level.value,
            "scenario": result.scenario.value,
            "flags": result.rule_based_flags,
            "actuator": result.actuator_command,
            "cluster": result.ml_cluster,
        })

    return {"scenario": scenario, "total_rows": len(results), "results": results}


# ── Çalıştır ─────────────────────────────────
# uvicorn api:app --reload --port 8000
