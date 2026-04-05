# Smart Co-Pilot — Decision Engine

## Kurulum

```bash
pip install fastapi uvicorn scikit-learn pandas openpyxl google-generativeai
```

## Proje Yapısı

```
decision_engine/
├── decision_engine.py   ← Karar motoru (Rule-Based + ML + XAI)
├── api.py               ← FastAPI gateway
├── data/
│   ├── synthetic_heatwave_scenario.xlsx
│   └── synthetic_drought_scenario.xlsx
└── README.md
```

## Çalıştırma

```bash
# 1. Veri dosyalarını data/ klasörüne koy
# 2. (İsteğe bağlı) Gemini API key'ini ayarla
export GEMINI_API_KEY="your-key-here"

# 3. Sunucuyu başlat
uvicorn api:app --reload --port 8000
```

## Endpoint'ler

| Method | Endpoint        | Açıklama                            | Süre     |
|--------|----------------|-------------------------------------|----------|
| GET    | /health        | Servis ve model durumu              | < 10ms   |
| GET    | /thresholds    | Bitki eşik değerleri + literatür    | < 10ms   |
| POST   | /decide        | Hızlı karar (XAI yok)              | < 50ms   |
| POST   | /decide/xai    | Tam karar + Gemini açıklaması       | ~1-3 sn  |
| POST   | /simulate      | Dataset simülasyonu                 | < 200ms  |

## Örnek İstek (Next.js / cURL)

```bash
curl -X POST http://localhost:8000/decide \
  -H "Content-Type: application/json" \
  -d '{"temperature": 41.5, "humidity": 22.0, "water_level": 75.0}'
```

## Örnek Yanıt

```json
{
  "timestamp": "2024-02-02T20:40:00",
  "alert_level": "CRITICAL",
  "scenario": "HEATWAVE",
  "actuator_command": {"fan": true, "watering_pump": false, "water_pump": false},
  "rule_based_flags": ["CRITICAL: Sıcaklık 41.5°C — Kritik ısı stresi"],
  "ml_cluster": 1,
  "ml_trend": {"temperature": 56.7, "humidity": 24.9, "water_level": 74.2},
  "user_action_required": true,
  "confidence_score": 1.0
}
```

## Mimari Karar Akışı

```
SensorReading
     │
     ▼
Phase 1: Rule-Based Engine    ← Deterministik, %100 güven, < 1ms
     │ (alert, scenario, flags, actuator)
     ▼
Phase 2: ML Engine            ← K-Means cluster + Trend regresyon
     │ (cluster_id, trend_30min)
     ▼
Phase 3: Gemini XAI           ← Doğal dil açıklama + literatür
     │ (xai_explanation)
     ▼
DecisionResult JSON           ← FastAPI → Next.js Dashboard
```

## Donanım Entegrasyonu (ESP32/Raspberry Pi)

Gerçek donanım geldiğinde sadece veri kaynağını değiştirin:

```python
# Simülasyon (şu an)
payload = load_from_excel("data/heatwave.xlsx")

# Gerçek donanım (ileride)
payload = mqtt_client.subscribe("farm/sensors")
```
API endpoint'leri aynı kalır.
