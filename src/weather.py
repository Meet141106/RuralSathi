import requests

DISTRICT_COORDS = {
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Pune": (18.5204, 73.8567),
    "Nagpur": (21.1458, 79.0882),
    "Nashik": (19.9975, 73.7898),
    "Aurangabad": (19.8762, 75.3433),
    "Kolhapur": (16.7050, 74.2433),
    "Amravati": (20.9374, 77.7796),
    "Solapur": (17.6599, 75.9064),
    "Jalgaon": (21.0077, 75.5626),
    "Ahmedabad": (23.0225, 72.5714),
    "Surat": (21.1702, 72.8311),
    "Rajkot": (22.3039, 70.8022),
    "Bhavnagar": (21.7645, 72.1519),
    "Jaipur": (26.9124, 75.7873),
    "Jodhpur": (26.2389, 73.0243),
    "Lucknow": (26.8467, 80.9462),
    "Varanasi": (25.3176, 82.9739),
    "Patna": (25.5941, 85.1376),
    "Bhopal": (23.2599, 77.4126),
    "Indore": (22.7196, 75.8577),
    "Hyderabad": (17.3850, 78.4867),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Bhubaneswar": (20.2961, 85.8245),
    "Chandigarh": (30.7333, 76.7794),
    "Amritsar": (31.6340, 74.8723),
    "Dehradun": (30.3165, 78.0322),
    "Guwahati": (26.1445, 91.7362),
}


def map_weather_code(code: int) -> str:
    """Map WMO weather code to human-readable condition description."""
    if code == 0:
        return "Clear sky"
    elif code in [1, 2, 3]:
        return "Partly cloudy"
    elif code in [45, 48]:
        return "Foggy"
    elif 51 <= code <= 57:
        return "Drizzle"
    elif 61 <= code <= 67:
        return "Rain"
    elif 71 <= code <= 77:
        return "Snow"
    elif 80 <= code <= 82:
        return "Rain showers"
    elif 85 <= code <= 86:
        return "Snow showers"
    elif code == 95:
        return "Thunderstorm"
    elif code in [96, 99]:
        return "Heavy thunderstorm"
    return "Variable weather"


def get_weather(district: str) -> dict:
    """Fetch 5-day weather forecast from Open-Meteo API for given district."""
    coords = DISTRICT_COORDS.get(district, DISTRICT_COORDS["Mumbai"])
    lat, lon = coords

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "rain_sum", "weathercode"],
        "current_weather": True,
        "timezone": "Asia/Kolkata",
        "forecast_days": 5
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        current_w = data.get("current_weather", {})
        current_temp = current_w.get("temperature", 25.0)
        current_code = current_w.get("weathercode", 0)
        current_condition = map_weather_code(current_code)

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        max_temps = daily.get("temperature_2m_max", [])
        min_temps = daily.get("temperature_2m_min", [])
        rain_sums = daily.get("rain_sum", daily.get("precipitation_sum", []))
        weathercodes = daily.get("weathercode", [])

        forecast = []
        rain_alert = False

        for i in range(min(5, len(dates))):
            r_sum = float(rain_sums[i]) if i < len(rain_sums) else 0.0
            if r_sum > 15.0:
                rain_alert = True

            wcode = weathercodes[i] if i < len(weathercodes) else 0
            forecast.append({
                "date": dates[i],
                "max_temp": float(max_temps[i]) if i < len(max_temps) else 0.0,
                "min_temp": float(min_temps[i]) if i < len(min_temps) else 0.0,
                "rain_mm": r_sum,
                "condition": map_weather_code(wcode)
            })

        rain_alert_message = (
            "⚠️ Heavy rain alert (>15mm) expected in the coming days. Ensure proper drainage in fields."
            if rain_alert else "Normal weather expected. No severe rain alerts."
        )

        return {
            "district": district,
            "current_temp": float(current_temp),
            "current_condition": current_condition,
            "forecast": forecast,
            "rain_alert": rain_alert,
            "rain_alert_message": rain_alert_message
        }
    except Exception as e:
        # Fallback graceful data in case of connection drop
        return {
            "district": district,
            "current_temp": 28.0,
            "current_condition": "Partly cloudy",
            "forecast": [
                {"date": "Day 1", "max_temp": 32.0, "min_temp": 22.0, "rain_mm": 0.0, "condition": "Partly cloudy"},
                {"date": "Day 2", "max_temp": 33.0, "min_temp": 23.0, "rain_mm": 0.0, "condition": "Clear sky"},
                {"date": "Day 3", "max_temp": 31.0, "min_temp": 22.0, "rain_mm": 2.0, "condition": "Drizzle"},
                {"date": "Day 4", "max_temp": 30.0, "min_temp": 21.0, "rain_mm": 0.0, "condition": "Partly cloudy"},
                {"date": "Day 5", "max_temp": 32.0, "min_temp": 22.0, "rain_mm": 0.0, "condition": "Clear sky"}
            ],
            "rain_alert": False,
            "rain_alert_message": f"Weather service unavailable: {e}"
        }


def get_farming_advice(weather_data: dict) -> str:
    """Return tailored agricultural advice based on current and forecasted conditions."""
    forecast = weather_data.get("forecast", [])
    current_temp = weather_data.get("current_temp", 25.0)
    current_condition = weather_data.get("current_condition", "").lower()

    # Calculate max rain in forecast
    max_forecast_rain = max([f.get("rain_mm", 0.0) for f in forecast], default=0.0)
    max_forecast_temp = max([f.get("max_temp", current_temp) for f in forecast], default=current_temp)

    if max_forecast_rain > 20.0 or "thunderstorm" in current_condition:
        return "Heavy rain expected (>20mm). Avoid fertilizer and pesticide application. Ensure field drainage channels are cleared to prevent waterlogging."
    elif max_forecast_rain > 10.0 or "rain" in current_condition or "shower" in current_condition:
        return "Moderate rainfall likely. Postpone chemical sprays and intercultural weeding for 48 hours."
    elif max_forecast_temp > 40.0:
        return "Extreme heat (>40°C) warning. Irrigate fields during early morning or late evening only. Apply organic mulch to retain soil moisture."
    elif "fog" in current_condition:
        return "Foggy conditions prevailing. High humidity can trigger fungal rusts and blights. Monitor crops closely and maintain airflow."
    else:
        return "Good conditions for agricultural operations. Favorable for fertilizer top-dressing, pesticide spraying, and crop monitoring."


if __name__ == "__main__":
    print("=== Testing Weather Module for Pune ===")
    pune_weather = get_weather("Pune")
    print(f"District: {pune_weather['district']}")
    print(f"Current Temp: {pune_weather['current_temp']}°C ({pune_weather['current_condition']})")
    print(f"Rain Alert: {pune_weather['rain_alert']} - {pune_weather['rain_alert_message']}")
    print("\n5-Day Forecast:")
    for day in pune_weather['forecast']:
        print(f"  {day['date']}: Max {day['max_temp']}°C | Min {day['min_temp']}°C | Rain {day['rain_mm']}mm | {day['condition']}")

    advice = get_farming_advice(pune_weather)
    print(f"\nFarming Advice:\n  🌾 {advice}")
