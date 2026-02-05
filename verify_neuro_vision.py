
import asyncio
import logging
import os
from pathlib import Path
from neuro_architect import get_neuro_architect, NeuroArchitect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("validation")

def verify_hyper_v():
    print("🚀 Iniciando Validación de Ultragent Hyper-V...")
    neuro = get_neuro_architect()
    
    # 1. Simular Telemetría
    print("\n[1] Inyectando datos sintéticos...")
    neuro.ingest_telemetry("scout.py", "execution", {"duration": 0.5})
    neuro.ingest_telemetry("evolution.py", "variable_update", {"fitness_score": 92.5})
    neuro.ingest_telemetry("hud_manager.py", "error", {"message": "Simulated timeout"})
    print("✅ Telemetría ingerida.")
    
    # 2. Análisis de Impacto
    print("\n[2] Ejecutando Impact Analysis en 'scout.py'...")
    impact = neuro.analyze_impact("scout.py")
    print(f"   -> Riesgo: {impact.risk_score}")
    print(f"   -> Impacto Directo: {len(impact.direct_impact)} módulos")
    if impact.risk_score > 0:
        print("✅ Análisis de impacto funcional.")
    else:
        print("⚠️ Análisis de impacto retornó 0 (puede ser normal en grafo vacío).")

    # 3. Flow Tracing
    print("\n[3] Trazando flujo (scout -> mcp_server)...")
    # Nota: Puede que no exista conexión directa, pero probamos la herramienta
    trace = neuro.trace_flow("scout.py", "mcp_server.py")
    print(f"   -> Trace result: {trace}")
    
    # 4. Generación de Mapa 3D
    print("\n[4] Generando Neuro-Map WebGL...")
    html_path = neuro.export_neuro_map()
    
    if html_path.exists() and html_path.stat().st_size > 0:
        print(f"✅ Mapa generado exitosamente en: {html_path}")
        print(f"✅ Tamaño: {html_path.stat().st_size / 1024:.2f} KB")
    else:
        print("❌ Falló la generación del mapa HTML.")
        
    print("\n🎉 Validación Completada.")

if __name__ == "__main__":
    verify_hyper_v()
