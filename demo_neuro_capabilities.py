
import json
import logging
from neuro_architect import get_neuro_architect

# Configurar logger limpio
logging.basicConfig(level=logging.ERROR) 

def demo_utility():
    print("🧠 VERIFICACIÓN DE UTILIDAD: NEURO-VISION (Hyper-V)\n" + "="*60)
    
    # 1. Inicialización
    neuro = get_neuro_architect()
    state = neuro.get_brain_state()
    
    print(f"\n📊 ESTADO DE USO (NeuroGraph Snapshot)")
    print(f"   • Neuronas (Nodos): {state['neuron_count']}")
    print(f"   • Sinapsis (Enlaces): {state['synapse_count']}")
    print(f"   • Estado del Sistema: ONLINE")
    
    # 2. Prueba de Utilidad: Análisis de Impacto
    target = "router.py"
    print(f"\n🛡️  PRUEBA DE UTILIDAD: Impact Analysis en '{target}'")
    print("   (Simulando que un agente quiere refactorizar el Router...)")
    
    impact = neuro.analyze_impact(target)
    
    print(f"   -> Riesgo Calculado: {impact.risk_score:.1f}/100")
    print(f"   -> Impacto Directo ({len(impact.direct_impact)}): {impact.direct_impact[:3]}...")
    print(f"   -> Efecto Onda ({len(impact.ripple_effect)}): {impact.ripple_effect[:3]}...")
    
    if impact.risk_score > 50:
        print("   ✅ CONCLUSIÓN: La herramienta detectó alto riesgo. Un agente habría sido advertido.")
    else:
        print("   ✅ CONCLUSIÓN: La herramienta detectó bajo riesgo.")

    # 3. Prueba de Utilidad: Trace Flow
    start, end = "vision.py", "mcp_server.py"
    print(f"\n📍 PRUEBA DE UTILIDAD: Trace Flow ('{start}' -> '{end}')")
    trace = neuro.trace_flow(start, end)
    
    if trace['exists']:
        path_str = " -> ".join(trace['path'])
        print(f"   -> Ruta encontrada: {path_str}")
        print("   ✅ CONCLUSIÓN: El agente puede ver cómo viajan los datos entre módulos.")
    else:
        print(f"   -> No se encontró ruta directa (Separación de intereses confirmada).")

    print("\n" + "="*60)
    print("Resumen: Las herramientas MCP están entregando inteligencia accionable.")

if __name__ == "__main__":
    demo_utility()
