import logging
try:
    from openhands.controller.agent import Agent
    # Forzar importaciones de agentes comunes para registro
    import openhands.agencies.codeact.codeact_agent
    
    print("Agentes registrados:")
    for key in Agent.list_agents():
        print(f"- {key}")
        
except Exception as e:
    print(f"Error al analizar agentes: {e}")
    import traceback
    traceback.print_exc()
