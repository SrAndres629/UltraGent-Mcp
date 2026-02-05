import os
import inspect

# Forzar .NET 8.0
os.environ["PYTHONNET_RUNTIME"] = "coreclr"

try:
    from openhands.controller.agent import Agent
    from openhands.agencies.codeact.codeact_agent import CodeActAgent
    from openhands.controller.agent.codeact_agent import CodeActAgent as CodeActAgentAlt
    print("CodeActAgent importado.")
except ImportError as e:
    print(f"Fallo importación CodeActAgent: {e}")

try:
    from openhands.agencies.browsing.browsing_agent import BrowsingAgent
    print("BrowsingAgent importado.")
except ImportError:
    pass

try:
    from openhands.core.message import CmdRunAction
    sig = inspect.signature(CmdRunAction.__init__)
    print(f"Firma de CmdRunAction: {sig}")
except Exception as e:
    print(f"Fallo inspección CmdRunAction: {e}")

print("Agentes registrados en Agent.list_agents():")
try:
    print(Agent.list_agents())
except Exception as e:
    print(f"Error listando agentes: {e}")
