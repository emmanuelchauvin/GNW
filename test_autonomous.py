import asyncio
import logging
from unittest.mock import MagicMock
from gnw_engine import IgnitionEngine
from api_bridge import MiniMaxBridge

# Configure logging
logging.basicConfig(level=logging.INFO)

class MockBridge(MiniMaxBridge):
    def __init__(self):
        pass
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def generate_response(self, prompt, system_prompt):
        # Determine if this is a Monitor call or a Module call
        if "MONITEUR" in system_prompt:
            # First few calls return low certainty, then high
            # We can use a counter or random, but for deterministic test let's rely on ... ?
            # Let's return a fixed sequence based on call count if possible, 
            # but here we don't have state easily. 
            # Let's just return a "Success" state to check the loop parses it.
            # To test the loop, we probably want to simulate increasing certainty?
            
            # Simple hack: checking prompt content or just random
            return {
                "certainty": 95, 
                "feedback": "Consensus parfait.", 
                "conflict_detected": False
            }
        else:
            # Standard module response
            return {
                "priority": 8,
                "analysis": "Test Analysis",
                "module_name": "TestModule"
            }

async def test_loop():
    print("--- Testing Autonomous Loop ---")
    bridge = MockBridge()
    engine = IgnitionEngine(bridge)
    
    # We want to verified it yields workspaces
    count = 0
    async for workspace in engine.run_autonomous_cycle("Test input", max_iters=3, target_certainty=90):
        count += 1
        print(f"Step {count}: Winner={workspace.winning_module}, Certainty={workspace.certainty}%")
        
    if count == 1:
        print("✅ SUCCESS: Loop stopped immediately because certainty (95%) > target (90%)")
    else:
        print(f"❌ FAILURE: Loop ran {count} times instead of 1.")

if __name__ == "__main__":
    asyncio.run(test_loop())
