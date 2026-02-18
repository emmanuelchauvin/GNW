import asyncio
import os
from dotenv import load_dotenv
from api_bridge import OpenRouterVisionBridge

async def test_vision():
    load_dotenv()
    
    # We need a small dummy image for testing structure
    # A 1x1 white pixel PNG
    base64_pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
    import base64
    image_bytes = base64.b64decode(base64_pixel)
    
    try:
        bridge = OpenRouterVisionBridge()
        print(f"Testing OpenRouter Vision with model: {bridge.model_name}")
        
        result = await bridge.analyze_image(
            image_bytes=image_bytes,
            prompt="What do you see in this image? (Should be a white pixel)",
            mime_type="image/png"
        )
        
        print("\n--- Result ---")
        print(result)
        print("--------------\n")
        
        if "Echec" in result:
            print("❌ Test failed.")
        else:
            print("✅ Test succeeded structure-wise.")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")

if __name__ == "__main__":
    asyncio.run(test_vision())
