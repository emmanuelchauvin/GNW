# Projet Ignition — Global Neuronal Workspace (GNW)

An implementation of **Stanislas Dehaene's Global Neuronal Workspace theory** in Python, simulating a conscious agent through the orchestration of specialized unconscious modules and a global broadcasting mechanism.

## 🧠 Core Concept

The "Global Neuronal Workspace" (GNW) theory posits that consciousness arises from the **broadcasting** of information across the brain. 

In this implementation:
1.  **Unconscious Processing**: Specialized modules (Vision, Language, Geometry, etc.) process stimuli in parallel.
2.  **Ignition & Arbitration**: The system selects the most relevant analyzing module ("ignition").
3.  **Broadcasting**: The winning information is shared in the "Global Workspace", becoming available to all other processes and creating a moment of "consciousness".
4.  **Metacognition**: A Monitor module evaluates the internal coherence of the system.

## 🏗️ Architecture

### Unconscious Modules
Each module has a specialized "cognitive lens" defined by a specific System Prompt:

*   **👁️ Vision (`VisionModule`)**: Uses **Ollama (Gemma 3)** to analyze visual stimuli, detecting text, surfaces, and lighting.
*   **📐 Géomètre (`Geometre`)**: Analyzes spatial relationships, topology, and physical coherence.
*   **🗣️ Linguiste (`Linguiste`)**: Focuses on syntax, grammar, and referential integrity (e.g., checking if a mentioned object actually exists).
*   **🎭 Social (`Social`)**: Decodes social dynamics, theory of mind, and emotions.
*   **🤔 Pragmatique (`Pragmatique`)**: Infers intent, context, irony, and implicit meaning.

### Ignition Engine
The `IgnitionEngine` orchestrates the cognitive cycle:
1.  **Process**: Sends stimulus to all modules.
2.  **Arbitrate**: Selects the response with the highest priority score (0-10).
3.  **Broadcast**: Updates the `GlobalWorkspace` with the winner's analysis.
4.  **Loop**: Can run autonomously, re-injecting the previous state to refine understanding.

## 🛠️ Setup & Installation

### Prerequisites
*   Python 3.10+
*   [Ollama](https://ollama.com/) (for Vision module)
*   A **MiniMax API Key** (or compatible OpenAI-format key)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/emmanuelchauvin/GNW.git
    cd GNW
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Ollama (Vision)**:
    Ensure Ollama is running and pull the default vision model (Gemma 3):
    ```bash
    ollama pull gemma3:27b
    ```
    *(Note: You can change the model name in `gnw_engine.py` if needed)*

4.  **Environment Variables**:
    Create a `.env` file in the root directory:
    ```ini
    # API Key for MiniMax (Unconscious text modules)
    MINIMAX_KEY=your_minimax_api_key_here
    
    # Optional: Override Base URL if not using standard MiniMax endpoint
    # OPENAI_BASE_URL=https://api.minimax.io/v1
    ```

## 🚀 Usage

### Interface en ligne de commande
```python
import asyncio
from api_bridge import MiniMaxBridge
from gnw_engine import IgnitionEngine

async def main():
    async with MiniMaxBridge() as bridge:
        engine = IgnitionEngine(bridge)
        
        # Run a cognitive cycle on a text stimulus
        workspace = await engine.run("The cat is on the mat.")
        
        print(workspace.summary())

if __name__ == "__main__":
    asyncio.run(main())
```

### Interface Visuelle (Streamlit)
Pour lancer le tableau de bord interactif et visualiser le cycle cognitif en temps réel :

```bash
streamlit run dashboard.py
```
Cela ouvrira votre navigateur à l'adresse `http://localhost:8501`. Vous pourrez y configurer votre clé API, choisir le moteur de vision et injecter des stimuli textuels et visuels.

## 📂 Project Structure

*   `gnw_engine.py`: Main engine, Global Workspace class, and arbitration logic.
*   `modules_inconscients.py`: Definitions of all cognitive modules (Geometre, Linguiste, etc.).
*   `api_bridge.py`: Async client for MiniMax API (handling retries and JSON parsing).
*   `dashboard.py`: (Optional) Visualization interface.

---
*Note: This project is a conceptual implementation inspired by cognitive neuroscience theories.*
