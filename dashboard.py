import streamlit as st
import asyncio
import os
from api_bridge import MiniMaxBridge
from gnw_engine import IgnitionEngine, GlobalWorkspace

# --- Configuration de la page ---
st.set_page_config(
    page_title="THE IGNITION PROJECT",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session State Management ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "is_running" not in st.session_state:
    st.session_state.is_running = False

# --- Styles CSS personnalisés ---
st.markdown("""
<style>
    /* Force Dark Mode Colors */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .highlight-green {
        color: #50c878;
        font-weight: bold;
    }
    .ignition-box {
        border: 2px solid #50c878;
        border-radius: 10px;
        padding: 20px;
        background-color: rgba(80, 200, 120, 0.1);
        text-align: center;
        box-shadow: 0 0 15px rgba(80, 200, 120, 0.5);
        margin-bottom: 20px;
    }
    .ignition-title {
        color: #50c878;
        font-size: 1.5em;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .winning-module {
        font-size: 2em;
        font-weight: bold;
        color: #ffffff;
    }
    section[data-testid="stSidebar"] {
        background-color: #161b22;
    }
</style>
""", unsafe_allow_html=True)

# --- Barre Latérale : Config ---
with st.sidebar:
    st.title("🧠  GNW Simulation")
    st.markdown("---")
    
    st.header("Configuration")
    env_api_key = os.getenv("MINIMAX_KEY")
    if env_api_key:
        st.success("API Key détectée.", icon="✅")
        api_key_input = env_api_key
    else:
        api_key_input = st.text_input("MiniMax API Key", type="password")
        if api_key_input:
            os.environ["MINIMAX_KEY"] = api_key_input

    st.markdown("---")
    st.subheader("Moteur de Vision")
    vision_provider = st.radio(
        "Choisir le moteur",
        ["Ollama (Local)", "OpenRouter (Cloud)"],
        index=0,
        help="Ollama utilise Gemma 3 en local. OpenRouter utilise Nemotron Nano 12B VL."
    )
    vision_provider_code = "ollama" if "Ollama" in vision_provider else "openrouter"

    st.markdown("---")
    st.subheader("Paramètres Boucle")
    max_iters = st.slider("Max Itérations", 1, 10, 5)
    target_certainty = st.slider("Cible Certitude (%)", 50, 100, 90)

# --- Interface Principale ---
st.title("THE IGNITION PROJECT (Phase 2)")
st.caption("Poursuite Automatique de la Pensée & Méta-Cognition")

# Affichage de l'historique (Chat)
if st.session_state.chat_history:
    st.divider()
    st.subheader("📜 Flux de Pensée (Stream of Thought)")
    for idx, (role, content, extra) in enumerate(st.session_state.chat_history):
        with st.chat_message(role, avatar="🧠" if role == "assistant" else "👤"):
            st.markdown(content)
            if extra:
                with st.expander("Détails Inconscients"):
                    st.json(extra)

# Zone d'Input
with st.container():
    col_input, col_file = st.columns([4, 1])
    with col_input:
        stimulus = st.chat_input("Injecter un stimulus dans l'espace global...")
    with col_file:
         uploaded_file = st.file_uploader("Image (Optionnel)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")


# --- Logique de Simulation Loop ---
async def run_autonomous_loop(initial_text, image_bytes, api_key):
    # 1. Render User Message immediately
    with st.chat_message("user", avatar="👤"):
        if image_bytes:
            st.image(image_bytes, caption="Stimulus Visuel", width=300)
        st.markdown(initial_text)
    
    # Add to history
    st.session_state.chat_history.append(("user", initial_text, {"image": image_bytes} if image_bytes else None))
    
    # 2. Status container for the loop metadata
    status_container = st.status("🚀 Cycle cognitif en cours...", expanded=True)
    
    try:
        async with MiniMaxBridge(api_key=api_key) as bridge:
            engine = IgnitionEngine(bridge, vision_provider=vision_provider_code)
            
            step_count = 0
            
            # 3. Stream the thought process
            # Note: We pass image_data to the loop
            async for workspace in engine.run_autonomous_cycle(
                initial_stimulus=initial_text,
                image_data=image_bytes,
                max_iters=max_iters,
                target_certainty=target_certainty
            ):
                step_count += 1
                
                # Update Status
                status_container.write(f"**Itération {step_count}** : Ignition du module *{workspace.winning_module}*.")
                status_container.write(f"Certitude : {workspace.certainty}% | Moniteur : {workspace.feedback}")
                
                # Render Assistant Message LIVE
                # Check if there was an instruction from the previous step (which is workspace.feedback but from prev iter)
                # Actually, the workspace object contains the *result* of the current step.
                # The instruction for *this* step came from the *previous* workspace.feedback.
                # We don't easily have the previous workspace here unless we track it.
                # BUT, `workspace` has a `feedback` field which is the monitor's output *for this step*.
                # The "Focus Attentionnel" for the *next* step is this feedback.
                # To show what was the focus *of this step*, we need to track it.
                # Typically, the "Analysis" already reflects the focus.
                
                # Let's visualize the Monitor's *current* feedback as "Next Focus".
                
                msg_content = (
                    f"**[{workspace.winning_module}]** (Prio {workspace.priority}/10)\n\n"
                    f"{workspace.analysis}\n\n"
                    f"---\n"
                )
                
                # If Monitor has a strong opinion (feedback), visualize it as "Focus Attentionnel"
                if workspace.feedback:
                     msg_content += f"🎯 **Focus Attentionnel (Moniteur)** : *{workspace.feedback}*\n"
                else:
                     msg_content += f"*👀 Moniteur : {workspace.feedback}*\n"
                
                # Metadata for the "Unconscious Journal"
                journal_data = {
                    "step": step_count,
                    "winner": workspace.winning_module,
                    "certainty": workspace.certainty,
                    "monitor_feedback": workspace.feedback,
                    "all_modules": [
                        {
                            "module": m.get("module_name"),
                            "priority": m.get("priority"),
                            "analysis": m.get("analysis")
                        } 
                        for m in workspace.all_results
                    ]
                }
                
                with st.chat_message("assistant", avatar="🧠"):
                    st.markdown(msg_content)
                    
                    # 📓 JOURNAL DE L'INCONSCIENT (Introspection)
                    with st.expander(f"📓 Journal de l'Inconscient (Itération {step_count})", expanded=False):
                        st.caption("Trace brute des processus inconscients")
                        
                        # Tabs for different views
                        tab_overview, tab_json = st.tabs(["Vue d'ensemble", "JSON Brut"])
                        
                        with tab_overview:
                            # Visual comparison of priorities
                            for mod_res in workspace.all_results:
                                m_name = mod_res.get("module_name", "Unknown")
                                m_prio = mod_res.get("priority", 0)
                                m_analysis = mod_res.get("analysis", "")
                                
                                # Highlight the winner
                                is_winner = (m_name == workspace.winning_module)
                                icon = "🏆" if is_winner else "▪️"
                                
                                st.markdown(f"**{icon} {m_name}** (Prio {m_prio})")
                                st.markdown(f"> {m_analysis}")
                                st.divider()

                        with tab_json:
                            st.json(journal_data)
                
                # Add to history for persistence
                st.session_state.chat_history.append(("assistant", msg_content, journal_data))

            status_container.update(label=f"✅ Cycle terminé en {step_count} itérations", state="complete", expanded=False)

    except Exception as e:
        status_container.update(label="❌ Erreur Critique", state="error")
        st.error(f"Erreur : {str(e)}")

# Gestion de l'input
if stimulus:
    if not api_key_input:
        st.error("Clé API manquante.")
    else:
        # Prepare image bytes if present
        img_bytes = uploaded_file.getvalue() if uploaded_file else None
        
        # Run the loop
        asyncio.run(run_autonomous_loop(stimulus, img_bytes, api_key_input))

# Bouton Clear
if st.session_state.chat_history:
    if st.button("Effacer la mémoire"):
        st.session_state.chat_history = []
        st.rerun()

