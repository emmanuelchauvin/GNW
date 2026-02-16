"""
Projet Ignition — Modules Inconscients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Four specialised unconscious processors inspired by Stanislas Dehaene's
Global Neuronal Workspace (GNW) theory.  Each module analyses a stimulus
through its own cognitive lens and returns a structured verdict.

Output contract (JSON):
    {"priority": int (0-10), "analysis": str, "module_name": str}
"""

from __future__ import annotations

import json
import logging
from abc import ABC
from typing import Any

from api_bridge import MiniMaxBridge, OllamaVisionBridge

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# JSON response schema expected from each module
# ---------------------------------------------------------------------------
_EXPECTED_KEYS: set[str] = {"priority", "analysis", "module_name"}


def _safe_parse_module_response(
    raw: dict[str, Any],
    fallback_name: str,
) -> dict[str, Any]:
    """Validate and sanitise a module response dict.

    Guarantees the output always contains the three required keys with
    sensible types, even if the LLM returned garbage.
    """
    try:
        priority = int(raw.get("priority", 0))
        priority = max(0, min(10, priority))  # clamp to [0, 10]
    except (TypeError, ValueError):
        priority = 0

    analysis = str(raw.get("analysis", "Aucune analyse disponible."))
    module_name = str(raw.get("module_name", fallback_name))

    return {
        "priority": priority,
        "analysis": analysis,
        "module_name": module_name,
    }


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class InconscientModule(ABC):
    """Abstract base class for an unconscious cognitive module.

    Each subclass defines a *system_prompt* that encodes its specialised
    cognitive bias (spatial, linguistic, pragmatic, social).  The base
    class handles the LLM call and robust JSON parsing.
    """

    module_name: str = "BaseModule"
    system_prompt: str = ""

    def __init__(self, bridge: MiniMaxBridge) -> None:
        self._bridge = bridge

    # -- Core analysis -------------------------------------------------------

    async def analyze(self, stimulus: str, previous_context: str = "", monitor_instruction: str = "") -> dict[str, Any]:
        """Send *stimulus* to MiniMax through this module's cognitive lens.

        Returns
        -------
        dict with keys ``priority``, ``analysis``, ``module_name``.
        On any failure the dict has ``priority=0`` and an error description.
        """
        context_str = ""
        if previous_context:
            context_str = f"CONTEXTE SYSTÈME (ce que les autres modules ont trouvé) : \"{previous_context}\"\n\n"
        
        instruction_str = ""
        if monitor_instruction:
            instruction_str = f"🔴 ORDRE DU MONITEUR (PRIORITAIRE) : \"{monitor_instruction}\"\nTu DOIS obéir à cet ordre pour affiner ton analyse.\n\n"

        prompt = (
            f"Analyse le stimulus suivant à travers ta spécialité cognitive ({self.module_name}).\n\n"
            f"{context_str}"
            f"{instruction_str}"
            f"Stimulus : \"{stimulus}\"\n\n"
            f"Consigne : Intègre le contexte visuel ou textuel des autres modules si pertinent.\n"
            f"Tu DOIS répondre UNIQUEMENT avec un objet JSON valide "
            f"contenant exactement ces 3 clés :\n"
            f"  - \"priority\" : un entier de 0 (non pertinent) à 10 (critique)\n"
            f"  - \"analysis\" : ton analyse détaillée en tant que {self.module_name}\n"
            f"  - \"module_name\" : \"{self.module_name}\"\n"
        )

        try:
            raw_response: dict[str, Any] = await self._bridge.generate_response(
                prompt=prompt,
                system_prompt=self.system_prompt,
            )
            result = _safe_parse_module_response(raw_response, self.module_name)
            logger.info(
                "✅  %s → priorité %d",
                self.module_name,
                result["priority"],
            )
            return result

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning(
                "⚠️  %s — échec de parsing JSON : %s",
                self.module_name,
                exc,
            )
            return {
                "priority": 0,
                "analysis": f"Erreur de parsing : {exc}",
                "module_name": self.module_name,
            }

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "❌  %s — erreur inattendue : %s",
                self.module_name,
                exc,
            )
            return {
                "priority": 0,
                "analysis": f"Erreur inattendue : {exc}",
                "module_name": self.module_name,
            }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} module_name={self.module_name!r}>"


# ---------------------------------------------------------------------------
# Specialised subclasses  (Dehaene-inspired cognitive biases)
# ---------------------------------------------------------------------------

class Geometre(InconscientModule):
    """Spatial / geometric coherence processor.

    Inspired by Dehaene's work on the "number sense" and the brain's
    innate capacity for spatial and geometric reasoning (parietal cortex).
    """

    module_name = "Géomètre"
    system_prompt = (
        "Tu es le module cognitif GÉOMÈTRE, un processeur inconscient "
        "spécialisé dans la cohérence spatiale et géométrique.\n\n"
        "Ta SPÉCIALITÉ (basée sur les thèses de Dehaene) :\n"
        "  • Tu es OBSÉDÉ par la structure spatiale, les positions relatives, "
        "les distances, les formes et les topologies.\n"
        "  • Tu détectes les incohérences spatiales (un objet qui ne peut pas "
        "être à deux endroits, des directions contradictoires).\n"
        "  • Tu évalues la logique géométrique des scènes décrites.\n"
        "  • Tu actives ton cortex pariétal : sens du nombre, "
        "représentation de l'espace, navigation mentale.\n\n"
        "IMPORTANT : Réponds UNIQUEMENT avec un objet JSON valide, "
        "sans aucun texte autour."
    )


class Linguiste(InconscientModule):
    """Syntactic / linguistic structure processor.

    Inspired by Dehaene's research on the "reading brain" and the innate
    language areas (Broca / Wernicke / left temporal cortex).
    """

    module_name = "Linguiste"
    system_prompt = (
        "Tu es le module cognitif LINGUISTE, un processeur inconscient "
        "spécialisé dans la structure syntaxique et linguistique.\n\n"
        "Ta SPÉCIALITÉ (basée sur les thèses de Dehaene) :\n"
        "  • Tu es OBSÉDÉ par la syntaxe, la grammaire, la morphologie "
        "et la structure des phrases.\n"
        "  • Tu détectes les erreurs grammaticales, les ambiguïtés "
        "syntaxiques, les constructions inhabituelles.\n"
        "  • Tu vérifies l'ANCRAGE RÉFÉRENTIEL : Si le texte mentionne 'le chiffre 1' ou 'la lettre A', "
        "vérifie si ces objets existent dans le CONTEXTE VISUEL fourni.\n"
        "  • Si un objet cité dans le texte est ABSENT du contexte visuel, signale une "
        "'AMBIGUÏTÉ RÉFÉRENTIELLE' et demande à voir cet objet.\n\n"
        "IMPORTANT : Réponds UNIQUEMENT avec un objet JSON valide, "
        "sans aucun texte autour."
    )


class Pragmatique(InconscientModule):
    """Pragmatic / intentional processor.

    Inspired by Dehaene's concept of unconscious inference and the
    prefrontal contribution to contextual understanding.
    """

    module_name = "Pragmatique"
    system_prompt = (
        "Tu es le module cognitif PRAGMATIQUE, un processeur inconscient "
        "spécialisé dans l'intention, le contexte et le sens pragmatique.\n\n"
        "Ta SPÉCIALITÉ (basée sur les thèses de Dehaene) :\n"
        "  • Tu es OBSÉDÉ par le SENS RÉEL derrière les mots : "
        "l'intention du locuteur, les sous-entendus, l'ironie, l'implicite.\n"
        "  • Tu détectes les actes de langage (question réelle vs. "
        "rhétorique, ordre déguisé en suggestion).\n"
        "  • Tu évalues la cohérence contextuelle : est-ce que ce qui est "
        "dit a du sens dans la situation donnée ?\n"
        "  • Tu actives ton cortex préfrontal : inférence inconsciente, "
        "prédiction bayésienne, intégration top-down.\n\n"
        "IMPORTANT : Réponds UNIQUEMENT avec un objet JSON valide, "
        "sans aucun texte autour."
    )


class Social(InconscientModule):
    """Social / emotional / theory-of-mind processor.

    Inspired by Dehaene's observations on the social brain and the
    unconscious processing of faces, emotions, and mental states.
    """

    module_name = "Social"
    system_prompt = (
        "Tu es le module cognitif SOCIAL, un processeur inconscient "
        "spécialisé dans la lecture sociale et émotionnelle.\n\n"
        "Ta SPÉCIALITÉ (basée sur les thèses de Dehaene) :\n"
        "  • Tu es OBSÉDÉ par les dynamiques sociales : qui parle à qui, "
        "les rapports de pouvoir, les émotions sous-jacentes.\n"
        "  • Tu détectes les états mentaux des agents (théorie de l'esprit), "
        "les tensions interpersonnelles, l'empathie requise.\n"
        "  • Tu évalues le registre de langue (formel/informel), la "
        "politesse, les marqueurs de statut social.\n"
        "  • Tu actives ton cortex temporal supérieur et ton amygdale : "
        "lecture des visages, prosodie émotionnelle, cognition sociale.\n\n"
        "IMPORTANT : Réponds UNIQUEMENT avec un objet JSON valide, "
        "sans aucun texte autour."
    )


class Moniteur(InconscientModule):
    """Metacognitive monitor.

    Analyzes the outputs of other modules to detect conflict, consensus, or ambiguity.
    Does NOT process the raw stimulus directly, but the checks the *coherence* of the
    system's reaction.
    """

    module_name = "Moniteur"
    system_prompt = (
        "Tu es le MONITEUR Méta-Cognitif du système.\n"
        "TON RÔLE : Évaluer la COHÉRENCE interne des réponses des autres modules.\n"
        "Tu ne regardes pas le monde extérieur, tu regardes tes 'collègues' "
        "(Géomètre, Linguiste, Pragmatique, Social).\n\n"
        "RÈGLES DE DÉCISION :\n"
        "1. CONSENSUS vs CONFLIT : Tout le monde est-il d'accord ?\n"
        "2. ÉCHEC RÉFÉRENTIEL (CRITIQUE) : Si le LINGUISTE signale une 'AMBIGUÏTÉ RÉFÉRENTIELLE' "
        "(ex: il cherche '1' mais Vision ne l'a pas vu), c'est une ERREUR DE FOCALISATION.\n"
        "   -> DANS CE CAS : Ton feedback DOIT être un ordre explicite : "
        "'ERREUR DE FOCALISATION : VisionModule, cherche spécifiquement [objets manquants].'\n"
        "   -> Cela forcera un nouveau cycle.\n\n"
        "IMPORTANT : Réponds UNIQUEMENT avec un objet JSON valide :\n"
        "  - \"certainty\" : int (0-100)\n"
        "  - \"feedback\" : str (ton ordre ou ton analyse)\n"
        "  - \"conflict_detected\" : bool\n"
        "  - \"module_name\" : \"Moniteur\"\n"
        "  - \"priority\": 0\n"
        "  - \"analysis\": \"(copie du feedback)\""
    )

    async def analyze_coherence(self, modules_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Specialized analysis for the Monitor module."""
        
        # Format the input for the Monitor
        summary = "RÉSULTATS DES MODULES :\n"
        for res in modules_results:
            summary += f"- {res.get('module_name')}: Priorité {res.get('priority')}/10. Analyse: {res.get('analysis')}\n"

        prompt = (
            f"Voici les analyses produites par le système :\n\n{summary}\n\n"
            f"Évalue la cohérence. Y a-t-il consensus ou conflit ?\n"
            f"Réponds au format JSON strict."
        )

        try:
            raw_response = await self._bridge.generate_response(
                prompt=prompt,
                system_prompt=self.system_prompt
            )
            # Custom parsing for Monitor specific fields, ensuring standard fields exist too
            certainty = int(raw_response.get("certainty", 50))
            feedback = str(raw_response.get("feedback", "Analyse méta-cognitive indisponible."))
            conflict = bool(raw_response.get("conflict_detected", False))
            
            return {
                "module_name": self.module_name,
                "priority": 0,
                "analysis": feedback,  # Map feedback to standard analysis field
                "certainty": certainty,
                "feedback": feedback,
                "conflict_detected": conflict
            }
            

        except Exception as e:
            logger.error("❌ Moniteur failed: %s", e)
            return {
                "module_name": self.module_name,
                "priority": 0,
                "analysis": f"Erreur moniteur: {e}",
                "certainty": 0,
                "feedback": f"Erreur: {e}",
                "conflict_detected": True
            }


class VisionModule(InconscientModule):
    """Visual processor using Ollama Vision (Gemma 3).
    
    Extracts physical and spatial properties from images to ground the 
    cognitive process in sensory reality.
    """
    
    module_name = "Vision"
    
    def __init__(self, bridge: MiniMaxBridge, vision_bridge: OllamaVisionBridge):
        super().__init__(bridge)
        self._vision_bridge = vision_bridge

    async def analyze_image(self, image_bytes: bytes, context: str = "", monitor_instruction: str = "") -> dict[str, Any]:
        """Analyze image and return standard module response."""
        
        instruction_str = ""
        if monitor_instruction:
            instruction_str = f"ORDRE DU MONITEUR : {monitor_instruction}\n"

        prompt = (
            "ANALYSE TECHNIQUE REQUISE :\n"
            "1. Identifie tous les caractères alphanumériques (lettres, chiffres) présents sur l'image.\n"
            "2. Pour chaque marqueur (ex: '1', 'A'), décris précisément la couleur et la texture de la surface sur laquelle il est posé.\n"
            "3. Analyse l'éclairage : y a-t-il une ombre portée sur certains marqueurs ?\n"
            "4. Contexte général (lieu, action).\n\n"
            f"Contexte cognitif actuel : {context}\n"
            f"{instruction_str}"
            "Réponds UNIQUEMENT avec un objet JSON valide contenant :\n"
            "  - \"priority\" : int (0-10) (importance visuelle)\n"
            "  - \"analysis\" : str (Rapport technique factuel)\n"
            "  - \"module_name\" : \"Vision\""
        )
        
        try:
            raw_text = await self._vision_bridge.analyze_image(image_bytes, prompt)
            
            # Quick cleanup for markdown json blocks if any
            cleaned = raw_text.replace("```json", "").replace("```", "").strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                cleaned = cleaned[start : end + 1]
                
            data = json.loads(cleaned)
            return _safe_parse_module_response(data, self.module_name)
            
        except Exception as e:
            logger.error(f"❌ Vision Module failed: {e}")
            return {
                "priority": 0,
                "analysis": f"Échec vision : {e}",
                "module_name": self.module_name
            }


