"""
Projet Ignition — GNW Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Global Neuronal Workspace engine implementing Dehaene's ignition model.

Architecture:
  1. Four unconscious modules process a stimulus in parallel.
  2. An arbitration step selects the highest-priority response (ignition).
  3. The winning analysis is broadcast to the global workspace.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from modules_inconscients import (
    Geometre,
    InconscientModule,
    Linguiste,
    Pragmatique,
    Social,
    Social,
    Moniteur,
    VisionModule,
)
from api_bridge import MiniMaxBridge, OllamaVisionBridge, OpenRouterVisionBridge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global Workspace — stores the current conscious state
# ---------------------------------------------------------------------------
@dataclass
class GlobalWorkspace:
    """Represents the current *conscious* state of the system.

    After ignition, this object holds the winning module's analysis and
    all raw results for introspection.
    """

    winning_module: str = ""
    analysis: str = ""
    priority: int = 0
    certainty: int = 0
    feedback: str = ""
    all_results: list[dict[str, Any]] = field(default_factory=list)
    raw_stimulus: str = ""

    @property
    def is_conscious(self) -> bool:
        """Return True if ignition has occurred (a module has won)."""
        return self.priority > 0 and bool(self.winning_module)

    def summary(self) -> str:
        """Human-readable summary of the workspace state."""
        if not self.is_conscious:
            return "⚫  Espace de travail vide — aucune ignition."
        return (
            f"🔥  IGNITION — Module gagnant : {self.winning_module} "
            f"(priorité {self.priority}/10)\n"
            f"🎯  Certitude : {self.certainty}%\n"
            f"📝  Analyse : {self.analysis}\n"
            f"👀  Moniteur : {self.feedback}"
        )


# ---------------------------------------------------------------------------
# Ignition Engine — orchestrates the cognitive cycle
# ---------------------------------------------------------------------------
class IgnitionEngine:
    """Orchestrates the unconscious → ignition → broadcast cycle.

    Usage::

        async with MiniMaxBridge() as bridge:
            engine = IgnitionEngine(bridge)
            workspace = await engine.run("Le chat est sur le tapis.")
            print(workspace.summary())
    """

    def __init__(self, bridge: MiniMaxBridge, vision_provider: str = "ollama") -> None:
        self._bridge = bridge
        self._workspace = GlobalWorkspace()

        # Instantiate the four unconscious modules
        self._modules: list[InconscientModule] = [
            Geometre(bridge),
            Linguiste(bridge),
            Pragmatique(bridge),
            Social(bridge),
        ]
        self._monitor = Moniteur(bridge)
        
        # Vision Module (Ollama / OpenRouter)
        self._has_vision = False
        try:
            if vision_provider == "openrouter":
                self._vision_bridge = OpenRouterVisionBridge()
                self._vision_module = VisionModule(bridge, self._vision_bridge)
                self._has_vision = True
                logger.info("👁️  Module Vision (OpenRouter/Nemotron) activé.")
            else:
                self._vision_bridge = OllamaVisionBridge()
                self._vision_module = VisionModule(bridge, self._vision_bridge)
                self._has_vision = True
                logger.info("👁️  Module Vision (Ollama/Gemma3) activé.")
        except ImportError as e:
            logger.warning(f"⚠️  Module Vision désactivé (Dépendance manquante: {e}).")
        except Exception as e:
            logger.warning(f"⚠️  Module Vision désactivé (Erreur init: {e}).")

        logger.info(
            "🧠  IgnitionEngine initialisé avec %d modules : %s + Moniteur + Vision(%s)",
            len(self._modules),
            ", ".join(m.module_name for m in self._modules),
            "ON" if self._has_vision else "OFF"
        )

    # -- Step 1 : Parallel unconscious processing ----------------------------

    async def process_stimulus(
        self,
        input_text: str,
        image_data: bytes | None = None,
        previous_context: str = "",
        monitor_instruction: str = ""
    ) -> list[dict[str, Any]]:
        """Launch all modules in parallel on the given *input_text*.
        
        If *image_data* is provided, VisionModule runs first to prime the others.
        """
        logger.info("📨  Stimulus reçu : \"%s\"", input_text)
        if previous_context:
            logger.info("↩️  Contexte précédent injecté : \"%s\"", previous_context[:50] + "...")
        if monitor_instruction:
            logger.info("🔴  Instruction Moniteur : \"%s\"", monitor_instruction)
        
        logger.info("⏳  Lancement parallèle de %d modules…", len(self._modules))
        
        # 1. Vision Analysis first (if image present)
        vision_context = ""
        vision_result = None
        
        if image_data and self._has_vision:
            logger.info("👁️  Analyse visuelle en cours...")
            vision_result = await self._vision_module.analyze_image(
                image_data, 
                context=previous_context,
                monitor_instruction=monitor_instruction
            )
            if vision_result["priority"] > 0:
                vision_context = f"\n[CONTEXTE VISUEL DU MODULE VISION] : {vision_result['analysis']}\n"
                logger.info("👁️  Contexte visuel extrait : %s...", vision_context[:50])

        # 2. Prepare tasks for other modules (injecting vision context)
        full_stimulus = input_text + vision_context
        
        tasks = [
            module.analyze(full_stimulus, previous_context, monitor_instruction) 
            for module in self._modules
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[dict[str, Any]] = []
        
        # Add Vision result if available
        if vision_result:
            results.append(vision_result)

        for idx, result in enumerate(raw_results):
            module_name = self._modules[idx].module_name
            if isinstance(result, Exception):
                logger.error(
                    "❌  Module %s a levé une exception : %s",
                    module_name,
                    result,
                )
                results.append({
                    "priority": 0,
                    "analysis": f"Exception : {result}",
                    "module_name": module_name,
                })
            elif isinstance(result, dict):
                results.append(result)
            else:
                logger.warning(
                    "⚠️  Module %s a renvoyé un type inattendu : %s",
                    module_name,
                    type(result).__name__,
                )
                results.append({
                    "priority": 0,
                    "analysis": f"Type inattendu : {type(result).__name__}",
                    "module_name": module_name,
                })

        logger.info(
            "📊  Résultats reçus : %s",
            " | ".join(
                f"{r['module_name']}={r['priority']}" for r in results
            ),
        )
        return results

    # -- Step 2 : Arbitration (ignition) ------------------------------------

    @staticmethod
    def arbitrate(results: list[dict[str, Any]]) -> dict[str, Any]:
        """Select the module with the highest priority score.

        This *is* the ignition: the winning module's content becomes
        globally accessible in the workspace.

        If all priorities are equal (or zero), the first module wins
        as a deterministic tie-breaker.
        """
        if not results:
            logger.warning("⚠️  Arbitrage appelé sans résultats.")
            return {
                "priority": 0,
                "analysis": "Aucun résultat à arbitrer.",
                "module_name": "Aucun",
            }

        winner = max(results, key=lambda r: r.get("priority", 0))
        logger.info(
            "🔥  IGNITION — Le module '%s' l'emporte avec priorité %d/10.",
            winner["module_name"],
            winner["priority"],
        )
        return winner

    # -- Step 3 : Broadcast -------------------------------------------------

    def broadcast(
        self, 
        winning_analysis: dict[str, Any],
        monitor_feedback: dict[str, Any]
    ) -> GlobalWorkspace:
        """Broadcast the winning analysis to the global workspace.

        For now this simply updates the workspace state and logs the
        result.  Future versions will propagate to downstream consumers.
        """
        self._workspace = GlobalWorkspace(
            winning_module=winning_analysis.get("module_name", "Inconnu"),
            analysis=winning_analysis.get("analysis", ""),
            priority=winning_analysis.get("priority", 0),
            certainty=monitor_feedback.get("certainty", 0),
            feedback=monitor_feedback.get("feedback", ""),
            all_results=self._workspace.all_results,
            raw_stimulus=self._workspace.raw_stimulus,
        )

        # -- Console output --------------------------------------------------
        print("\n" + "=" * 60)
        print("  🧠  GLOBAL NEURONAL WORKSPACE — ÉTAT CONSCIENT")
        print("=" * 60)
        print(f"  Module gagnant  : {self._workspace.winning_module}")
        print(f"  Priorité        : {self._workspace.priority}/10")
        print(f"  Certitude       : {self._workspace.certainty}%")
        print(f"  Analyse         : {self._workspace.analysis}")
        print(f"  Moniteur        : {self._workspace.feedback}")
        print("-" * 60)
        print("  Résumé de tous les modules :")
        for r in self._workspace.all_results:
            marker = "🏆" if r["module_name"] == self._workspace.winning_module else "  "
            print(f"    {marker} {r['module_name']:12s} → priorité {r['priority']}/10")
        print("=" * 60 + "\n")

        return self._workspace

    # -- Convenience: full cycle ---------------------------------------------


    async def run(
        self, 
        input_text: str, 
        image_data: bytes | None = None,
        previous_context: str = "",
        monitor_instruction: str = ""
    ) -> GlobalWorkspace:
        """Execute the full cognitive cycle: process → arbitrate → broadcast.
        """
        logger.info("🚀  Démarrage du cycle cognitif…")

        # Step 1 — Parallel unconscious processing
        results = await self.process_stimulus(input_text, image_data, previous_context, monitor_instruction)

        # Store raw results and stimulus
        self._workspace.all_results = results
        self._workspace.raw_stimulus = input_text

        # Step 2 — Arbitration (ignition)
        winner = self.arbitrate(results)

        # Step 2.5 — Metacognition (Monitor checks coherence)
        logger.info("🧐  Moniteur vérifie la cohérence…")
        monitor_result = await self._monitor.analyze_coherence(results)
        logger.info(
            "👀  Feedback Moniteur : %s (Certitude %d%%)", 
            monitor_result.get("feedback"), 
            monitor_result.get("certainty")
        )

        # Step 3 — Broadcast
        # Inject monitor feedback into broadcast
        workspace = self.broadcast(winner, monitor_result)

        return workspace

    # -- Autonomous Loop -----------------------------------------------------

    async def run_autonomous_cycle(
        self, 
        initial_stimulus: str, 
        image_data: bytes | None = None,
        max_iters: int = 5,
        target_certainty: int = 90
    ):
        """Run an autonomous cognitive loop.
        
        Yields the GlobalWorkspace state at each step.
        Stops when certainty >= target_certainty or max_iters is reached.
        Feeds back the monitor's feedback + winning analysis as context for the next step.
        """
        context = ""
        instruction = ""
        user_stopped = False # Flag accessible if we want to stop externally? 
        # For now, we rely on the caller to break the generator if needed or just let it run.
        
        for i in range(max_iters):
            logger.info(f"🔄  Cycle Autonome : Itération {i+1}/{max_iters}")
            
            # Run one cycle
            workspace = await self.run(
                initial_stimulus, 
                image_data=image_data, 
                previous_context=context,
                monitor_instruction=instruction
            )
            
            # Yield results for real-time UI updates
            yield workspace

            # Check termination
            if workspace.certainty >= target_certainty:
                logger.info(f"✅  Certitude cible atteinte ({workspace.certainty}% >= {target_certainty}%)")
                break
                
            logger.info("↩️  Feedback réinjecté pour la prochaine itération.")


    # -- Accessors -----------------------------------------------------------

    @property
    def workspace(self) -> GlobalWorkspace:
        """Current workspace state (read-only accessor)."""
        return self._workspace

    @property
    def modules(self) -> list[InconscientModule]:
        """Registered unconscious modules."""
        return list(self._modules)
