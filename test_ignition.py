"""
Projet Ignition — Test d'Ignition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
End-to-end test: fires a stimulus through the 4 unconscious modules
and displays the ignition result.

Usage:
    python test_ignition.py
"""

from __future__ import annotations

import asyncio
import logging
import sys

from api_bridge import MiniMaxBridge, MiniMaxBridgeError
from gnw_engine import IgnitionEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Sample stimuli to test different module affinities
STIMULI: list[str] = [
    "Le chat est assis sur le tapis près de la fenêtre, à gauche de la table.",
]


async def main() -> None:
    """Run the full ignition cycle on each sample stimulus."""
    logger.info("🚀  Projet Ignition — Test du Moteur Cognitif")
    logger.info("=" * 55)

    try:
        async with MiniMaxBridge() as bridge:
            engine = IgnitionEngine(bridge)

            for i, stimulus in enumerate(STIMULI, 1):
                logger.info(
                    "── Stimulus %d/%d ──────────────────────────",
                    i,
                    len(STIMULI),
                )
                workspace = await engine.run(stimulus)

                if workspace.is_conscious:
                    logger.info(
                        "🎯  Ignition réussie — %s a gagné (priorité %d/10)",
                        workspace.winning_module,
                        workspace.priority,
                    )
                else:
                    logger.warning("⚫  Aucune ignition pour ce stimulus.")

    except MiniMaxBridgeError as exc:
        logger.error("❌  Erreur API MiniMax : %s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("❌  Erreur de configuration : %s", exc)
        sys.exit(1)

    logger.info("✅  Test d'ignition terminé avec succès.")


if __name__ == "__main__":
    asyncio.run(main())
