"""
Projet Ignition — Connection Test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Quick smoke-test that validates the MiniMax bridge works end-to-end.

Expected model response: {"test": "ok"}
"""

from __future__ import annotations

import asyncio
import logging
import sys

from api_bridge import MiniMaxBridge, MiniMaxBridgeError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Run a simple connectivity check against MiniMax M2.5."""
    logger.info("🚀  Projet Ignition — Test de connexion MiniMax")
    logger.info("=" * 55)

    prompt: str = (
        'Respond with exactly this JSON object and nothing else: {"test": "ok"}'
    )
    system_prompt: str = (
        "You are a strict JSON generator. "
        "Always respond with valid JSON objects only, no extra text."
    )

    try:
        async with MiniMaxBridge() as bridge:
            logger.info("📡  Envoi de la requête au modèle MiniMax M2.5…")
            result = await bridge.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
            )

            logger.info("✅  Réponse reçue : %s", result)

            # Validate the response
            if isinstance(result, dict) and result.get("test") == "ok":
                logger.info("🎯  Test RÉUSSI — La connexion fonctionne parfaitement.")
            else:
                logger.warning(
                    "⚠️  Réponse inattendue. Attendu {'test': 'ok'}, reçu : %s",
                    result,
                )
                sys.exit(1)

    except MiniMaxBridgeError as exc:
        logger.error("❌  Échec de la connexion : %s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("❌  Erreur de configuration : %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
