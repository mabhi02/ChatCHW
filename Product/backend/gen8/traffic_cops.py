"""Tier-2 traffic-cop validator.

The router in gen8 must emit TWO DMN-style decision tables instead of a
single flat rows list:
  * `cop1_queue_builder` -- hit_policy=COLLECT; adds modules to the queue
    when symptom/trigger flags fire.
  * `cop2_next_module`   -- hit_policy=UNIQUE;  picks exactly one next
    module each pass.

`validate_router` returns a list of human-readable errors; empty means
the router is well-formed. The pipeline logs any non-empty list as a
warning and the verifier raises an error-severity divergence for each.
"""

from __future__ import annotations


def validate_router(router: dict) -> list[str]:
    """Check that `router` is a two-cop dict with correct hit policies."""
    errors: list[str] = []
    if not isinstance(router, dict):
        return ["router is not a dict"]

    if "cop1_queue_builder" not in router:
        errors.append("missing cop1_queue_builder")
    if "cop2_next_module" not in router:
        errors.append("missing cop2_next_module")

    cop1 = router.get("cop1_queue_builder") or {}
    cop2 = router.get("cop2_next_module") or {}

    if cop1 and cop1.get("hit_policy") != "COLLECT":
        errors.append(
            f"cop1_queue_builder hit_policy must be COLLECT, got {cop1.get('hit_policy')!r}"
        )
    if cop2 and cop2.get("hit_policy") != "UNIQUE":
        errors.append(
            f"cop2_next_module hit_policy must be UNIQUE, got {cop2.get('hit_policy')!r}"
        )

    if cop1:
        rules = cop1.get("rules")
        if not isinstance(rules, list) or len(rules) == 0:
            errors.append("cop1_queue_builder.rules must be a non-empty list")
        else:
            for i, rule in enumerate(rules):
                if not isinstance(rule, dict):
                    errors.append(f"cop1_queue_builder.rules[{i}] is not a dict")
                    continue
                actions = rule.get("actions") or {}
                if not isinstance(actions, dict) or "add_to_queue" not in actions:
                    errors.append(
                        f"cop1_queue_builder.rules[{i}] must have actions.add_to_queue"
                    )

    if cop2:
        rules = cop2.get("rules")
        if not isinstance(rules, list) or len(rules) == 0:
            errors.append("cop2_next_module.rules must be a non-empty list")
        else:
            for i, rule in enumerate(rules):
                if not isinstance(rule, dict):
                    errors.append(f"cop2_next_module.rules[{i}] is not a dict")
                    continue
                actions = rule.get("actions") or {}
                has_next = isinstance(actions, dict) and (
                    "next_module" in actions or "end_visit" in actions
                )
                if not has_next:
                    errors.append(
                        f"cop2_next_module.rules[{i}] must have actions.next_module or actions.end_visit"
                    )

    return errors


def migrate_flat_router_to_cops(router: dict) -> dict:
    """Upgrade a gen7-style flat `{rows: [...]}` router into the two-cop shape.

    Used when Stage 3 accidentally emits the old flat shape despite the new
    prompt -- we salvage the output rather than losing an entire run. Every
    migrated rule is tagged with `_migrated_from_flat_router: true` so the
    verifier flags it as a warn-severity divergence.
    """
    if not isinstance(router, dict):
        return router
    if "cop1_queue_builder" in router or "cop2_next_module" in router:
        return router
    rows = router.get("rows") or router.get("rules") or []
    if not isinstance(rows, list):
        return router

    cop1_rules: list[dict] = []
    cop2_rules: list[dict] = []
    dropped_rows: list[dict] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            dropped_rows.append({"index": i, "reason": "row_not_a_dict", "row_repr": repr(row)[:200]})
            continue
        cond_src = row.get("condition", "")
        target = row.get("output_module") or row.get("next_module")
        rid = row.get("id") or f"r_{i}"
        if not target:
            dropped_rows.append({
                "index": i, "id": rid, "reason": "no_target_module",
                "condition": cond_src,
            })
            continue
        if str(target).startswith("mod_"):
            cop2_rules.append({
                "id": rid,
                "conditions": {"expr": cond_src},
                "actions": {"next_module": target},
                "_migrated_from_flat_router": True,
            })
        cop1_rules.append({
            "id": f"{rid}_enqueue",
            "conditions": {"expr": cond_src},
            "actions": {"add_to_queue": [target] if isinstance(target, str) else list(target)},
            "_migrated_from_flat_router": True,
        })

    return {
        "cop1_queue_builder": {
            "hit_policy": "COLLECT",
            "description": "Migrated from gen7-style flat router",
            "rules": cop1_rules,
        },
        "cop2_next_module": {
            "hit_policy": "UNIQUE",
            "description": "Migrated from gen7-style flat router",
            "rules": cop2_rules,
        },
        "_migrated_from_flat_router": True,
        "_migration_dropped_rows": dropped_rows,
    }
