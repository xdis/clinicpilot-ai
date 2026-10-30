"""Mutation check for the memory guards: removes one guard at a time and expects its test to fail.

    cd ai-service && LLM_PROVIDER=fake .venv/bin/python ../scripts/mutation_check.py

Each mutation edits a source file in place, runs the offline suite, and restores the file (also on Ctrl+C).
A mutation is "caught" when the named test fails. Exit code 0 only if every mutation is caught.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "ai-service" / "app"

# (guard, file, original code, mutated code, test that must fail)
MUTATIONS = [
    ("episodes drop medical sentences", "memory.py",
     "kept = [x for x in sentences if x and not _SAFETY.search(x) and not looks_like_injection(x)]",
     "kept = [x for x in sentences if x and not looks_like_injection(x)]",
     "test_episodes_never_keep_medical_facts"),
    ("preferences about someone else are rejected", "memory.py",
     "if not _FIRST_PERSON.search(message) or _THIRD_PARTY.search(message):", "if not _FIRST_PERSON.search(message):",
     "test_preferences_about_someone_else_are_not_the_patients"),
    ("the newest turn is always kept", "memory.py",
     "        if not kept:  # the newest turn alone", "        if False:  # the newest turn alone",
     "test_an_oversized_message_never_wipes_the_recent_context"),
    ("erasure tombstone drops queued writes", "memory.py",
     "            if self._erased_since(patient_id, since):\n                return []",
     "            if False:\n                return []",
     "test_queued_write_older_than_an_erasure_is_dropped"),
    ("retention deletes on disk", "memory.py",
     'if rank >= self.episode_limit or i.value.get("ts", 0) < cutoff:', "if rank >= self.episode_limit:",
     "test_retention_deletes_expired_episodes_on_disk"),
    ("injection gate skips extraction", "memory.py",
     '        if "possible_prompt_injection" in flags:\n            return False',
     "        if False:\n            return False",
     "test_instruction_like_message_is_not_written"),
    ("no gap: fold as soon as the window overflows", "memory.py",
     "        if len(self.window(history)) == len(history):\n            return history, summary, 0",
     "        if len(history) - len(self.window(history)) < 6:\n            return history, summary, 0",
     "test_no_gap_every_message_is_in_the_window_or_the_summary"),
    ("staff-review items only from the latest message", "memory.py",
     "            if update.safety_statement and _SAFETY.search(message):", "            if update.safety_statement:",
     "test_safety_statement_only_from_latest_message_and_only_once"),
    ("staff-review items are deduplicated", "memory.py",
     "                if stmt.lower() not in existing:", "                if True:",
     "test_safety_statement_only_from_latest_message_and_only_once"),
    ("recall skips irrelevant episodes", "memory.py",
     "top = [e for _, sc, e in ranked if sc > 0][:k] if hi > 0 else eps[:1]", "top = [e for _, sc, e in ranked][:k]",
     "test_recall_skips_irrelevant_episodes"),
    ("no stale time-of-day reminder", "agent.py",
     'and not (s and re.search(r"\\b(morning|afternoon|evening)s?\\b", s["user_message"], re.I)):', ":",
     "test_no_stale_time_reminder_when_the_message_changes_it"),
]


def main() -> int:
    caught = 0
    for name, file, old, new, test in MUTATIONS:
        path = APP / file
        src = path.read_text()
        if src.count(old) != 1:
            print(f"SKIP  {name}: guard not found exactly once in {file} (code changed?)")
            continue
        try:
            path.write_text(src.replace(old, new))
            out = subprocess.run([sys.executable, "-m", "pytest", "-o", "addopts=", "-q", "-x", "-k", test],
                                 cwd=APP.parent, capture_output=True, text=True).stdout
        finally:
            path.write_text(src)
        ok = "failed" in out
        caught += ok
        print(f"{'CAUGHT' if ok else 'MISSED'}  {name}  ({test})")
    print(f"\n{caught}/{len(MUTATIONS)} mutations caught")
    return 0 if caught == len(MUTATIONS) else 1


if __name__ == "__main__":
    sys.exit(main())
