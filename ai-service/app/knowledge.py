"""Clinic knowledge base and hybrid retrieval.

Each clinic, doctor and treatment row becomes a self-contained "card" with a stable id
(C001, D103, T003) that answers can cite. Retrieval fuses two rankings with Reciprocal
Rank Fusion (RRF, k=60):
  1. lexical: BM25 over card text (exact names, fees, hours, policies);
  2. semantic: the specialty the LLM inferred from the patient's description
     (e.g. "itchy rash" -> Dermatology), which BM25 alone would miss.
The dataset is tiny, so this is deliberately simple, transparent and dependency-light;
swapping in a vector index only changes the second ranker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rank_bm25 import BM25Okapi

RRF_K = 60
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "is", "are", "with", "my", "i", "me", "do", "you", "what", "can", "please", "dr"}


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Card:
    id: str
    kind: str  # clinic | doctor | treatment
    title: str
    text: str
    specialties: list[str] = field(default_factory=list)


class KnowledgeBase:
    def __init__(self, catalog: dict[str, Any]):
        self.clinics = {c["id"]: c for c in catalog["clinics"]}
        self.doctors = {d["id"]: d for d in catalog["doctors"]}
        self.treatments = {t["id"]: t for t in catalog["treatments"]}
        self.specialties = sorted({d["specialty"] for d in self.doctors.values()})
        self.cards = self._build_cards()
        self._bm25 = BM25Okapi([_tokens(c.title + " " + c.text) for c in self.cards])

    # ---------- cards ----------
    def _build_cards(self) -> list[Card]:
        cards: list[Card] = []
        for c in self.clinics.values():
            docs = [d for d in self.doctors.values() if d["clinicId"] == c["id"]]
            doc_list = ", ".join("{} ({}, {})".format(d["fullName"], d["specialty"], d["id"]) for d in docs)
            cards.append(
                Card(
                    id=c["id"],
                    kind="clinic",
                    title=c["name"],
                    text=(
                        f"Clinic {c['id']} {c['name']}. Address: {c['address']}. Phone: {c['phone']}. "
                        f"Opening hours: {c['operatingHours']}. Cancellation policy: {c['cancellationPolicy']} "
                        f"Emergency: {c['emergencyInstructions']} "
                        f"Doctors: {doc_list}."
                    ),
                    specialties=sorted({d["specialty"] for d in docs}),
                )
            )
        for d in self.doctors.values():
            clinic = self.clinics[d["clinicId"]]
            treats = [t for t in self.treatments.values() if d["specialty"] in t["providedBySpecialties"]]
            treat_list = ", ".join("{} ({})".format(t["name"], t["id"]) for t in treats) or "none listed"
            cards.append(
                Card(
                    id=d["id"],
                    kind="doctor",
                    title=d["fullName"],
                    text=(
                        f"Doctor {d['id']} {d['fullName']}, specialty {d['specialty']}, at {clinic['name']} ({clinic['id']}). "
                        f"Languages: {d['languagesSpoken']}. Consultation fee: ${d['consultationFeeUsd']:.2f}. "
                        f"Bio: {d['bio']} Treatments offered: "
                        f"{treat_list}."
                    ),
                    specialties=[d["specialty"]],
                )
            )
        for t in self.treatments.values():
            providers = [d for d in self.doctors.values() if d["specialty"] in t["providedBySpecialties"]]
            provider_list = ", ".join("{} ({})".format(d["fullName"], d["id"]) for d in providers) or "no listed doctor"
            cards.append(
                Card(
                    id=t["id"],
                    kind="treatment",
                    title=t["name"],
                    text=(
                        f"Treatment {t['id']} {t['name']}, category {t['category']}. Duration {t['durationMinutes']} minutes. "
                        f"Price ${t['priceUsd']:.2f}. Pre-treatment instructions: {t['preTreatmentInstructions']} "
                        f"Provided by: {provider_list}."
                    ),
                    specialties=list(t["providedBySpecialties"]),
                )
            )
        return cards

    # ---------- retrieval ----------
    def retrieve(self, query: str, specialty: str | None = None, k: int = 5) -> list[Card]:
        scores = self._bm25.get_scores(_tokens(query)) if _tokens(query) else [0.0] * len(self.cards)
        lexical = [i for i in sorted(range(len(self.cards)), key=lambda i: -scores[i]) if scores[i] > 0]
        semantic: list[int] = []
        if specialty:
            # doctors first, then treatments, then clinics that have the specialty
            order = {"doctor": 0, "treatment": 1, "clinic": 2}
            semantic = sorted(
                (i for i, c in enumerate(self.cards) if specialty in c.specialties),
                key=lambda i: order[self.cards[i].kind],
            )
        fused: dict[int, float] = {}
        for ranking in (lexical, semantic):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        ranked = sorted(fused, key=lambda i: -fused[i])[:k]
        return [self.cards[i] for i in ranked]

    def card(self, card_id: str) -> Card | None:
        return next((c for c in self.cards if c.id == card_id), None)

    # ---------- entity resolution (deterministic, used by the booking flow) ----------
    def resolve_doctor(self, ref: str | None) -> dict[str, Any] | None:
        if not ref:
            return None
        ref_l = ref.lower().replace("dr.", "").replace("dr ", "").strip()
        if ref.upper() in self.doctors:
            return self.doctors[ref.upper()]
        for d in self.doctors.values():
            name = d["fullName"].lower().replace("dr.", "").strip()
            parts = name.split()
            if ref_l == name or ref_l in parts or (len(ref_l) > 3 and ref_l in name):
                return d
        for d in self.doctors.values():  # "the dermatologist", "orthopedics doctor"
            if d["specialty"].lower()[:6] in ref_l:
                matches = [x for x in self.doctors.values() if x["specialty"] == d["specialty"]]
                return d if len(matches) == 1 else None
        return None

    def resolve_treatment(self, ref: str | None) -> dict[str, Any] | None:
        if not ref:
            return None
        if ref.upper() in self.treatments:
            return self.treatments[ref.upper()]
        ref_t = set(_tokens(ref))
        best, best_score = None, 0.0
        for t in self.treatments.values():
            name_t = set(_tokens(t["name"] + " " + t["category"]))
            overlap = len(ref_t & name_t)
            score = max(overlap / max(1, len(name_t)), overlap / max(1, len(ref_t)) * 0.5)
            if score > best_score:
                best, best_score = t, score
        return best if best_score >= 0.34 else None

    def treatments_for_doctor(self, doctor: dict[str, Any]) -> list[dict[str, Any]]:
        return [t for t in self.treatments.values() if doctor["specialty"] in t["providedBySpecialties"]]

    def doctors_for_treatment(self, treatment: dict[str, Any]) -> list[dict[str, Any]]:
        return [d for d in self.doctors.values() if d["specialty"] in treatment["providedBySpecialties"]]

    def doctors_for_specialty(self, specialty: str) -> list[dict[str, Any]]:
        return [d for d in self.doctors.values() if d["specialty"] == specialty]

    def normalize_specialty(self, value: str | None) -> str | None:
        if not value:
            return None
        v = value.lower()
        for s in self.specialties:
            if s.lower() == v or s.lower()[:6] in v or v[:6] in s.lower():
                return s
        return None
