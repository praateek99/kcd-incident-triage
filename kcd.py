"""
KCD: Knowledge-guided Cost-aware Deferral for cloud incident triage.

Core library: synthetic incident generator (Statuspage-style first updates),
simulated human reviewer, cost model, data-driven triage agent, knowledge-rule
layer, deferral scores and evaluation. Everything is seeded and deterministic.

Random streams are separated so that conditions can be compared on the same cases:
  * company severity offsets  -> company_offsets(seed)
  * latent incidents          -> generate_latent(n, seed, offsets)
  * data defects              -> apply_defects(df, rate, seed)   (common random numbers:
                                 the 60% defect set is a superset of the 30% set)
  * reviewer outcomes         -> simulate_reviewer(y, accuracy, seed), one per incident
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# 1. Domain vocabulary
# ---------------------------------------------------------------------------
LEVELS = ["none", "minor", "major", "critical"]          # Statuspage impact levels
K = len(LEVELS)

COMPONENTS = {  # business criticality of the affected service (0..1)
    "payments": 1.00, "auth": 1.00, "database": 0.90, "api": 0.80,
    "cdn": 0.70, "notifications": 0.40, "dashboard": 0.40, "docs": 0.10,
}
CORE = {"payments", "auth", "database"}
SYMPTOMS = {"outage": 1.00, "degraded": 0.55, "delay": 0.35, "cosmetic": 0.05}
SCOPES = {"all": 1.00, "multi_region": 0.80, "single_region": 0.50, "subset": 0.30}
STATUS_BY_SYMPTOM = {  # component status posted in the FIRST update (noisy)
    "outage":   (["major_outage", "partial_outage", "degraded_performance"], [0.60, 0.30, 0.10]),
    "degraded": (["partial_outage", "degraded_performance", "major_outage"], [0.35, 0.55, 0.10]),
    "delay":    (["degraded_performance", "partial_outage", "operational"],  [0.70, 0.15, 0.15]),
    "cosmetic": (["operational", "degraded_performance"],                    [0.75, 0.25]),
}
TEMPLATES = {
    "outage":   ["{c} is currently unavailable", "users are unable to access {c}",
                 "we are seeing a complete outage of {c}", "{c} is returning errors for all requests"],
    "degraded": ["we are seeing elevated error rates on {c}", "increased latency affecting {c}",
                 "some requests to {c} are failing intermittently"],
    "delay":    ["{c} processing is delayed", "we are investigating delays in {c}",
                 "{c} jobs are queued and running slowly"],
    "cosmetic": ["a display issue is affecting {c}", "we are investigating a minor visual issue with {c}"],
}
SCOPE_TEXT = {"all": "for all customers", "multi_region": "across multiple regions",
              "single_region": "in one region", "subset": "for a subset of customers"}
VAGUE = ["We are investigating reports of issues.", "We are currently investigating this issue.",
         "We are looking into a problem and will provide an update shortly."]
SECURITY_TEXT = ["We are also reviewing potential unauthorized access.",
                 "We are assessing possible data integrity impact."]
OUTAGE_WORDS = re.compile(r"unavailable|unable to access|complete outage|errors for all")
SECURITY_WORDS = re.compile(r"unauthorized|data integrity")
SYMPTOM_WORDS = re.compile(r"unavailable|unable|outage|error|latency|failing|delay|queued|slow|display|visual")
DEFECT_KINDS = ["vague", "no_component", "no_scope", "conflict"]


# ---------------------------------------------------------------------------
# 2. Synthetic incident generator
# ---------------------------------------------------------------------------
def company_offsets(seed: int, n_companies: int = 6, sd: float = 0.25) -> np.ndarray:
    """Company-specific severity offsets (each company applies its own SEV convention)."""
    return np.random.default_rng(seed).normal(0, sd, n_companies)


def generate_latent(n: int, seed: int, offsets: np.ndarray, noise_sd: float = 0.35) -> pd.DataFrame:
    """Clean incident-time fields plus the eventual impact label (no defects).

    Company identity is recorded but is NOT used as a model feature; the learner sees a
    pooled multi-company distribution.
    """
    rng = np.random.default_rng(seed)
    comp_names, sym_names, scope_names = list(COMPONENTS), list(SYMPTOMS), list(SCOPES)
    rows = []
    for i in range(n):
        co = int(rng.integers(len(offsets)))
        comp = comp_names[rng.choice(len(comp_names), p=[.10, .10, .10, .22, .10, .14, .16, .08])]
        sym = sym_names[rng.choice(4, p=[.20, .40, .25, .15])]
        scope = scope_names[rng.choice(4, p=[.12, .18, .35, .35])]
        security = bool(rng.random() < (0.10 if comp in CORE else 0.03))
        z = (1.6 * SYMPTOMS[sym] + 1.0 * COMPONENTS[comp] + 0.9 * SCOPES[scope]
             + 0.9 * security + offsets[co] + rng.normal(0, noise_sd))
        label = int(np.searchsorted([1.05, 1.95, 2.85], z))           # 0..3
        st_vals, st_p = STATUS_BY_SYMPTOM[sym]
        status = st_vals[rng.choice(len(st_vals), p=st_p)]
        text = rng.choice(TEMPLATES[sym]).format(c=comp) + " " + SCOPE_TEXT[scope] + "."
        sec_text = str(rng.choice(SECURITY_TEXT)) if security else ""
        if security:
            text += " " + sec_text
        vague = str(rng.choice(VAGUE)) + (" " + sec_text if security else "")
        rows.append(dict(id=i, company=f"C{co}", component=comp, scope=scope, status=status,
                         text=text, vague_text=vague, label=label))
    return pd.DataFrame(rows)


def apply_defects(df: pd.DataFrame, rate: float, seed: int) -> pd.DataFrame:
    """Inject incident-time data defects with an independent random stream.

    Common random numbers: each incident gets one uniform draw u and one pre-drawn defect
    pattern; it is defective when u < rate. A higher rate therefore corrupts a superset of
    the incidents corrupted at a lower rate, with identical defects on the shared cases.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    u = rng.random(n)
    n_def = rng.integers(1, 3, size=n)
    perms = np.array([rng.permutation(len(DEFECT_KINDS)) for _ in range(n)])
    out = df.copy()
    defects = [""] * n
    for i in np.where(u < rate)[0]:
        kinds = sorted(DEFECT_KINDS[k] for k in perms[i, :n_def[i]])
        for d in kinds:
            if d == "vague":
                out.at[i, "text"] = df.at[i, "vague_text"]
            elif d == "no_component":
                out.at[i, "component"] = "missing"
            elif d == "no_scope":
                out.at[i, "scope"] = "missing"
            elif d == "conflict":
                out.at[i, "status"] = "operational" if df.at[i, "status"] != "operational" else "major_outage"
        defects[i] = "+".join(kinds)
    out["defects"] = defects
    return out


# ---------------------------------------------------------------------------
# 3. Cost model and simulated reviewer (simulated, illustrative cost units)
# ---------------------------------------------------------------------------
@dataclass
class CostModel:
    under: float = 400.0       # per severity level under-triaged, scaled by (1 + true level)
    over: float = 300.0        # per severity level over-triaged (wasted escalation, paging)
    review: float = 120.0      # human review of one deferred incident
    human_acc: float = 0.92    # reviewer picks the true level, else an adjacent level

    def matrix(self) -> np.ndarray:
        C = np.zeros((K, K))
        for t in range(K):
            for p in range(K):
                C[t, p] = self.under * (t - p) * (1 + t) if p < t else self.over * (p - t)
        return C

    def reviewer_matrix(self) -> np.ndarray:
        """H[t, j] = P(reviewer chooses j | true level t): correct w.p. acc, otherwise a valid
        adjacent level (both neighbours equally likely in the interior; the single neighbour
        at the boundaries)."""
        H = np.zeros((K, K))
        e = 1 - self.human_acc
        for t in range(K):
            H[t, t] = self.human_acc
            nbrs = [j for j in (t - 1, t + 1) if 0 <= j < K]
            for j in nbrs:
                H[t, j] = e / len(nbrs)
        return H

    def human_cost_by_level(self) -> np.ndarray:
        """Expected reviewer error cost h(t) = sum_j H(j|t) C(t,j) for each true level."""
        return (self.reviewer_matrix() * self.matrix()).sum(1)


def simulate_reviewer(y: np.ndarray, cm: CostModel, seed: int) -> np.ndarray:
    """One reviewer outcome per incident, drawn from H; reused across all policies."""
    rng = np.random.default_rng(seed)
    H = cm.reviewer_matrix()
    u = rng.random(len(y))
    cum = H[y].cumsum(1)
    return (u[:, None] > cum).sum(1).clip(0, K - 1)


# ---------------------------------------------------------------------------
# 4. Data-driven triage agent
# ---------------------------------------------------------------------------
class TriageAgent:
    """TF-IDF (update text) + one-hot (component, scope, status) multinomial logistic regression.

    Probabilities are not post-hoc calibrated unless calibrate=True, in which case sigmoid
    calibration is fitted by 3-fold cross-validation on the TRAINING data only.
    """

    def __init__(self, seed: int = 0, calibrate: bool = False):
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.ohe = OneHotEncoder(handle_unknown="ignore")
        base = LogisticRegression(max_iter=2000, C=2.0, random_state=seed)
        self.clf = CalibratedClassifierCV(base, method="sigmoid", cv=3) if calibrate else base

    def _X(self, df, fit=False):
        cats = df[["component", "scope", "status"]]
        if fit:
            return hstack([self.tfidf.fit_transform(df.text), self.ohe.fit_transform(cats)]).tocsr()
        return hstack([self.tfidf.transform(df.text), self.ohe.transform(cats)]).tocsr()

    def fit(self, df):
        self.clf.fit(self._X(df, fit=True), df.label)
        return self

    def predict_proba(self, df) -> np.ndarray:
        P = np.zeros((len(df), K))
        P[:, self.clf.classes_] = self.clf.predict_proba(self._X(df))
        return P

    @staticmethod
    def decide(P: np.ndarray, C: np.ndarray):
        """Bayes decision: minimise expected cost. Returns (decision, expected risk r(x))."""
        exp_cost = P @ C
        return exp_cost.argmin(1), exp_cost.min(1)


# ---------------------------------------------------------------------------
# 5. Knowledge-rule layer (study-defined rules motivated by SRE practice)
# ---------------------------------------------------------------------------
RULES = {  # name: (bound, level, description)
    "KR1_security_floor":    ("floor", 2, "Security or data-integrity signal => at least major"),
    "KR2_core_outage_floor": ("floor", 2, "Outage language on a core service => at least major"),
    "KR3_status_floor":      ("floor", 2, "First update posts major_outage status => at least major"),
    "KR4_wide_core_outage":  ("floor", 3, "Core-service outage across regions or all customers => critical"),
    "KR5_cosmetic_ceiling":  ("ceiling", 1, "Cosmetic issue on a known non-core service, no observed "
                                            "security signal => at most minor"),
}


def knowledge_bounds(df: pd.DataFrame):
    """Return (lower bound, upper bound, fired-rule matrix) on the impact level of each incident.

    Rules read only incident-time fields and never the label. A missing field prevents the
    rules that depend on it from firing (an unknown component is neither core nor non-core).
    Floors are combined with max, ceilings with min, and floors take precedence.
    """
    text = df.text.str.lower()
    outage = text.str.contains(OUTAGE_WORDS).values
    sec = text.str.contains(SECURITY_WORDS).values
    cosmetic = text.str.contains("display|visual").values
    comp, scope, status = df.component.values, df.scope.values, df.status.values
    known_comp = np.isin(comp, list(COMPONENTS))
    core = np.isin(comp, list(CORE))
    wide = np.isin(scope, ["all", "multi_region"])
    fired = pd.DataFrame({
        "KR1_security_floor": sec,
        "KR2_core_outage_floor": core & outage,
        "KR3_status_floor": status == "major_outage",
        "KR4_wide_core_outage": core & outage & wide,
        "KR5_cosmetic_ceiling": cosmetic & known_comp & ~core & ~sec,
    })
    lo = np.zeros(len(df), int)
    hi = np.full(len(df), K - 1)
    for name, (kind, level, _) in RULES.items():
        f = fired[name].values
        if kind == "floor":
            lo = np.where(f, np.maximum(lo, level), lo)
        else:
            hi = np.where(f, np.minimum(hi, level), hi)
    hi = np.maximum(hi, lo)
    return lo, hi, fired


def rules_only_decision(df: pd.DataFrame, default: int = 1) -> np.ndarray:
    """Pure knowledge-based reference agent: default level clipped into the knowledge bounds."""
    lo, hi, _ = knowledge_bounds(df)
    return np.clip(np.full(len(df), default), lo, hi)


def constrained_decision(P: np.ndarray, C: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """Knowledge-constrained Bayes decision: minimise expected cost within [lo, hi].

    Under the same P, r*(x) >= r(x) always; the bounds can only lower REALISED cost when
    the probability model is wrong in the direction the rules correct.
    """
    exp_cost = P @ C
    idx = np.arange(C.shape[1])
    allowed = (idx[None, :] >= lo[:, None]) & (idx[None, :] <= hi[:, None])
    exp_cost = np.where(allowed, exp_cost, np.inf)
    return exp_cost.argmin(1), exp_cost.min(1)


def deferral_cost(P: np.ndarray, cm: CostModel) -> np.ndarray:
    """Expected cost of deferring: D(x) = c_r + sum_t P(t|x) h(t)."""
    return cm.review + P @ cm.human_cost_by_level()


def completeness_score(df: pd.DataFrame) -> np.ndarray:
    """Count of missing or uninformative incident-time fields."""
    has_sym = df.text.str.lower().str.contains(SYMPTOM_WORDS).values
    return (df.component.values == "missing").astype(int) + (df.scope.values == "missing") + (~has_sym)


# ---------------------------------------------------------------------------
# 6. Deferral and evaluation
# ---------------------------------------------------------------------------
def quota(budget: float, n: int) -> int:
    return int(math.ceil(budget * n - 1e-9))


def defer_top(score: np.ndarray, budget: float, tiebreak: np.ndarray) -> np.ndarray:
    """Defer exactly ceil(budget*n) incidents with the highest score (fixed per-incident tie-break)."""
    n = len(score)
    m = quota(budget, n)
    order = np.lexsort((tiebreak, -score))
    mask = np.zeros(n, bool)
    mask[order[:m]] = True
    return mask


def evaluate(y, decision, deferred, human, cm: CostModel) -> dict:
    """Cost of a policy given per-incident reviewer outcomes `human` (shared across policies)."""
    C = cm.matrix()
    final = np.where(deferred, human, decision)
    err_cost = C[y, final]
    severe = y >= 2
    ans = ~deferred
    return dict(
        deferred=deferred.mean(),
        answered_error=(decision[ans] != y[ans]).mean() if ans.any() else 0.0,
        severe_undertriage=((final < y) & severe).sum() / max(severe.sum(), 1),
        cost_per_incident=(err_cost.sum() + cm.review * deferred.sum()) / len(y),
    )


def calibration_metrics(P: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict:
    """Multiclass Brier score, log loss and top-label expected calibration error."""
    Y = np.eye(K)[y]
    brier = ((P - Y) ** 2).sum(1).mean()
    logloss = -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean()
    conf, pred = P.max(1), P.argmax(1)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = sum(abs((pred[bins == b] == y[bins == b]).mean() - conf[bins == b].mean()) * (bins == b).mean()
              for b in range(n_bins) if (bins == b).any())
    return dict(brier=brier, log_loss=logloss, ece=ece)
