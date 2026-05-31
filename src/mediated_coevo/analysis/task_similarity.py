"""Task metadata similarity helpers for graph precomputation."""

from __future__ import annotations

import itertools
import json
import math
import re
import statistics
import tomllib
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field


STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "all",
        "also",
        "an",
        "and",
        "are",
        "as",
        "at",
        "based",
        "be",
        "before",
        "below",
        "by",
        "can",
        "create",
        "each",
        "etc",
        "few",
        "file",
        "files",
        "for",
        "from",
        "given",
        "has",
        "have",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "must",
        "need",
        "no",
        "not",
        "of",
        "on",
        "or",
        "output",
        "per",
        "should",
        "than",
        "that",
        "the",
        "their",
        "then",
        "this",
        "to",
        "use",
        "using",
        "via",
        "when",
        "where",
        "which",
        "who",
        "whose",
        "will",
        "with",
        "within",
        "without",
        "write",
        "you",
        "your",
    }
)

DOMAIN_TERMS: frozenset[str] = frozenset(
    {
        "api",
        "bash",
        "bgp",
        "bibtex",
        "build",
        "cif",
        "citation",
        "ci",
        "coin",
        "control",
        "csv",
        "cve",
        "data",
        "debugging",
        "detection",
        "diff",
        "docker",
        "dot",
        "erlang",
        "excel",
        "finance",
        "form",
        "game",
        "gpu",
        "graph",
        "html",
        "java",
        "json",
        "llm",
        "maintenance",
        "mario",
        "maven",
        "modeling",
        "network",
        "numerical",
        "optimization",
        "parser",
        "patch",
        "pdf",
        "physics",
        "powerlifting",
        "python",
        "qa",
        "quantum",
        "report",
        "repository",
        "routing",
        "scheduling",
        "science",
        "seismic",
        "security",
        "simulation",
        "spreadsheet",
        "ssh",
        "threejs",
        "typescript",
        "vehicle",
        "visualization",
        "web",
        "xlsx",
        "yaml",
    }
)

OUTPUT_MARKERS: dict[str, tuple[str, ...]] = {
    "dot": (".dot", "digraph"),
    "csv": (".csv", "csv"),
    "html": (".html", "html", "web app"),
    "json": (".json", "json"),
    "markdown": (".md", "markdown"),
    "patch": ("patch", "diff"),
    "pdf": (".pdf", "pdf"),
    "xlsx": (".xlsx", "excel", "spreadsheet"),
    "yaml": (".yaml", ".yml", "yaml"),
}

CAPABILITY_MARKERS: dict[str, frozenset[str]] = {
    "build-debugging": frozenset(
        {
            "build",
            "ci",
            "cve",
            "debugging",
            "erlang",
            "maven",
            "patch",
            "security",
            "ssh",
        }
    ),
    "data-processing": frozenset({"csv", "data", "excel", "spreadsheet", "xlsx"}),
    "document-research": frozenset({"bibtex", "citation", "form", "pdf", "research"}),
    "optimization-routing": frozenset(
        {"optimization", "routing", "scheduling", "vehicle"}
    ),
    "simulation-modeling": frozenset(
        {"control", "numerical", "physics", "quantum", "simulation"}
    ),
    "structured-json-output": frozenset({"graph", "json", "report"}),
    "visualization-web": frozenset({"d3", "html", "threejs", "visualization", "web"}),
}


class ScoreWeights(BaseModel):
    """Weights for task metadata similarity components."""

    category: float = 0.30
    tags: float = 0.22
    skills: float = 0.18
    capability: float = 0.14
    io_shape: float = 0.10
    instruction_text: float = 0.06


class TaskSimilarityProfile(BaseModel):
    """Graph node ingredients extracted from a local SkillFlow task."""

    task_id: str
    category: str = ""
    difficulty: str = ""
    tags: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    environment_files: list[str] = Field(default_factory=list)
    output_types: list[str] = Field(default_factory=list)
    domain_terms: list[str] = Field(default_factory=list)
    capability_labels: list[str] = Field(default_factory=list)


class PairSimilarity(BaseModel):
    """Weighted metadata similarity between two task nodes."""

    source: str
    target: str
    score: float
    components: dict[str, float]
    shared: dict[str, list[str]]
    kept_after_p20_cut: bool = False
    kept_after_threshold_cut: bool = False


class TaskGraphPrecompute(BaseModel):
    """Precomputed graph ingredients and threshold-pruned edge set."""

    task_count: int
    pair_count: int
    score_weights: ScoreWeights
    percentile_cut: float
    p20_threshold: float
    edge_score_threshold: float | None = None
    active_threshold: float
    threshold_kind: str
    kept_edge_count: int
    cut_edge_count: int
    score_min: float
    score_max: float
    score_mean: float
    score_median: float
    components_after_cut: list[list[str]]
    profiles: dict[str, TaskSimilarityProfile]
    pairwise_similarity: list[PairSimilarity]


class _ProfileFeatures(BaseModel):
    profile: TaskSimilarityProfile
    category_terms: set[str]
    tag_terms: set[str]
    skill_terms: set[str]
    instruction_terms: set[str]
    output_terms: set[str]
    capability_terms: set[str]


def build_task_graph_precompute(
    tasks_root: Path,
    *,
    percentile_cut: float = 0.20,
    edge_score_threshold: float | None = None,
    score_weights: ScoreWeights | None = None,
) -> TaskGraphPrecompute:
    """Build metadata profiles, all pair scores, and threshold-pruned edges."""
    features = load_task_features(tasks_root)
    if not features:
        raise ValueError(f"no task directories found under {tasks_root}")

    weights = score_weights or ScoreWeights()
    pairwise_similarity = compute_pairwise_similarity(features, weights)
    p20_threshold = percentile_threshold(
        [pair.score for pair in pairwise_similarity],
        percentile_cut,
    )
    active_threshold = (
        edge_score_threshold if edge_score_threshold is not None else p20_threshold
    )
    threshold_kind = "absolute_score" if edge_score_threshold is not None else "p20"
    pruned_pairs = [
        pair.model_copy(
            update={
                "kept_after_p20_cut": pair.score >= p20_threshold,
                "kept_after_threshold_cut": pair.score >= active_threshold,
            }
        )
        for pair in pairwise_similarity
    ]
    kept_pairs = [pair for pair in pruned_pairs if pair.kept_after_threshold_cut]
    scores = [pair.score for pair in pruned_pairs]

    return TaskGraphPrecompute(
        task_count=len(features),
        pair_count=len(pruned_pairs),
        score_weights=weights,
        percentile_cut=percentile_cut,
        p20_threshold=p20_threshold,
        edge_score_threshold=edge_score_threshold,
        active_threshold=active_threshold,
        threshold_kind=threshold_kind,
        kept_edge_count=len(kept_pairs),
        cut_edge_count=len(pruned_pairs) - len(kept_pairs),
        score_min=min(scores),
        score_max=max(scores),
        score_mean=round(statistics.mean(scores), 6),
        score_median=round(statistics.median(scores), 6),
        components_after_cut=connected_components(features.keys(), kept_pairs),
        profiles={task_id: features[task_id].profile for task_id in sorted(features)},
        pairwise_similarity=sorted(
            pruned_pairs,
            key=lambda pair: (pair.source, pair.target),
        ),
    )


def load_task_features(tasks_root: Path) -> dict[str, _ProfileFeatures]:
    """Extract metadata-driven graph node features from local task directories."""
    features: dict[str, _ProfileFeatures] = {}
    for task_dir in sorted(tasks_root.iterdir(), key=lambda path: path.name):
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        features[task_dir.name] = _load_task_feature(task_dir)
    return features


def compute_pairwise_similarity(
    features: dict[str, _ProfileFeatures],
    score_weights: ScoreWeights,
) -> list[PairSimilarity]:
    """Compute weighted similarity for every unordered task pair."""
    pairs: list[PairSimilarity] = []
    for source, target in itertools.combinations(sorted(features), 2):
        source_features = features[source]
        target_features = features[target]
        components = _component_scores(source_features, target_features)
        score = round(
            score_weights.category * components["category"]
            + score_weights.tags * components["tags"]
            + score_weights.skills * components["skills"]
            + score_weights.capability * components["capability"]
            + score_weights.io_shape * components["io_shape"]
            + score_weights.instruction_text * components["instruction_text"],
            6,
        )
        pairs.append(
            PairSimilarity(
                source=source,
                target=target,
                score=score,
                components=components,
                shared=_shared_evidence(source_features, target_features),
            )
        )
    return pairs


def percentile_threshold(values: list[float], percentile: float) -> float:
    """Return a deterministic nearest-rank percentile threshold."""
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in the interval (0, 1]")
    if not values:
        raise ValueError("cannot compute a percentile over no values")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def connected_components(
    task_ids: Iterable[str],
    kept_pairs: list[PairSimilarity],
) -> list[list[str]]:
    """Return connected components induced by kept edges."""
    ids = set(task_ids)
    adjacency = {task_id: set[str]() for task_id in ids}
    for pair in kept_pairs:
        adjacency[pair.source].add(pair.target)
        adjacency[pair.target].add(pair.source)

    components: list[list[str]] = []
    seen: set[str] = set()
    for task_id in sorted(ids):
        if task_id in seen:
            continue
        stack = [task_id]
        component: list[str] = []
        seen.add(task_id)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def write_task_graph_artifacts(
    precompute: TaskGraphPrecompute, output_dir: Path
) -> None:
    """Write graph profile, pairwise score, and summary JSON artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "task_profiles.json",
        {
            "task_count": precompute.task_count,
            "profiles": {
                task_id: profile.model_dump(mode="json")
                for task_id, profile in precompute.profiles.items()
            },
        },
    )
    _write_json(
        output_dir / "pairwise_similarity.json",
        {
            "pair_count": precompute.pair_count,
            "percentile_cut": precompute.percentile_cut,
            "p20_threshold": precompute.p20_threshold,
            "edge_score_threshold": precompute.edge_score_threshold,
            "active_threshold": precompute.active_threshold,
            "threshold_kind": precompute.threshold_kind,
            "pairs": [
                pair.model_dump(mode="json") for pair in precompute.pairwise_similarity
            ],
        },
    )
    _write_json(
        output_dir / "graph_summary.json",
        precompute.model_dump(
            mode="json",
            exclude={"profiles", "pairwise_similarity"},
        ),
    )


def _load_task_feature(task_dir: Path) -> _ProfileFeatures:
    task_id = task_dir.name
    task_config = tomllib.loads((task_dir / "task.toml").read_text())
    metadata = task_config.get("metadata", {})
    instruction = (task_dir / "instruction.md").read_text(errors="ignore")
    test_text = _read_optional(task_dir / "tests" / "test_outputs.py")
    skill_names = _skill_names(task_dir)
    environment_files = _environment_files(task_dir)
    category = str(metadata.get("category", "")).strip()
    tags = [str(tag).strip() for tag in metadata.get("tags", [])]

    category_terms = _expand_terms([category])
    tag_terms = _expand_terms(tags)
    skill_terms = _expand_terms(skill_names)
    instruction_terms = _tokens(instruction)
    output_terms = _infer_output_types(instruction, test_text, environment_files)
    domain_terms = sorted(
        (
            category_terms
            | tag_terms
            | skill_terms
            | instruction_terms
            | _tokens(test_text)
            | output_terms
        )
        & DOMAIN_TERMS
    )
    capability_terms = _infer_capabilities(
        category_terms | tag_terms | skill_terms | instruction_terms | output_terms
    )

    profile = TaskSimilarityProfile(
        task_id=task_id,
        category=category,
        difficulty=str(metadata.get("difficulty", "")).strip(),
        tags=tags,
        skills=skill_names,
        environment_files=environment_files,
        output_types=sorted(output_terms),
        domain_terms=domain_terms,
        capability_labels=sorted(capability_terms),
    )
    return _ProfileFeatures(
        profile=profile,
        category_terms=category_terms,
        tag_terms=tag_terms,
        skill_terms=skill_terms,
        instruction_terms=instruction_terms,
        output_terms=output_terms,
        capability_terms=capability_terms,
    )


def _component_scores(
    source: _ProfileFeatures,
    target: _ProfileFeatures,
) -> dict[str, float]:
    source_category = _normalize(source.profile.category)
    target_category = _normalize(target.profile.category)
    if source_category and source_category == target_category:
        category_score = 1.0
    else:
        category_score = _jaccard(source.category_terms, target.category_terms)

    return {
        "category": round(category_score, 6),
        "tags": round(_jaccard(source.tag_terms, target.tag_terms), 6),
        "skills": round(_jaccard(source.skill_terms, target.skill_terms), 6),
        "capability": round(
            _jaccard(source.capability_terms, target.capability_terms),
            6,
        ),
        "io_shape": round(_jaccard(source.output_terms, target.output_terms), 6),
        "instruction_text": round(
            _jaccard(source.instruction_terms, target.instruction_terms),
            6,
        ),
    }


def _shared_evidence(
    source: _ProfileFeatures,
    target: _ProfileFeatures,
) -> dict[str, list[str]]:
    return {
        "capabilities": sorted(
            set(source.profile.capability_labels)
            & set(target.profile.capability_labels)
        ),
        "domain_terms": sorted(
            set(source.profile.domain_terms) & set(target.profile.domain_terms)
        ),
        "output_types": sorted(
            set(source.profile.output_types) & set(target.profile.output_types)
        ),
        "skills": sorted(set(source.profile.skills) & set(target.profile.skills)),
        "tags": sorted(set(source.profile.tags) & set(target.profile.tags)),
    }


def _skill_names(task_dir: Path) -> list[str]:
    skills_dir = task_dir / "environment" / "skills"
    return sorted(path.parent.name for path in skills_dir.glob("*/SKILL.md"))


def _environment_files(task_dir: Path) -> list[str]:
    environment_dir = task_dir / "environment"
    if not environment_dir.exists():
        return []
    return sorted(path.name for path in environment_dir.iterdir() if path.is_file())


def _infer_output_types(
    instruction: str,
    test_text: str,
    environment_files: list[str],
) -> set[str]:
    haystack = "\n".join([instruction, test_text, *environment_files]).lower()
    return {
        output_type
        for output_type, markers in OUTPUT_MARKERS.items()
        if any(marker in haystack for marker in markers)
    }


def _infer_capabilities(terms: set[str]) -> set[str]:
    return {
        capability
        for capability, markers in CAPABILITY_MARKERS.items()
        if terms & markers
    }


def _expand_terms(values: list[str]) -> set[str]:
    terms: set[str] = set()
    for value in values:
        normalized = _normalize(value)
        if normalized:
            terms.add(normalized)
            terms.update(normalized.split("-"))
        terms.update(_tokens(value))
    return terms


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOP_WORDS
    }


def _jaccard(source: set[str], target: set[str]) -> float:
    if not source and not target:
        return 0.0
    return len(source & target) / len(source | target)


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _read_optional(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="ignore")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
