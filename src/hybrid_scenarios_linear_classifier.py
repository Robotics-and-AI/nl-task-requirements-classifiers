"""
Task Instruction Classifier for Robot Scenarios

This script generates synthetic natural-language task instructions, encodes them
with a hybrid representation (sentence embeddings + handcrafted capability
features), and trains a Linear Discriminant Analysis (LDA) classifier to predict
the corresponding robot-task scenario.

Capability terminology
----------------------
The robot capability requirements use levels, not classes:
    requirements = [mobility_level, manipulation_level, payload_level]

Each capability level is an integer from 1 to 4. The word "class" is reserved
for the machine-learning classifier's scenario labels.

What it does:
- Generates train/eval task instructions from scenario templates
- Encodes each instruction with semantic + rule-based capability features
- Trains and evaluates a linear classifier
- Tests one random instruction
- Saves visualization plots:
  - analysis_robot_tasks.png
  - confusion_matrix.png
  - pca_variance.png

How to run:
1. Install dependencies:
   pip install numpy sentence-transformers scikit-learn matplotlib seaborn inflect torch pyyaml

2. Save this file, for example as:
   hybrid_scenarios_linear_classifier.py

3. Run it:
   python src/hybrid_scenarios_linear_classifier.py

Notes:
- The default sentence-transformers model is "sentence-transformers/all-mpnet-base-v2".
- The sentence-transformers model will be downloaded automatically the first time
  you run the script, so internet access is needed once.
- This code uses Linear Discriminant Analysis; it does not use Latent Dirichlet Allocation.
"""

import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import inflect
import numpy as np
import seaborn as sns
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
REQUIREMENT_ORDER = ("mobility_level", "manipulation_level", "payload_level")
CAPABILITY_LEVEL_LABELS = ("unknown", "level1", "level2", "level3", "level4")
PAYLOAD_LEVEL_THRESHOLDS_KG = (3.0, 5.0, 10.0)
MANIPULATION_LEVEL_THRESHOLDS_CM = (50.0, 90.0)
DEFAULT_SCENARIO_TEMPLATES_FILE = Path(__file__).resolve().parents[1] / "templates" / "scenarios.yaml"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "scenarios"
EXPECTED_EVAL_TEMPLATES_PER_SCENARIO = 5

# Global inflect engine. Kept for template/helper compatibility when numbers
# need to be converted to words in generated instructions.
_inflect = inflect.engine()


# ----------------------------------------------------------------------
# HYBRID ENCODER (semantic + capability levels)
# ----------------------------------------------------------------------


class TaskInstructionProcessor:
    # Regex helpers used by the 18-dimensional hybrid feature extractor.
    _kg_re = re.compile(r"(\d+(?:\.\d+)?)\s*(?:kg|kilogram|kilograms)\b", re.IGNORECASE)
    _cm_re = re.compile(r"(\d+(?:\.\d+)?)\s*(?:cm|centimeter|centimeters)\b", re.IGNORECASE)
    _meter_re = re.compile(r"\b(?:meter|meters)\b", re.IGNORECASE)

    @staticmethod
    def _extract_max_value(pattern: re.Pattern, text: str) -> Optional[float]:
        values = [float(m.group(1)) for m in pattern.finditer(text)]
        return max(values) if values else None

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        print(f"Loading hybrid encoder -> {model_name}")
        self.model = SentenceTransformer(model_name)

        # Get semantic dimension from a dummy encoding
        dummy = self.model.encode("test", convert_to_tensor=True)
        self.semantic_dim = dummy.shape[-1]

        # 3 (units) + 5 (payload) + 5 (manipulation) + 5 (mobility) = 18
        self.hybrid_features_count = 18
        self.embedding_dim = self.semantic_dim + self.hybrid_features_count

        print(f"Hybrid encoder ready: {self.semantic_dim} (semantic) + {self.hybrid_features_count} (hybrid) "
              f"= {self.embedding_dim} dims")

    def _extract_hybrid_features(self, text: str) -> np.ndarray:
        """
        Build small, human-interpretable capability features from the text.

        Output layout (18 dims total):
          - unit flags (3): [has_kg, has_cm, has_meter]
          - payload level one-hot (5): [unknown, level1, level2, level3, level4]
          - manipulation level one-hot (5): [unknown, level1(no manipulation), level2, level3, level4]
          - mobility level one-hot (5): [unknown, level1, level2, level3, level4]

        Notes:
        - "unknown" is used when the relevant cues are missing from the instruction text.
        - These heuristic features are derived from text; the ground-truth capability
          levels come from NLTaskGenerator.task_requirements_map.
        - Ground-truth requirements use this order:
          [mobility_level, manipulation_level, payload_level].
        """
        lower = text.lower()

        kg_val = self._extract_max_value(self._kg_re, text)
        cm_val = self._extract_max_value(self._cm_re, text)

        # 1) Unit flags (3)
        has_kg = 1.0 if kg_val is not None else 0.0
        has_cm = 1.0 if cm_val is not None else 0.0
        has_meter = 1.0 if self._meter_re.search(text) else 0.0
        unit_vec = np.array([has_kg, has_cm, has_meter], dtype=np.float32)

        # Helper: 5-way one-hot with unknown at index 0
        def onehot5_unknown() -> np.ndarray:
            v = np.zeros(5, dtype=np.float32)
            v[0] = 1.0  # unknown by default
            return v

        # 2) Payload level one-hot (5)
        # [unknown, level1, level2, level3, level4]
        payload = onehot5_unknown()
        if kg_val is not None:
            payload[:] = 0.0
            # Level mapping by payload weight:
            # level1 <= 3 kg, level2 <= 5 kg, level3 <= 10 kg, level4 > 10 kg.
            if kg_val <= PAYLOAD_LEVEL_THRESHOLDS_KG[0]:
                payload[1] = 1.0
            elif kg_val <= PAYLOAD_LEVEL_THRESHOLDS_KG[1]:
                payload[2] = 1.0
            elif kg_val <= PAYLOAD_LEVEL_THRESHOLDS_KG[2]:
                payload[3] = 1.0
            else:
                payload[4] = 1.0

        # 3) Manipulation level one-hot (5)
        # [unknown, level1(no manipulation), level2, level3, level4]
        manip = onehot5_unknown()

        no_manip_phrases = [
            "no manipulation", "no reach", "no arm", "fixed gripper not available", "cannot manipulate", "zero manipulation",
            "no object handling required", "handling actions not required", "without performing any manipulation",
            "without any pick and place actions", "no need for manipulation",
        ]
        if any(p in lower for p in no_manip_phrases):
            manip[:] = 0.0
            manip[1] = 1.0
        elif cm_val is not None:
            manip[:] = 0.0
            # level2 <= 50 cm, level3 <= 90 cm, level4 > 90 cm
            # (level1 is explicitly "no manipulation", handled above)
            if cm_val <= MANIPULATION_LEVEL_THRESHOLDS_CM[0]:
                manip[2] = 1.0
            elif cm_val <= MANIPULATION_LEVEL_THRESHOLDS_CM[1]:
                manip[3] = 1.0
            else:
                manip[4] = 1.0

        # 4) Mobility level one-hot (5): [unknown, level1, level2, level3, level4]
        mobility = onehot5_unknown()

        level1_keys = [
            "no mobility", "fixed position", "stationary", "no repositioning needed",
            "without relocating", "no change in location", "in place", "no relocation",
            "without moving", "no need for mobility", "without needing to move",
            "requiring no movement", "next to you"
        ]
        level2_keys = ["planar", "flat", "smooth"]
        level3_keys = [
            "high mobility", "narrow workspace", "high planar", "compact entryways",
            "limited space", "narrow spaces", "tight access", "confined paths",
            "narrow corridors", "compact areas", "limited passageways",
            "constrained space", "restricted paths", "narrow aisles",
            "tight passages", "compact access"
        ]
        level4_keys = [
            "uneven ground", "uneven terrain", "stairways", "stairs", "uneven floors",
            "irregular ground", "stepped surface", "ramps and stairs",
            "unstructured environment", "irregular floor", "discontinuous terrain",
            "steps", "uneven surfaces", "uneven corridors"
        ]

        if any(k in lower for k in level4_keys):
            mobility[:] = 0.0
            mobility[4] = 1.0
        elif any(k in lower for k in level3_keys):
            mobility[:] = 0.0
            mobility[3] = 1.0
        elif any(k in lower for k in level2_keys):
            mobility[:] = 0.0
            mobility[2] = 1.0
        elif any(k in lower for k in level1_keys):
            mobility[:] = 0.0
            mobility[1] = 1.0
        # else remains unknown

        # Concatenate: 3 + 5 + 5 + 5 = 18
        return np.concatenate([unit_vec, payload, manip, mobility], axis=0).astype(np.float32)

    def encode_instruction(self, instruction: str) -> np.ndarray:
        semantic = self.model.encode(
            instruction,
            convert_to_tensor=False,
            normalize_embeddings=True
        ).astype(np.float32)

        hybrid = self._extract_hybrid_features(instruction)
        return np.concatenate([semantic, hybrid])

    def encode_batch(self, instructions: List[str]) -> np.ndarray:
        semantics = self.model.encode(
            instructions,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_tensor=False
        ).astype(np.float32)

        hybrids = np.stack([self._extract_hybrid_features(text) for text in instructions])
        return np.concatenate([semantics, hybrids], axis=1)

    def get_embedding_dim(self) -> int:
        return self.embedding_dim


class NLTaskGenerator:
    """
    Generates task instructions + requirements from an external YAML template file.

    Expected YAML layout:
      scenarios:
        scenario1:
          requirements:
            mobility_level: 1
            manipulation_level: 2
            payload_level: 1
          ranges:
            weight_kg: [0.01, 3.0]
            reach_cm: [1, 50]
            x_m: [-20.0, 20.0]   # optional
            y_m: [-20.0, 20.0]   # optional
          options:
            units: [...]
            size_adjectives: [...]
            places: [...]
          templates:
            train: [...]
            eval: [...]

    generate_task(...) returns:
      {
        "instruction": str,
        "category": str,  # scenario label used by the classifier
        "requirements": np.ndarray shape(3,), dtype float32
      }

    The requirements vector stores capability levels in REQUIREMENT_ORDER:
      [mobility_level, manipulation_level, payload_level]
    """

    _PLACEHOLDER_RE = re.compile(r"\{([^}:]+)(?::[^}]*)?\}")
    _INDEXED_KEY_RE = re.compile(r"^(w|d|adj|place|units)(\d+)$")

    def __init__(self, templates_path: Optional[Any] = None):
        self.templates_path = Path(templates_path) if templates_path is not None else DEFAULT_SCENARIO_TEMPLATES_FILE
        self.config = self._load_yaml_config(self.templates_path)
        self.task_templates, self.task_requirements_map = self._normalize_yaml_config(self.config)

        # Ensure the scenario keys match between templates and the requirements map.
        template_keys = set(self.task_templates.keys())
        req_keys = set(self.task_requirements_map.keys())

        missing = template_keys - req_keys
        extra = req_keys - template_keys

        if missing:
            raise KeyError(f"Missing requirements for scenarios: {sorted(missing)}")
        if extra:
            raise KeyError(f"Requirements map has unknown scenarios: {sorted(extra)}")

        # Validate template placeholders and scenario range structure.
        self._validate_templates()

    @staticmethod
    def _load_yaml_config(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(
                f"Scenario template file not found: {path}\n"
                "Expected location: <repo-root>/templates/scenarios.yaml. You can also pass templates_path explicitly."
            )

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict) or "scenarios" not in data:
            raise ValueError("Template YAML must be a mapping with a top-level 'scenarios' key")
        if not isinstance(data["scenarios"], dict) or not data["scenarios"]:
            raise ValueError("Template YAML must define at least one scenario")

        return data

    @staticmethod
    def _normalize_yaml_config(config: Dict[str, Any]) -> tuple[Dict[str, Dict[str, Any]], Dict[str, List[int]]]:
        """Convert the YAML schema into the internal structure used by the generator."""
        task_templates: Dict[str, Dict[str, Any]] = {}
        task_requirements_map: Dict[str, List[int]] = {}
        errors: List[str] = []

        for scenario_name, raw in config["scenarios"].items():
            if not isinstance(raw, dict):
                errors.append(f"{scenario_name}: scenario entry must be a mapping")
                continue

            ranges = raw.get("ranges", {})
            options = raw.get("options", {})
            templates = raw.get("templates", {})
            requirements = raw.get("requirements", {})
            declared_counts = raw.get("template_counts", {})

            try:
                scenario = {
                    "weight_range": tuple(ranges["weight_kg"]),
                    "distance_range": tuple(ranges["reach_cm"]),
                    "units": list(options.get("units", [1])),
                    "size_adjectives": list(options.get("size_adjectives", ["medium"])),
                    "place": list(options.get("places", ["station"])),
                    "train": list(templates["train"]),
                    "eval": list(templates["eval"]),
                }
            except KeyError as exc:
                errors.append(f"{scenario_name}: missing required YAML field {exc}")
                continue
            except TypeError as exc:
                errors.append(f"{scenario_name}: invalid YAML field type: {exc}")
                continue

            for option_name in ("units", "size_adjectives", "place"):
                if not scenario.get(option_name):
                    errors.append(f"{scenario_name}: options.{option_name} must be a non-empty list")

            if "x_m" in ranges or "y_m" in ranges:
                if "x_m" not in ranges or "y_m" not in ranges:
                    errors.append(f"{scenario_name}: x_m and y_m must be provided together")
                else:
                    scenario["X_range"] = tuple(ranges["x_m"])
                    scenario["Y_range"] = tuple(ranges["y_m"])

            train_count = len(scenario["train"])
            eval_count = len(scenario["eval"])

            if declared_counts:
                if declared_counts.get("train") != train_count:
                    errors.append(
                        f"{scenario_name}: declared train count {declared_counts.get('train')} "
                        f"does not match actual count {train_count}"
                    )
                if declared_counts.get("eval") != eval_count:
                    errors.append(
                        f"{scenario_name}: declared eval count {declared_counts.get('eval')} "
                        f"does not match actual count {eval_count}"
                    )

            if eval_count != EXPECTED_EVAL_TEMPLATES_PER_SCENARIO:
                errors.append(
                    f"{scenario_name}: expected {EXPECTED_EVAL_TEMPLATES_PER_SCENARIO} eval templates, got {eval_count}"
                )

            try:
                task_requirements_map[scenario_name] = [
                    int(requirements[name]) for name in REQUIREMENT_ORDER
                ]
            except KeyError as exc:
                errors.append(f"{scenario_name}: missing requirement field {exc}")
                continue
            except (TypeError, ValueError) as exc:
                errors.append(f"{scenario_name}: invalid requirement value: {exc}")
                continue

            task_templates[scenario_name] = scenario

        if errors:
            msg = "Template YAML normalization failed:\n" + "\n".join(f"- {e}" for e in errors[:80])
            if len(errors) > 80:
                msg += f"\n...and {len(errors) - 80} more."
            raise ValueError(msg)

        return task_templates, task_requirements_map

    # ----------------------------------------------------------------------
    # Placeholder sampling + template filling
    # ----------------------------------------------------------------------
    @staticmethod
    def _sample_placeholders(scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a sampler for placeholders used inside templates.

        Supported placeholders (base keys):
          {w}     -> weight float
          {d}     -> distance int
          {adj}   -> adjective from size_adjectives
          {place} -> location string
          {units} -> integer from "units" list
          {place_src}, {place_dst} -> paired locations (stable within an instruction)
          {x1},{y1},{x2},{y2} -> coordinates if X_range/Y_range exist

        Also supports indexed variants for diversity:
          {adj1}, {adj2}, ...  (each index sampled independently, but stable within the instruction)
          {w1}, {w2}, ...
          {d1}, {d2}, ...
          {place1}, {place2}, ...
          {units1}, {units2}, ...
        """
        w_min, w_max = scenario["weight_range"]
        d_min, d_max = scenario["distance_range"]

        adj_list = scenario.get("size_adjectives", ["medium"])
        place_list = scenario.get("place", ["station"])
        units_list = scenario.get("units", [1])

        # Pre-sample paired places so {place_src}/{place_dst} are consistent.
        if len(place_list) >= 2:
            place_src_val, place_dst_val = random.sample(place_list, 2)
        else:
            place_src_val = place_dst_val = place_list[0]

        # Optional coordinates.
        x1_val = y1_val = x2_val = y2_val = None
        if "X_range" in scenario and "Y_range" in scenario:
            x1_val = float(round(random.uniform(*scenario["X_range"]), 2))
            y1_val = float(round(random.uniform(*scenario["Y_range"]), 2))
            x2_val = float(round(random.uniform(*scenario["X_range"]), 2))
            y2_val = float(round(random.uniform(*scenario["Y_range"]), 2))

        cache: Dict[str, Any] = {}

        # Base generators.
        def gen_w() -> float: return float(round(random.uniform(w_min, w_max), 2))
        def gen_d() -> int: return int(random.randint(d_min, d_max))
        def gen_adj() -> str: return str(random.choice(adj_list))
        def gen_place() -> str: return str(random.choice(place_list))
        def gen_units() -> int: return int(random.choice(units_list))

        # Fixed/paired generators.
        def gen_place_src() -> str: return str(place_src_val)
        def gen_place_dst() -> str: return str(place_dst_val)
        def gen_x1(): return x1_val
        def gen_y1(): return y1_val
        def gen_x2(): return x2_val
        def gen_y2(): return y2_val

        base_generators = {
            "w": gen_w,
            "d": gen_d,
            "adj": gen_adj,
            "place": gen_place,
            "units": gen_units,
            "place_src": gen_place_src,
            "place_dst": gen_place_dst,
            "x1": gen_x1, "y1": gen_y1, "x2": gen_x2, "y2": gen_y2,
        }

        def get(key: str):
            if key in cache:
                return cache[key]

            fn = base_generators.get(key)
            if fn is None:
                m = NLTaskGenerator._INDEXED_KEY_RE.match(key)
                if m:
                    base = m.group(1)
                    fn = base_generators.get(base)

            val = fn() if fn else None
            cache[key] = val
            return val

        return {"_get": get}

    @staticmethod
    def _fill_template(template: str, sampler: Dict[str, Any]) -> str:
        """Replace placeholders like {w:.2f}, {d}, {adj1}, {place_dst}, {x1:.2f}, {units}."""
        get = sampler["_get"]

        def replace(match):
            key = match.group(1)
            fmt = match.group(2) or ""
            value = get(key)

            if value is None:
                return match.group(0)

            try:
                return f"{value:{fmt}}" if fmt else str(value)
            except Exception:
                return str(value)

        return re.sub(r"\{([^}:]+)(?::([^}]*))?\}", replace, template)

    def _validate_templates(self) -> None:
        """
        Validate scenario structure and ensure every placeholder used in templates is supported.
        Also ensures x/y placeholders are only used when X_range/Y_range exist.
        Raises ValueError with a helpful message if something is wrong.
        """
        allowed_base = {
            "w", "d", "adj", "place", "units",
            "place_src", "place_dst",
            "x1", "y1", "x2", "y2",
        }
        coord_keys = {"x1", "y1", "x2", "y2"}

        errors: List[str] = []

        def _is_num(x) -> bool:
            return isinstance(x, (int, float, np.integer, np.floating))

        def _check_range(name: str, rng, scenario_name: str):
            if not (isinstance(rng, (tuple, list)) and len(rng) == 2):
                errors.append(f"{scenario_name}: {name} must be a (min,max) tuple/list, got {rng}")
                return
            lo, hi = rng[0], rng[1]
            if not (_is_num(lo) and _is_num(hi)):
                errors.append(f"{scenario_name}: {name} values must be numeric, got {rng}")
                return
            if float(lo) > float(hi):
                errors.append(f"{scenario_name}: {name} must satisfy min <= max, got {rng}")

        for scenario_name, scenario in self.task_templates.items():
            for required in ("weight_range", "distance_range", "train", "eval"):
                if required not in scenario:
                    errors.append(f"{scenario_name}: missing required field '{required}'")

            if "weight_range" in scenario:
                _check_range("weight_range", scenario.get("weight_range"), scenario_name)
            if "distance_range" in scenario:
                _check_range("distance_range", scenario.get("distance_range"), scenario_name)

            for split in ("train", "eval"):
                templates = scenario.get(split)
                if not isinstance(templates, list) or len(templates) == 0:
                    errors.append(f"{scenario_name}.{split}: must be a non-empty list of strings")
                elif not all(isinstance(t, str) for t in templates):
                    errors.append(f"{scenario_name}.{split}: all templates must be strings")

            if isinstance(scenario.get("eval"), list) and len(scenario["eval"]) != EXPECTED_EVAL_TEMPLATES_PER_SCENARIO:
                errors.append(
                    f"{scenario_name}.eval: expected {EXPECTED_EVAL_TEMPLATES_PER_SCENARIO} templates, "
                    f"got {len(scenario['eval'])}"
                )

            has_x = "X_range" in scenario
            has_y = "Y_range" in scenario
            if has_x != has_y:
                errors.append(f"{scenario_name}: X_range and Y_range must be provided together")

            has_coords = has_x and has_y
            if has_coords:
                _check_range("X_range", scenario.get("X_range"), scenario_name)
                _check_range("Y_range", scenario.get("Y_range"), scenario_name)

            for split in ("train", "eval"):
                templates = scenario.get(split, [])
                if not isinstance(templates, list):
                    continue

                for idx, tmpl in enumerate(templates):
                    if not isinstance(tmpl, str):
                        continue

                    keys = self._PLACEHOLDER_RE.findall(tmpl)
                    for key in keys:
                        if key in allowed_base:
                            if key in coord_keys and not has_coords:
                                errors.append(
                                    f"{scenario_name}.{split}[{idx}]: uses {{{key}}} but scenario has no X_range/Y_range\n"
                                    f"  Template: {tmpl}"
                                )
                            continue

                        if self._INDEXED_KEY_RE.match(key):
                            continue

                        errors.append(
                            f"{scenario_name}.{split}[{idx}]: unknown placeholder {{{key}}}\n"
                            f"  Template: {tmpl}"
                        )

        if errors:
            msg = "Template/scenario validation failed:\n" + "\n".join(f"- {e}" for e in errors[:80])
            if len(errors) > 80:
                msg += f"\n...and {len(errors) - 80} more."
            raise ValueError(msg)

    def generate_task(self, category: str, train_mode: bool = True) -> Dict[str, Any]:
        """
        Generate one task for the given category.

        Args:
          category: scenario key (e.g., "scenario1")
          train_mode:
            True  -> sample scenario["train"] templates
            False -> sample scenario["eval"] templates
        """
        if category not in self.task_templates:
            raise ValueError(f"Unknown category: {category}")

        scenario = self.task_templates[category]
        if "train" not in scenario or "eval" not in scenario:
            raise ValueError(f"Scenario '{category}' must define both 'train' and 'eval' template lists")

        templates = scenario["train"] if train_mode else scenario["eval"]
        if not templates:
            raise ValueError(
                f"No templates for category '{category}' in {'train' if train_mode else 'eval'} mode"
            )

        template = random.choice(templates)
        instruction = self._fill_template(template, self._sample_placeholders(scenario))

        # Convert the scenario's capability levels to a numeric vector.
        requirements = np.asarray(self.task_requirements_map[category], dtype=np.float32)

        return {
            "instruction": instruction,
            "category": category,
            "requirements": requirements,
        }

    def generate_task_random(self, train_mode: bool = True) -> Dict[str, Any]:
        category = random.choice(list(self.task_templates.keys()))
        return self.generate_task(category, train_mode=train_mode)


# ----------------------------------------------------------------------
# GLOBAL CONFIG & DATASET BUILDER
# ----------------------------------------------------------------------
TRAIN_PER_SCENARIO = 500
EVAL_PER_SCENARIO = 125
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
# ----------------------------------------------------------------------


def scenario_sort_key(category: str):
    """Sort labels like scenario1, scenario2, ..., scenario10 numerically."""
    match = re.search(r"\d+$", category)
    return (0, int(match.group())) if match else (1, category)


def build_dataset():
    generator = NLTaskGenerator()
    encoder = TaskInstructionProcessor()

    cats = sorted(generator.task_templates.keys(), key=scenario_sort_key)
    n_scenarios = len(cats)

    print(f"Generating dataset: {n_scenarios} scenarios x ("
          f"{TRAIN_PER_SCENARIO} train + {EVAL_PER_SCENARIO} eval ) = "
          f"{n_scenarios * (TRAIN_PER_SCENARIO + EVAL_PER_SCENARIO)} samples")

    train_tasks = [generator.generate_task(c, True)  for c in cats for _ in range(TRAIN_PER_SCENARIO)]
    eval_tasks  = [generator.generate_task(c, False) for c in cats for _ in range(EVAL_PER_SCENARIO)]

    X_train = encoder.encode_batch([t["instruction"] for t in train_tasks])
    X_eval  = encoder.encode_batch([t["instruction"] for t in eval_tasks])
    y_train = np.array([t["category"] for t in train_tasks])
    y_eval  = np.array([t["category"] for t in eval_tasks])

    return X_train, y_train, X_eval, y_eval, encoder, n_scenarios


# ----------------------------------------------------------------------
# MAIN + DYNAMIC PLOTS
# ----------------------------------------------------------------------
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X_tr, y_tr, X_ev, y_ev, encoder, n_scenarios = build_dataset()
    print(f"Dataset ready: {X_tr.shape} train, {X_ev.shape} eval, {n_scenarios} scenarios")

    # ------------------------------------------------------------------
    # RAW EMBEDDING vs HYBRID COMPARISON
    # ------------------------------------------------------------------
    X_tr_raw = X_tr[:, :encoder.semantic_dim]
    X_ev_raw = X_ev[:, :encoder.semantic_dim]

    # Raw embedding only
    lda_raw = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda_raw.fit(X_tr_raw, y_tr)
    raw_acc = accuracy_score(y_ev, lda_raw.predict(X_ev_raw))
    print(f"Raw Embedding LDA Accuracy: {raw_acc:.4f}")

    # Hybrid (embedding + handcrafted features)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(X_tr, y_tr)
    hybrid_acc = accuracy_score(y_ev, lda.predict(X_ev))
    print(f"Hybrid LDA Accuracy:        {hybrid_acc:.4f}")
    print(f"Hybrid gain over raw:       {hybrid_acc - raw_acc:+.4f}")

    # PCA + LDA on raw embedding only
    pca_raw = PCA(n_components=128, whiten=True, random_state=42)
    X_tr_raw_pca = pca_raw.fit_transform(X_tr_raw)
    X_ev_raw_pca = pca_raw.transform(X_ev_raw)
    lda_raw_pca = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda_raw_pca.fit(X_tr_raw_pca, y_tr)
    raw_pca_acc = accuracy_score(y_ev, lda_raw_pca.predict(X_ev_raw_pca))
    print(f"Raw PCA(128D)+LDA Accuracy: {raw_pca_acc:.4f}")
    print(f"Raw PCA retained variance:  {np.sum(pca_raw.explained_variance_ratio_):.4%}")

    # PCA + LDA on hybrid features
    pca = PCA(n_components=128, whiten=True, random_state=42)
    X_tr_pca = pca.fit_transform(X_tr)
    X_ev_pca = pca.transform(X_ev)
    lda_pca = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda_pca.fit(X_tr_pca, y_tr)
    hybrid_pca_acc = accuracy_score(y_ev, lda_pca.predict(X_ev_pca))
    print(f"Hybrid PCA(128D)+LDA Accuracy: {hybrid_pca_acc:.4f}")
    print(f"Hybrid PCA retained variance:  {np.sum(pca.explained_variance_ratio_):.4%}")
    print(f"Hybrid PCA gain over raw PCA:  {hybrid_pca_acc - raw_pca_acc:+.4f}")

    print("\nSingle test:")
    sample = NLTaskGenerator().generate_task_random()
    vec_hybrid = encoder.encode_instruction(sample["instruction"]).reshape(1, -1)
    vec_raw = vec_hybrid[:, :encoder.semantic_dim]

    print(f"Instruction: {sample['instruction']}")
    print(f"True       : {sample['category']}")
    print(f"Pred raw   : {lda_raw.predict(vec_raw)[0]}")
    print(f"Pred hybrid: {lda.predict(vec_hybrid)[0]}")

    #DYNAMIC PLOTS
    plt.style.use('seaborn-v0_8-whitegrid')

    category_list = sorted(set(y_tr), key=scenario_sort_key)
    label_to_id = {cat: i for i, cat in enumerate(category_list)}

    # Dynamic palette
    base_colors = np.vstack([plt.cm.tab20(np.linspace(0,1,20)),
                             plt.cm.tab20b(np.linspace(0,1,20)),
                             plt.cm.Set3(np.linspace(0,1,12))])
    palette = mcolors.ListedColormap(base_colors[:n_scenarios])

    # t-SNE
    print("\nComputing t-SNE...")
    tsne = TSNE(n_components=2, perplexity=40, init='pca', learning_rate='auto',
                random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_ev[:3000])
    colors_num = [label_to_id[c] for c in y_ev[:3000]]

    plt.figure(figsize=(15, 12))
    plt.scatter(X_tsne[:,0], X_tsne[:,1], c=colors_num, cmap=palette,
                s=250, edgecolor='k', linewidth=0.5, alpha=0.9)
    plt.title(f'Analysis of Robot Task Instructions\n{n_scenarios} Scenarios', fontsize=22, pad=20)
    cbar = plt.colorbar(ticks=range(0, n_scenarios, max(1, n_scenarios//10)))
    cbar.set_label('Scenario ID')
    cbar.ax.set_yticklabels([f'S{i+1}' for i in range(0, n_scenarios, max(1, n_scenarios//10))])
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / 'analysis_robot_tasks.png', dpi=400)
    plt.show()

    # Confusion Matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_ev, lda.predict(X_ev), labels=category_list)
    plt.figure(figsize=(max(14, n_scenarios//3), max(12, n_scenarios//3)))
    sns.heatmap(cm, cmap="Blues", cbar_kws={'shrink': 0.8},
                xticklabels=[f'S{i+1}' for i in range(n_scenarios)],
                yticklabels=[f'S{i+1}' for i in range(n_scenarios)])
    plt.title(f'Confusion Matrix ({n_scenarios} scenarios)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=300)
    plt.show()

    # PCA curve (unchanged)
    print("Generating PCA variance curve...")
    pca_full = PCA(n_components=256, random_state=42)
    pca_full.fit(X_tr)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(cum_var)+1), cum_var, 'b-', linewidth=2.5)
    plt.axvline(128, color='red', linestyle='--', label=f'128 → {cum_var[127]:.1%}')
    plt.xlabel('Components')
    plt.ylabel('Cumulative Variance')
    plt.title('PCA Explained Variance')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / 'pca_variance.png', dpi=300)
    plt.show()

    print(f'Plots saved in: {OUTPUT_DIR}')
