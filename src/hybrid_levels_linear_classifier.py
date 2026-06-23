"""
Attribute-wise Task Instruction Classifier for Robot Scenarios

This script generates synthetic natural-language task instructions, encodes them
with a hybrid representation (sentence embeddings + handcrafted rule-based features),
and trains three independent Linear Discriminant Analysis (LDA) classifiers to predict:

- mobility level
- manipulation level
- payload level

What it does:
- Generates train/eval task instructions from scenario templates
- Encodes each instruction using semantic + hybrid features
- Trains one LDA model per attribute
- Evaluates attribute-wise accuracy
- Tests one random instruction
- Saves analysis and confusion-matrix visualizations in outputs/levels/

How to run:
1. Install dependencies:
   python -m pip install numpy sentence-transformers scikit-learn matplotlib seaborn inflect torch pyyaml

2. Save this file, for example as:
   hybrid_levels_linear_classifier.py

3. Run it:
   python src/hybrid_levels_linear_classifier.py

Notes:
- The sentence-transformers model will be downloaded automatically the first time.
- This script trains three separate classifiers, not one classifier.
- Scenario templates are loaded from templates/scenarios.yaml.
- This code uses Linear Discriminant Analysis; it does not use Latent Dirichlet Allocation.
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import inflect
import os
import yaml

# ----------------------------------------------------------------------
# GLOBAL SETTINGS
# ----------------------------------------------------------------------
# Keep the inflect engine available for template wording/number-to-word helpers.
_inflect = inflect.engine()

random.seed(42)
np.random.seed(42)

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
ATTRIBUTES = ("mobility", "manipulation", "payload")
LEVEL_VALUES = (1, 2, 3, 4)

# Number of synthetic examples generated for each scenario template.
TRAIN_PER_SCENARIO = 500
EVAL_PER_SCENARIO = 125

# The requirements vector always uses this order in scenarios.yaml.
REQUIREMENT_ORDER = ("mobility_level", "manipulation_level", "payload_level")
DEFAULT_SCENARIO_TEMPLATES_FILE = Path(__file__).resolve().parents[1] / "templates" / "scenarios.yaml"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "levels"
EXPECTED_EVAL_TEMPLATES_PER_SCENARIO = 5

# Capability thresholds used by the handcrafted feature extractor.
PAYLOAD_LEVEL_THRESHOLDS_KG = (3.0, 5.0, 10.0)
MANIPULATION_LEVEL_THRESHOLDS_CM = (50.0, 90.0)

# ----------------------------------------------------------------------
# HYBRID ENCODER
# ----------------------------------------------------------------------
class TaskInstructionProcessor:
    _kg_re = re.compile(r"(\d+(?:\.\d+)?)\s*(?:kg|kilogram|kilograms)\b", re.IGNORECASE)
    _cm_re = re.compile(r"(\d+(?:\.\d+)?)\s*(?:cm|centimeter|centimeters)\b", re.IGNORECASE)
    _meter_re = re.compile(r"\b(?:meter|meters)\b", re.IGNORECASE)

    @staticmethod
    def _extract_max_value(pattern: re.Pattern, text: str):
        values = [float(m.group(1)) for m in pattern.finditer(text)]
        return max(values) if values else None

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        print(f"Loading hybrid encoder → {model_name}")
        self.model = SentenceTransformer(model_name)
        dummy = self.model.encode("test", convert_to_tensor=True)
        self.semantic_dim = dummy.shape[-1]
        self.hybrid_features_count = 18
        self.embedding_dim = self.semantic_dim + self.hybrid_features_count
        print(f"Hybrid encoder ready: {self.semantic_dim} (semantic) + {self.hybrid_features_count} (hybrid) = {self.embedding_dim} dims")

    def _extract_hybrid_features(self, text: str) -> np.ndarray:
        """
        Build small, human-interpretable features from the text.

        Output layout (18 dims total):
          - unit flags (3): [has_kg, has_cm, has_meter]
          - payload one-hot (5): [unknown, level1, level2, level3, level4]
          - manipulation one-hot (5): [unknown, level1(no manipulation), level2, level3, level4]
          - mobility one-hot (5): [unknown, level1, level2, level3, level4]

        Notes:
        - "unknown" is used when the relevant cues are missing from the instruction text.
        - These are heuristic features; the *ground truth* task requirement levels come from NLTaskGenerator.
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

        # 2) Payload one-hot (5)
        # [unknown, level1, level2, level3, level4]
        payload = onehot5_unknown()
        if kg_val is not None:
            payload[:] = 0.0
            if kg_val <= PAYLOAD_LEVEL_THRESHOLDS_KG[0]:
                payload[1] = 1.0
            elif kg_val <= PAYLOAD_LEVEL_THRESHOLDS_KG[1]:
                payload[2] = 1.0
            elif kg_val <= PAYLOAD_LEVEL_THRESHOLDS_KG[2]:
                payload[3] = 1.0
            else:
                payload[4] = 1.0

        # 3) Manipulation one-hot (5)
        # [unknown, level1(no manipulation), level2, level3, level4]
        manip = onehot5_unknown()

        # Optional: detect explicit "no manipulation" phrasing
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
            # Your original levels:
            # level2 <=50cm, level3 <=90cm, level4 >90cm
            # (level1 is explicitly "no manipulation", handled above)
            if cm_val <= MANIPULATION_LEVEL_THRESHOLDS_CM[0]:
                manip[2] = 1.0
            elif cm_val <= MANIPULATION_LEVEL_THRESHOLDS_CM[1]:
                manip[3] = 1.0
            else:
                manip[4] = 1.0

        # 4) Mobility one-hot (5): [unknown, level1, level2, level3, level4]
        #
        # Mobility level semantics (as used by your templates):
        # 1 = stationary/no mobility
        # 2 = planar/open-area mobility
        # 3 = high maneuverability in constrained spaces (tight corridors, narrow aisles)
        # 4 = uneven terrain / stairs / obstacles
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
        semantic = self.model.encode(instruction, convert_to_tensor=False, normalize_embeddings=True).astype(np.float32)
        hybrid = self._extract_hybrid_features(instruction)
        return np.concatenate([semantic, hybrid])

    def encode_batch(self, instructions: List[str]) -> np.ndarray:
        semantics = self.model.encode(instructions, batch_size=64, show_progress_bar=True,
                                      normalize_embeddings=True, convert_to_tensor=False).astype(np.float32)
        hybrids = np.stack([self._extract_hybrid_features(text) for text in instructions])
        return np.concatenate([semantics, hybrids], axis=1)


# ----------------------------------------------------------------------
# FULL TASK GENERATOR (all 46 scenarios)
# ----------------------------------------------------------------------

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
        "category": str,  # scenario label used to derive capability levels
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




def scenario_sort_key(category: str):
    """Sort labels like scenario1, scenario2, ..., scenario10 numerically."""
    match = re.search(r"\d+$", category)
    return (0, int(match.group())) if match else (1, category)

# ----------------------------------------------------------------------
# Helper to build per-attribute labels
# ----------------------------------------------------------------------
def build_attribute_labels(scenario_names, gen=None):
    gen = gen or NLTaskGenerator()
    lookup = {}
    for sc in scenario_names:
        req = gen.task_requirements_map[sc]
        lookup[sc] = {
            "mobility": req[0],
            "manipulation": req[1],
            "payload": req[2],
        }
    return lookup

# ----------------------------------------------------------------------
# Dataset builder
# ----------------------------------------------------------------------
def build_dataset():
    gen = NLTaskGenerator()
    encoder = TaskInstructionProcessor()
    cats = sorted(gen.task_templates.keys(), key=scenario_sort_key)
    print(f"Found {len(cats)} scenario templates")

    train_tasks = [gen.generate_task(c, True) for c in cats for _ in range(TRAIN_PER_SCENARIO)]
    eval_tasks  = [gen.generate_task(c, False) for c in cats for _ in range(EVAL_PER_SCENARIO)]

    X_train = encoder.encode_batch([t["instruction"] for t in train_tasks])
    X_eval  = encoder.encode_batch([t["instruction"] for t in eval_tasks])

    scenario_train = [t["category"] for t in train_tasks]
    scenario_eval  = [t["category"] for t in eval_tasks]

    label_lookup = build_attribute_labels(cats, gen)

    y_train = {attr: np.array([label_lookup[s][attr] for s in scenario_train])
               for attr in ATTRIBUTES}
    y_eval  = {attr: np.array([label_lookup[s][attr] for s in scenario_eval])
               for attr in ATTRIBUTES}

    return X_train, X_eval, y_train, y_eval, encoder

# ----------------------------------------------------------------------
# Training & evaluation
# ----------------------------------------------------------------------
def train_and_evaluate(X_train, X_eval, y_train, y_eval, label=""):
    models = {}
    accs = {}
    for attr in ATTRIBUTES:
        print(f"\nTraining LDA for {attr.capitalize()} {label}...")
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        lda.fit(X_train, y_train[attr])
        pred = lda.predict(X_eval)
        acc = accuracy_score(y_eval[attr], pred)
        accs[attr] = acc
        models[attr] = lda
        print(f"   {attr.capitalize()} accuracy {label}: {acc:.4f}")
    return models, accs

# ----------------------------------------------------------------------
# Visualization helpers
# ----------------------------------------------------------------------
def plot_attribute_visualizations(X_eval, y_eval, models, output_dir=OUTPUT_DIR) -> None:
    """Create shared t-SNE plots and per-attribute confusion matrices."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS FOR MOBILITY / MANIPULATION / PAYLOAD")
    print("=" * 70)

    plt.style.use("seaborn-v0_8-whitegrid")

    # One color per capability level.
    level_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    custom_cmap = mcolors.ListedColormap(level_colors)
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)
    level_labels = [f"Level {level}" for level in LEVEL_VALUES]

    attr_display_names = {
        "mobility": "Mobility",
        "manipulation": "Manipulation",
        "payload": "Payload",
    }

    # Compute t-SNE once and reuse it for all three colorings.
    print("Computing single projection (shared across all plots)...")
    plot_count = min(3000, X_eval.shape[0])
    tsne = TSNE(
        n_components=2,
        perplexity=40,
        init="pca",
        learning_rate="auto",
        random_state=42,
        n_jobs=-1,
    )
    X_tsne = tsne.fit_transform(X_eval[:plot_count])

    print("Generating t-SNE visualizations...")
    for attr in ATTRIBUTES:
        attr_name = attr_display_names[attr]
        y_plot = y_eval[attr][:plot_count]

        plt.figure(figsize=(12, 9.5))
        scatter = plt.scatter(
            X_tsne[:, 0],
            X_tsne[:, 1],
            c=y_plot,
            cmap=custom_cmap,
            norm=norm,
            s=250,
            edgecolor="k",
            linewidth=0.5,
            alpha=0.9,
        )

        plt.title(
            f"Analysis of Task Instructions\nColored by {attr_name} Capability Level (Hybrid embeddings)",
            fontsize=22,
            pad=25,
            weight="bold",
        )

        cbar = plt.colorbar(scatter, ticks=list(LEVEL_VALUES), boundaries=bounds)
        cbar.set_label(f"{attr_name} Level", fontsize=14, labelpad=15)
        cbar.set_ticklabels(level_labels)
        cbar.ax.tick_params(labelsize=11)

        plt.xlabel("Dimension 1", fontsize=14)
        plt.ylabel("Dimension 2", fontsize=14)
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        plt.savefig(f"{output_dir}/analysis_{attr}.png", dpi=400, bbox_inches="tight")
        plt.close()

    print("Generating confusion matrices...")
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for idx, attr in enumerate(ATTRIBUTES):
        attr_name = attr_display_names[attr]
        pred = models[attr].predict(X_eval)
        cm = confusion_matrix(y_eval[attr], pred, labels=list(LEVEL_VALUES))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=list(LEVEL_VALUES),
            yticklabels=list(LEVEL_VALUES),
            ax=axes[idx],
            annot_kws={"size": 16},
            linewidths=0.5,
        )
        axes[idx].set_title(attr_name, fontsize=16, pad=15)
        axes[idx].set_xlabel("Predicted Level", fontsize=13)
        axes[idx].set_ylabel("True Level", fontsize=13)

    plt.suptitle("Confusion Matrices (Hybrid embeddings)", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrices_attributes.png", dpi=400, bbox_inches="tight")
    plt.close()

    print(f"All visualizations saved in folder: {output_dir}/")
    print("   -> analysis_mobility.png")
    print("   -> analysis_manipulation.png")
    print("   -> analysis_payload.png")
    print("   -> confusion_matrices_attributes.png")


# ----------------------------------------------------------------------
# Main script entry point
# ----------------------------------------------------------------------
def main() -> None:
    X_train, X_eval, y_train, y_eval, encoder = build_dataset()

    # Raw embedding only.
    X_train_raw = X_train[:, :encoder.semantic_dim]
    X_eval_raw = X_eval[:, :encoder.semantic_dim]

    # Hybrid = embedding + handcrafted features.
    X_train_hybrid = X_train
    X_eval_hybrid = X_eval

    models_raw, accuracies_raw = train_and_evaluate(
        X_train_raw, X_eval_raw, y_train, y_eval, label="(raw embedding)"
    )

    models_hybrid, accuracies_hybrid = train_and_evaluate(
        X_train_hybrid, X_eval_hybrid, y_train, y_eval, label="(hybrid)"
    )

    print("\n" + "=" * 72)
    print("FINAL ACCURACIES COMPARISON")
    print("=" * 72)
    for attr in ATTRIBUTES:
        raw_acc = accuracies_raw[attr]
        hybrid_acc = accuracies_hybrid[attr]
        delta = hybrid_acc - raw_acc
        print(
            f"{attr.capitalize():12} -> Raw: {raw_acc:.4%} | "
            f"Hybrid: {hybrid_acc:.4%} | Delta: {delta:+.4%}"
        )
    print("=" * 72)

    # Use hybrid models for the demo prediction.
    models = models_hybrid

    sample = NLTaskGenerator().generate_task_random()
    vec = encoder.encode_instruction(sample["instruction"]).reshape(1, -1)
    print(f"\nTest instruction: {sample['instruction']}")
    true = sample["requirements"]
    print(f"True      -> Mobility:{true[0]}  Manipulation:{true[1]}  Payload:{true[2]}")
    print("Predicted ->", end=" ")
    for attr, lda in models.items():
        p = lda.predict(vec)[0]
        print(f"{attr.capitalize()}:{p}", end="  ")
    print()

    plot_attribute_visualizations(X_eval, y_eval, models, OUTPUT_DIR)


if __name__ == "__main__":
    main()
