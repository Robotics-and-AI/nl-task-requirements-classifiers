# Robot Task Capability Classifiers

This repository contains two Python scripts for generating synthetic natural-language robot task instructions and training linear classifiers for robot task analysis.

The project uses a shared YAML file of scenario templates:

```text
instruction templates + scenario requirements -> synthetic dataset -> sentence embeddings + handcrafted features -> LDA classifiers
```

LDA means **Linear Discriminant Analysis**, not Latent Dirichlet Allocation.

## Repository structure

```text
robot-task-classifiers/
├── README.md
├── requirements.txt
├── templates/
│   └── scenarios.yaml
├── src/
│   ├── hybrid_scenarios_linear_classifier.py
│   └── hybrid_levels_linear_classifier.py
└── outputs/
```

## Approach

Each scenario in `templates/scenarios.yaml` defines:

- capability requirements: `mobility_level`, `manipulation_level`, `payload_level`
- numeric ranges for generated values, such as payload weight and reach distance
- options such as places, size adjectives, and unit counts
- train templates and eval templates

The scripts generate many task instructions by sampling from these templates. Each instruction is encoded using:

1. a semantic sentence embedding from a SentenceTransformer model
2. handcrafted capability features extracted from the text

The handcrafted features include cues such as kilograms, centimeters, movement keywords, manipulation keywords, and mobility/environment keywords.

## Scripts

### `src/hybrid_scenarios_linear_classifier.py`

Trains one LDA classifier to predict the full task scenario.

```text
instruction -> scenario label
```

Example outputs are labels such as `scenario1`, `scenario14`, or `scenario46`.

Use this script when you want to classify the complete task type.

### `src/hybrid_levels_linear_classifier.py`

Trains three independent LDA classifiers to predict robot capability levels:

```text
instruction -> mobility_level + manipulation_level + payload_level
```

Use this script when you want to estimate the robot capabilities required by a natural-language task.

## Templates

The shared template file is:

```text
templates/scenarios.yaml
```

Each scenario has this general structure:

```yaml
scenario1:
  requirements:
    mobility_level: 1
    manipulation_level: 2
    payload_level: 1

  ranges:
    weight_kg: [0.01, 3.0]
    reach_cm: [1, 50]

  options:
    units: [1, 2, 3]
    size_adjectives:
      - small
      - tiny
    places:
      - warehouse
      - assembly line

  templates:
    train:
      - "..."
    eval:
      - "..."
```

Every scenario is expected to have exactly 5 eval templates. The scripts validate this when loading the YAML file.

## Installation

Create a virtual environment if desired, then install dependencies:

```bash
pip install -r requirements.txt
```

The first run may download the selected SentenceTransformer model.

## Running

From the repository root, run:

```bash
python src/hybrid_scenarios_linear_classifier.py
```

or:

```bash
python src/hybrid_levels_linear_classifier.py
```

## Outputs

Generated plots are saved under `outputs/`:

```text
outputs/scenarios/
outputs/levels/
```

Typical outputs include:

- t-SNE analysis plots
- confusion matrices
- PCA variance plots
- printed train/eval accuracy
- one random test prediction

## Notes

The dataset is synthetic. This makes experiments reproducible and easy to control, but real-world deployment should include testing with natural task instructions written by users or robot operators.

The templates and requirements are intentionally separated from the training code so that the dataset can be reviewed, edited, and version-controlled independently.
