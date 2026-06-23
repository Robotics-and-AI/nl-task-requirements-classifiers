# Natural-Language Task Requirements Classifiers

This repository contains two Python scripts for generating synthetic natural-language task instructions and training linear classifiers for task requirement analysis.

The project uses a shared YAML file of scenario templates:

```text
instruction templates + scenario requirements
-> synthetic dataset
-> sentence embeddings + handcrafted features
-> linear classifiers
```

In this project, **LDA** means **Linear Discriminant Analysis**.

## Repository Structure

```text
nl-task-requirements-classifiers/
|-- README.md
|-- requirements.txt
|-- templates/
|   -- scenarios.yaml
|-- src/
|   -- hybrid_scenarios_linear_classifier.py
|   -- hybrid_levels_linear_classifier.py
|-- outputs/
```

## Approach

Each scenario in `templates/scenarios.yaml` defines:

- capability requirements: `mobility_level`, `manipulation_level`, and `payload_level`
- numeric ranges for generated values, such as payload weight and reach distance
- options such as places, size adjectives, and unit counts
- train templates and eval templates

The scripts generate synthetic task instructions by sampling from these templates.

Each generated instruction is encoded using:

- a semantic sentence embedding from a SentenceTransformer model
- handcrafted capability features extracted from the text

The handcrafted features capture explicit cues such as kilograms, centimeters, movement keywords, manipulation keywords, and mobility/environment keywords.

## Scripts

### `src/hybrid_scenarios_linear_classifier.py`

Trains one LDA classifier to predict the full task scenario.

```text
instruction -> scenario label
```

Example output labels include:

```text
scenario1
scenario14
scenario46
```

Use this script when you want to classify the complete task type.

### `src/hybrid_levels_linear_classifier.py`

Trains three independent LDA classifiers to predict capability requirement levels:

```text
instruction -> mobility_level + manipulation_level + payload_level
```

Use this script when you want to estimate the capabilities required by a natural-language task.

## Templates

The shared template file is:

```text
templates/scenarios.yaml
```

Each scenario follows this general structure:

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

Every scenario is expected to have exactly **20 train templates** and **5 eval templates**. The scripts validate this when loading the YAML file.

## Installation

Create a virtual environment if desired, then install the dependencies:

```bash
pip install -r requirements.txt
```

The first run may download the selected SentenceTransformer model.

## Running

From the repository root, run the scenario classifier:

```bash
python src/hybrid_scenarios_linear_classifier.py
```

Or run the capability-level classifier:

```bash
python src/hybrid_levels_linear_classifier.py
```

## Outputs

Generated plots are saved under:

```text
outputs/scenarios/
outputs/levels/
```

Typical outputs include:

- analysis plots
- confusion matrices
- PCA variance plots
- printed train/eval accuracy
- one random test prediction

## Notes

The dataset is synthetic. This makes experiments reproducible and easy to control, but real-world deployment should include testing with natural task instructions written by users or operators.

The templates and requirements are intentionally separated from the training code so that the dataset can be reviewed, edited, and version-controlled independently.
