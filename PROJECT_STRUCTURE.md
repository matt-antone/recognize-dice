# D6 Dice Recognition - Project Structure

## Folder Organization Standards

```
recognize-dice/
├── README.md                    # Project overview and quick start
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore patterns
│
├── docs/                       # Documentation
│   ├── guides/                 # How-to guides
│   │   ├── edge-impulse-workflow.md
│   │   ├── camera-setup.md
│   │   └── deployment-guide.md
│   ├── analysis/               # Technical analysis
│   │   ├── platform-comparison.md
│   │   ├── performance-benchmarks.md
│   │   └── ai-acceleration-analysis.md
│   └── research/               # Research and concepts
│       ├── pico2-feasibility.md
│       └── alternative-platforms.md
│
├── src/                        # Source code
│   ├── dice_detector/          # Main application package
│   │   ├── __init__.py
│   │   ├── camera_interface.py
│   │   ├── ai_processor.py
│   │   ├── dice_classifier.py
│   │   └── gui_app.py
│   └── utils/                  # Utility modules
│       ├── __init__.py
│       ├── config.py
│       └── logging_helper.py
│
├── scripts/                    # Utility scripts
│   ├── data_collection/
│   │   ├── collect_dice_images.py
│   │   └── download_dataset.py
│   ├── setup/
│   │   ├── install_dependencies.py
│   │   └── configure_camera.py
│   └── deployment/
│       ├── deploy_to_pi.py
│       └── performance_test.py
│
├── data/                       # Datasets and training data
│   ├── raw/                    # Original datasets
│   │   ├── roboflow_dice/
│   │   └── custom_captures/
│   ├── processed/              # Processed for training
│   │   ├── edge_impulse_format/
│   │   └── validation_set/
│   └── samples/                # Test images
│
├── models/                     # Trained models
│   ├── edge_impulse/           # Models from Edge Impulse
│   │   ├── dice_classifier_v1.tflite
│   │   ├── dice_classifier_v2.tflite
│   │   └── labels.txt
│   ├── benchmarks/             # Performance benchmarks
│   └── experimental/           # Research models
│
├── tests/                      # Test code
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   ├── performance/            # Performance tests
│   └── hardware/               # Hardware-specific tests
│
├── configs/                    # Configuration files
│   ├── pi3_config.yaml
│   ├── zero2w_config.yaml
│   └── deployment_configs/
│
├── memory-bank/                # Project memory system
│   ├── projectbrief.md
│   ├── activeContext.md
│   ├── systemPatterns.md
│   ├── techContext.md
│   ├── productContext.md
│   └── progress.md
│
└── deployment/                 # Deployment artifacts
    ├── pi3/                    # Pi 3 deployment
    ├── zero2w/                 # Pi Zero 2 W deployment
    └── docker/                 # Container deployments
```

## File Naming Conventions

### Python Files

- **snake_case** for all Python files
- **Descriptive names**: `dice_classifier.py` not `dc.py`
- **Clear purpose**: `camera_interface.py` not `interface.py`

### Documentation

- **kebab-case** for markdown files: `edge-impulse-workflow.md`
- **Version suffixes** where relevant: `deployment-guide-v2.md`
- **Platform prefixes** where applicable: `pi3-setup-guide.md`

### Data Files

- **Descriptive directories**: `roboflow_dice/` not `dataset1/`
- **Version control**: `dice_classifier_v1.tflite`
- **Environment tags**: `pi3_optimized_model.tflite`

## Organization Principles

### 1. Separation of Concerns

- **Source code** separated from **documentation**
- **Scripts** separated from **main application**
- **Data** separated from **models**

### 2. Platform Organization

- Platform-specific files in dedicated subdirectories
- Shared code in common locations
- Clear deployment separation

### 3. Development Workflow

- **Development**: `src/` + `tests/`
- **Data preparation**: `scripts/data_collection/` + `data/`
- **Model training**: External (Edge Impulse) → `models/`
- **Deployment**: `scripts/deployment/` + `deployment/`

### 4. Documentation Hierarchy

- **High-level**: Root README.md
- **Guides**: Step-by-step instructions in `docs/guides/`
- **Analysis**: Technical deep-dives in `docs/analysis/`
- **Research**: Experimental concepts in `docs/research/`

## Clean-up Actions Needed

### Files to Reorganize:

```bash
# Current → Target Location
collect_dice_data.py → scripts/data_collection/collect_dice_images.py
download_dice_dataset.py → scripts/data_collection/download_dataset.py
edge_impulse_guide.md → docs/guides/edge-impulse-workflow.md
pico2_dice_concept.md → docs/research/pico2-feasibility.md
pi_zero_2w_analysis.md → docs/analysis/platform-comparison.md
```

### Files to Create:

- `README.md` (project overview)
- `requirements.txt` (dependencies)
- `.gitignore` (ignore patterns)
- `src/dice_detector/__init__.py` (package structure)

### Directories to Create:

```bash
mkdir -p docs/{guides,analysis,research}
mkdir -p src/{dice_detector,utils}
mkdir -p scripts/{data_collection,setup,deployment}
mkdir -p data/{raw,processed,samples}
mkdir -p models/{edge_impulse,benchmarks,experimental}
mkdir -p tests/{unit,integration,performance,hardware}
mkdir -p configs
mkdir -p deployment/{pi3,zero2w,docker}
```

## Benefits of This Structure

### 1. **Scalability**

- Easy to add new platforms (Pi 5, Jetson, etc.)
- Clear separation for different model versions
- Room for growth without restructuring

### 2. **Collaboration**

- Clear where to find/add different types of files
- Consistent naming makes navigation intuitive
- Separation reduces merge conflicts

### 3. **Deployment**

- Platform-specific deployments clearly separated
- Easy to package for different targets
- Clear artifact organization

### 4. **Maintenance**

- Documentation co-located with relevant code
- Test organization matches source structure
- Configuration management simplified

This structure follows Python packaging best practices and scales well from development through production deployment.
