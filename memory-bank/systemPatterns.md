# System Patterns: D6 Dice Recognition App

## Overall Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Camera        │    │   AI Processing   │    │   Display       │
│   Module        │───▶│   Pipeline       │───▶│   Interface     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌──────────────────┐             │
         │              │   Model Store    │             │
         │              │   (TFLite)       │             │
         │              └──────────────────┘             │
         │                                                │
         └───────────────────────────────────────────────┘
                           Feedback Loop
```

## Core Components

### 1. Camera Interface Layer

**Purpose**: Handle Raspberry Pi camera operations
**Pattern**: Adapter Pattern for camera abstraction

```python
class CameraInterface:
    def __init__(self):
        self.camera = Picamera2()

    def start_preview(self):
        pass

    def capture_frame(self):
        pass

    def stop(self):
        pass
```

### 2. Computer Vision Pipeline

**Purpose**: Process frames through the AI model
**Pattern**: Pipeline Pattern for sequential processing

```python
class VisionPipeline:
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.model = DiceDetectionModel()
        self.postprocessor = ResultsProcessor()

    def process_frame(self, frame):
        # Preprocess → Inference → Postprocess
        pass
```

### 3. Model Management

**Purpose**: Handle TensorFlow Lite model operations
**Pattern**: Singleton Pattern for model instance

```python
class DiceDetectionModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.load_model()
        return cls._instance
```

### 4. Results Display

**Purpose**: Show detection results to user
**Pattern**: Observer Pattern for real-time updates

```python
class DisplayManager:
    def __init__(self):
        self.observers = []

    def add_observer(self, observer):
        pass

    def notify_detection(self, results):
        pass
```

## Data Flow Patterns

### Real-time Processing Loop

```
Camera Capture → Frame Buffer → Process Queue → Model Inference
      ↑                                              │
      │                                              ▼
Display Update ← Result Queue ← Post-processing ← Raw Predictions
```

### Threading Architecture

- **Main Thread**: GUI and user interaction
- **Camera Thread**: Continuous frame capture
- **Processing Thread**: Model inference
- **Display Thread**: Result rendering

## Design Patterns

### 1. Factory Pattern - Model Loading

Creates appropriate model based on hardware capabilities:

```python
class ModelFactory:
    @staticmethod
    def create_model(model_type="optimized"):
        if model_type == "optimized":
            return OptimizedDiceModel()
        elif model_type == "accurate":
            return AccurateDiceModel()
```

### 2. State Pattern - Application States

Manages different application states:

```python
class AppState:
    states = ["INITIALIZING", "READY", "PROCESSING", "ERROR"]
    current_state = "INITIALIZING"
```

### 3. Command Pattern - Processing Operations

Encapsulates processing operations:

```python
class ProcessFrameCommand:
    def __init__(self, frame, pipeline):
        self.frame = frame
        self.pipeline = pipeline

    def execute(self):
        return self.pipeline.process_frame(self.frame)
```

## Component Relationships

### Dependency Injection

- Models injected into pipeline
- Camera interface injected into capture manager
- Display components receive result streams

### Event-Driven Communication

- Frame capture events trigger processing
- Detection events trigger display updates
- Error events trigger recovery procedures

## Performance Patterns

### Resource Management

- **Pool Pattern**: Reuse expensive objects (model instances)
- **Cache Pattern**: Cache preprocessed frames
- **Lazy Loading**: Load models only when needed

### Optimization Strategies

- **Frame Skipping**: Process every Nth frame under load
- **ROI Processing**: Focus on regions of interest
- **Batch Processing**: Group operations when possible

## Error Handling Patterns

### Circuit Breaker

Prevent cascade failures in processing pipeline:

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5):
        self.failure_count = 0
        self.threshold = failure_threshold
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
```

### Graceful Degradation

- Fall back to simpler processing on resource constraints
- Reduce frame rate under high load
- Switch to CPU-only mode if GPU fails
