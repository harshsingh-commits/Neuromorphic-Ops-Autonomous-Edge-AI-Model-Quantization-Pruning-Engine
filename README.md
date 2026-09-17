# ⚡ Neuromorphic-Ops

## Autonomous Edge-AI Model Quantization & Pruning Engine

Neuromorphic-Ops is an autonomous AI-powered model optimization system that takes a PyTorch model through analysis, pruning/quantization, evaluation, refinement, ONNX export, human approval, and edge-oriented runtime deployment.

The project combines **PyTorch, LangGraph, FastAPI, Streamlit, ONNX Runtime, Docker, monitoring, experiment tracking, and persistent workflow state** into one model-optimization lifecycle.

---

# 🚀 What Does the Project Do?

A normal ML workflow often ends after training a model.

Neuromorphic-Ops focuses on what happens **after training**:

```text
Trained PyTorch Model
        ↓
Analyze
        ↓
Choose Optimization Strategy
        ↓
Prune / Quantize
        ↓
Evaluate
        ↓
Refine
        ↓
Export to ONNX
        ↓
Validate
        ↓
Human Approval
        ↓
ONNX Runtime Deployment
        ↓
Monitor
```

The goal is to make model optimization **automated, measurable, repeatable, and deployment-oriented**.

---

# 🔍 How the Project Works — Detailed Step-by-Step

## Step 1 — Start the Application

The project is containerized using Docker Compose.

The application starts two main services:

```text
                 Docker Compose
                      │
            ┌─────────┴─────────┐
            │                   │
            ▼                   ▼
      FastAPI Backend      Streamlit Frontend
        Port 8000             Port 8501
```

The backend exposes the REST API and executes the optimization workflow.

The frontend provides the interactive dashboard through which the user uploads models, starts optimization, reviews metrics, approves deployment, and downloads artifacts.

---

## Step 2 — Upload the PyTorch Model

The user selects a `.pt` or `.pth` model from the Streamlit dashboard.

Example demo model:

```text
demo/trained_demo_model.pt
```

The frontend sends the file to:

```text
POST /upload
```

The FastAPI backend performs upload validation before saving the model.

### Upload security checks

The backend checks:

```text
Filename exists
       ↓
Safe filename
       ↓
.pt / .pth extension
       ↓
Path traversal protection
       ↓
100 MB size limit
       ↓
Non-empty file
       ↓
Safe output path
       ↓
Store model
```

The uploaded model is stored inside the configured application output directory.

Example:

```text
User uploads:
demo/trained_demo_model.pt

Backend stores:
outputs/trained_demo_model.pt
```

The API returns the model path to the frontend.

---

## Step 3 — Model Loading

When optimization begins, the backend loads the uploaded PyTorch model.

The model loader supports a serialized `torch.nn.Module` and supported checkpoint wrappers.

The model is loaded on CPU and placed into evaluation mode:

```text
Model Artifact
      ↓
torch.load(..., map_location="cpu")
      ↓
Validate loaded object
      ↓
torch.nn.Module
      ↓
model.eval()
```

For the included trusted demo model, full-object loading is used with:

```python
weights_only=False
```

> Full PyTorch object deserialization should only be used with trusted model files because unpickling can execute code.

---

## Step 4 — Validate the Model Path

Before the optimization workflow is started, the backend validates that the model path:

```text
Exists
Is a regular file
Uses .pt or .pth
Is inside the configured output directory
Is not empty
```

This prevents arbitrary filesystem paths from being passed into the optimization pipeline.

---

## Step 5 — Start the LangGraph Workflow

The frontend sends an optimization request to:

```text
POST /optimize
```

The request contains information such as:

```json
{
  "model_path": "/app/outputs/trained_demo_model.pt",
  "strategy": null,
  "accuracy_threshold": 1.5
}
```

The backend then starts the LangGraph-based optimization workflow.

Conceptually:

```text
FastAPI
   ↓
Orchestrator
   ↓
LangGraph Workflow
   ↓
Agent / Node Execution
```

A unique workflow thread/run ID is created so that the run can be tracked.

---

## Step 6 — Model Analysis

The analysis stage examines the input model and gathers information required by subsequent optimization stages.

The workflow can use information such as:

```text
Model architecture
Parameter information
Layer information
Model size
Optimization-relevant characteristics
```

This information is stored in the workflow state and becomes available to later stages.

Conceptually:

```text
PyTorch Model
     ↓
Analyzer
     ↓
Model Information
     ↓
Workflow State
```

---

## Step 7 — Optimization Strategy Selection

Neuromorphic-Ops supports different optimization strategies:

```text
Automatic
Pruning
Quantization
Hybrid
```

When the user selects **Automatic**, the workflow can determine an appropriate optimization path based on the model and configured constraints.

The selected strategy is recorded in the workflow state.

Example:

```text
Strategy
   ↓
Pruning
```

or:

```text
Strategy
   ↓
Quantization
```

or:

```text
Strategy
   ↓
Hybrid
   ├── Pruning
   └── Quantization
```

---

## Step 8 — Model Pruning

When pruning is selected, the optimization stage applies pruning to reduce model parameters.

Conceptually:

```text
Dense Model
     ↓
Pruning Operation
     ↓
Selected Parameters Reduced / Zeroed
     ↓
Sparse Model
```

The configured pruning ratio controls the intended amount of pruning.

Example:

```text
Pruning Ratio = 0.20

Target:
approximately 20% parameter sparsity
```

The actual result depends on the model architecture and pruning method.

The workflow records pruning-related information so that the optimization can later be evaluated.

---

## Step 9 — Model Quantization

When quantization is selected, the model is transformed toward lower-precision representations supported by the optimization pipeline.

Conceptually:

```text
FP32 Model
    ↓
Quantization
    ↓
Lower-precision representation
    ↓
Optimized Model
```

The project records the chosen quantization configuration and resulting model artifact.

Quantization is useful when deployment constraints favor smaller model representations and reduced computational or memory requirements.

---

## Step 10 — Accuracy Evaluation

After optimization, the system evaluates the original and optimized models.

For supported evaluation datasets, the evaluation stage performs:

```text
Original Model
       ↓
Evaluation Dataset
       ↓
Original Accuracy
```

and:

```text
Optimized Model
       ↓
Evaluation Dataset
       ↓
Optimized Accuracy
```

The system calculates:

```text
Accuracy Drop =
Original Accuracy - Optimized Accuracy
```

The configured accuracy threshold acts as a constraint for the optimization workflow.

Example:

```text
Maximum Accuracy Drop = 1.5 pp
```

means the optimization can be evaluated against a maximum allowed accuracy difference of 1.5 percentage points.

---

## Step 11 — Latency Evaluation

The optimized model is benchmarked for inference latency.

The latency measurement process follows:

```text
Load Model
    ↓
Warm-up Runs
    ↓
Timed Inference Runs
    ↓
Latency Statistics
```

The benchmark records values such as:

```text
Median latency
Minimum latency
Maximum latency
```

The project also supports multiple batch sizes:

```text
Batch 1
Batch 8
Batch 16
```

This makes it possible to compare runtime behavior under different inference workloads.

---

## Step 12 — Memory Evaluation

The memory stage measures the optimized model's memory footprint.

Conceptually:

```text
Original Model
       ↓
Baseline Memory

Optimized Model
       ↓
Optimized Memory
```

Memory reduction is calculated when a valid baseline is available:

```text
Memory Reduction % =
((Baseline Memory - Optimized Memory)
 / Baseline Memory) × 100
```

This helps evaluate whether an optimization is useful for memory-constrained edge environments.

---

## Step 13 — Compression and Sparsity Analysis

The system also calculates optimization-related properties such as:

```text
Original model size
Optimized model size
Compression ratio
Pruning ratio
Sparsity
```

Compression ratio is represented as:

```text
Compression Ratio =
Original Size / Optimized Size
```

For example:

```text
Original = 0.14 MB
Optimized = 0.05 MB

Compression ≈ 3.07×
```

---

## Step 14 — Optimization Refinement

Neuromorphic-Ops is not designed to blindly assume that an optimization is successful.

The workflow can use evaluation results to determine whether refinement is required.

Conceptually:

```text
Optimization
      ↓
Evaluation
      ↓
Check Constraints
      │
      ├── Within constraints
      │       ↓
      │    Continue
      │
      └── Outside constraints
              ↓
          Refinement
              ↓
       Re-evaluate
```

The refinement history is stored in the workflow state.

This creates an optimization loop instead of a single one-shot transformation.

---

## Step 15 — Experiment Tracking

Each optimization run can record experiment metadata.

Tracked information includes:

```text
Run ID
Timestamp
Random seed
Model hash
Optimization configuration
Evaluation metrics
Workflow information
```

This makes optimization runs traceable and allows experiments to be compared.

Conceptually:

```text
Optimization Run
      ↓
Experiment Metadata
      ↓
Tracking Storage
      ↓
Future Comparison / Audit
```

---

## Step 16 — ONNX Export

After the optimized model reaches the required export stage, the system converts it to ONNX.

```text
Optimized PyTorch Model
          ↓
       ONNX Export
          ↓
optimized_model.onnx
```

ONNX provides a portable inference representation that can be used by ONNX Runtime and other compatible deployment environments.

---

## Step 17 — ONNX Artifact Validation

The generated ONNX artifact is validated before it is used for runtime inference.

The validation pipeline checks the artifact and its model structure.

The project also validates runtime input expectations.

Conceptually:

```text
optimized_model.onnx
        ↓
Artifact Validation
        ↓
Runtime Validation
        ↓
Ready for Inference
```

---

## Step 18 — ONNX Input Validation

Before inference, the input tensor is validated.

The system checks properties such as:

```text
Input rank
Input dimensions
Expected tensor shape
Tensor dtype
NaN values
Infinity values
Maximum input size
```

For the included DemoCNN example, the normal input shape is:

```text
[1, 3, 32, 32]
```

This protects the runtime from invalid or excessively large requests.

---

## Step 19 — ONNX Runtime Session Creation

The backend creates an ONNX Runtime inference session for the optimized artifact.

The runtime uses the CPU execution provider in the current deployment.

Conceptually:

```text
optimized_model.onnx
        ↓
ONNX Runtime
        ↓
CPUExecutionProvider
        ↓
Inference Session
```

---

## Step 20 — ONNX Session Caching

The project includes ONNX Runtime session caching to reduce unnecessary repeated session creation.

Instead of constructing a new session for every operation, the backend can reuse a valid existing session.

Conceptually:

```text
ONNX Path
   ↓
Calculate File Signature
   ↓
Session Cache Lookup
      │
      ├── Match found
      │      ↓
      │   Reuse Session
      │
      └── No match
             ↓
       Create New Session
             ↓
        Store in Cache
```

The cache uses the model file signature to detect whether the cached session still corresponds to the current artifact.

Important distinction:

```text
LangGraph Checkpoint
    = persistent workflow state

ONNX Session Cache
    = runtime session optimization
```

The session cache is process-local and is not intended to survive container restarts.

---

## Step 21 — ONNX Runtime Inference

The deployed ONNX model can be executed through:

```text
POST /inference/onnx
```

The backend generates a test tensor when the API receives an input shape.

Example:

```text
[1, 3, 32, 32]
```

The inference flow is:

```text
Request
   ↓
Validate Input Shape
   ↓
Create Input Tensor
   ↓
Validate Tensor
   ↓
ONNX Runtime Session
   ↓
Inference
   ↓
Output Tensor
   ↓
Latency / Runtime Information
```

The API returns runtime information such as:

```text
status
latency
output count
output shapes
providers
error
```

---

## Step 22 — ONNX Runtime Benchmark

The project exposes:

```text
POST /inference/onnx/benchmark
```

The benchmark runs multiple inference operations.

Example configuration:

```text
Warm-up Runs = 10
Timed Runs   = 50
```

The benchmark produces measurements such as:

```text
Median latency
Minimum latency
Maximum latency
Execution providers
Input shape
```

This allows the exported model to be evaluated independently of the original PyTorch runtime.

---

## Step 23 — Human-in-the-Loop Approval

Neuromorphic-Ops includes a human approval gate before final deployment.

The workflow can pause in an approval state.

The dashboard displays:

```text
Accuracy
Accuracy Drop
Latency
Memory
Compression
Strategy
Pruning Ratio
Quantization
```

The engineer then chooses:

```text
✓ Approve Deployment
```

or:

```text
✕ Reject Deployment
```

This prevents the system from automatically treating every optimization as deployment-ready without human review.

---

## Step 24 — Deployment

When the optimization is approved, the workflow moves toward deployment readiness.

The optimized ONNX artifact becomes available for runtime inference and download.

Deployment-oriented endpoints include:

```text
POST /inference/onnx
POST /inference/onnx/benchmark
GET  /download/onnx
```

---

## Step 25 — Monitoring and Observability

The FastAPI backend monitors application activity.

The monitoring system records information such as:

```text
HTTP method
Request path
HTTP status
Request latency
Runtime errors
Optimization started
Optimization completed
Optimization failures
```

The dashboard and API can inspect live monitoring information.

Available monitoring endpoints:

```text
GET /health
GET /health/live
GET /health/ready
GET /monitoring
```

---

## Step 26 — Health, Liveness and Readiness

The application exposes three levels of service verification.

### Health

```text
GET /health
```

Confirms that the API service is responding.

### Liveness

```text
GET /health/live
```

Confirms that the API process is alive.

### Readiness

```text
GET /health/ready
```

Checks that required runtime resources, including the expected optimized ONNX artifact, are available.

---

## Step 27 — Restart-Safe Workflow Persistence

The LangGraph workflow uses checkpoint persistence.

This allows workflow state to be stored in SQLite-backed checkpoint storage.

Conceptually:

```text
Running Workflow
      ↓
Checkpoint
      ↓
SQLite
      ↓
Backend Restart
      ↓
Restore Workflow State
```

This is especially useful for workflows that pause for human approval or require state continuity across backend restarts.

---

## Step 28 — Report Generation

After evaluation and workflow execution, the system produces optimization information that can be exposed through the generated report.

The report can contain information such as:

```text
Model details
Optimization strategy
Accuracy
Latency
Memory
Compression
Deployment information
Workflow state
```

The report is available through:

```text
GET /report
```

---

# 🧩 Complete Project Flow

```text
                         USER
                           │
                           ▼
                  Streamlit Dashboard
                           │
                           ▼
                    POST /upload
                           │
                           ▼
                Upload Security Checks
                           │
                           ▼
                 PyTorch Model Artifact
                           │
                           ▼
                    Model Validation
                           │
                           ▼
                   LangGraph Workflow
                           │
                           ▼
                    Model Analysis
                           │
                           ▼
                 Strategy Selection
                           │
                 ┌─────────┼─────────┐
                 ▼         ▼         ▼
              Pruning  Quantization  Hybrid
                 └─────────┼─────────┘
                           ▼
                       Evaluation
                  ┌────────┼────────┐
                  ▼        ▼        ▼
               Accuracy  Latency  Memory
                  └────────┼────────┘
                           ▼
                  Compression Analysis
                           │
                           ▼
                    Refinement Loop
                           │
                           ▼
                   Experiment Tracking
                           │
                           ▼
                      ONNX Export
                           │
                           ▼
                    ONNX Validation
                           │
                           ▼
                  ONNX Runtime Session
                           │
                           ▼
                   Session Cache
                           │
                           ▼
                    Runtime Testing
                           │
                           ▼
                  Human Approval Gate
                       │         │
                    Reject     Approve
                       │         │
                       ▼         ▼
                     Stop    Deployment Ready
                                 │
                                 ▼
                          ONNX Inference
                                 │
                                 ▼
                       Monitoring + Report
```

---

# 🛠️ Technology Stack

## AI / ML

- Python
- PyTorch
- NumPy
- Model Pruning
- Model Quantization

## Agentic Workflow

- LangGraph

## Backend

- FastAPI
- Pydantic
- Uvicorn

## Deployment

- ONNX
- ONNX Runtime
- Docker
- Docker Compose

## Frontend

- Streamlit
- Plotly
- Custom CSS

## Persistence & Monitoring

- SQLite
- LangGraph Checkpointing
- Runtime Monitoring
- Health / Readiness Checks
- Experiment Tracking

---

# ▶️ How to Run the Project

## Step 1 — Clone the Repository

```bash
git clone https://github.com/harshsingh-commits/Neuromorphic-Ops.git
cd Neuromorphic-Ops
```

## Step 2 — Start Docker Desktop

Make sure Docker Desktop is running.

## Step 3 — Build and Start the Application

```bash
docker compose up -d --build
```

## Step 4 — Check Containers

```bash
docker compose ps
```

Expected:

```text
neuromorphic-ops-backend
neuromorphic-ops-frontend
```

Both should be running.

## Step 5 — Check Backend Health

Windows PowerShell:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Expected:

```text
status version
------ -------
ok     1.0.0
```

## Step 6 — Open the Dashboard

```text
http://localhost:8501
```

## Step 7 — Upload the Demo Model

Use:

```text
demo/trained_demo_model.pt
```

## Step 8 — Configure Optimization

Choose:

```text
Automatic
Pruning
Quantization
Hybrid
```

Set the maximum allowed accuracy drop and click:

```text
Start Autonomous Optimization
```

## Step 9 — Review the Workflow

The dashboard shows:

```text
Upload
Analyze
Optimize
Evaluate
Approval
Deploy
```

## Step 10 — Approve or Reject

When the workflow reaches the approval stage:

```text
Approve Deployment
```

or:

```text
Reject Deployment
```

## Step 11 — Download Artifacts

After successful execution:

```text
optimized_model.onnx
optimization_report.json
```

---

# 🌐 Local Services

| Service | URL |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI Backend | http://localhost:8000 |
| Health | http://localhost:8000/health |
| Liveness | http://localhost:8000/health/live |
| Readiness | http://localhost:8000/health/ready |
| Monitoring | http://localhost:8000/monitoring |

---

# 🧪 Testing

Run all tests:

```powershell
python -m pytest -q
```

Run security tests:

```powershell
python -m pytest -q tests/test_security.py
```

Check backend syntax:

```powershell
python -m py_compile .\backend\main.py
python -m py_compile .\backend\services\onnx_service.py
```

---

# 🔧 Useful Docker Commands

### Backend logs

```powershell
docker compose logs -f backend
```

### Frontend logs

```powershell
docker compose logs -f frontend
```

### Restart

```powershell
docker compose restart
```

### Stop

```powershell
docker compose down
```

### Start without rebuild

```powershell
docker compose up -d
```

### Rebuild after code changes

```powershell
docker compose up -d --build
```

---

# 📊 Demo Results

Example optimization run:

| Metric | Result |
|---|---:|
| Original Model Size | 0.14 MB |
| Optimized Model Size | 0.05 MB |
| Compression Ratio | 3.07× |
| Baseline Accuracy | 10.20% |
| Optimized Accuracy | 10.20% |
| Accuracy Drop | 0.00 pp |

The system also measures latency, memory, sparsity, and runtime behavior.

> The included demo model is intended to demonstrate the optimization pipeline and deployment workflow. Its accuracy should not be interpreted as a production benchmark.

---

# 🔐 Security

Neuromorphic-Ops implements several security and validation layers:

- `.pt` / `.pth` extension validation
- Upload size limits
- Path traversal protection
- Safe model paths
- Temporary upload handling
- ONNX artifact validation
- Input shape validation
- Tensor dtype validation
- NaN / infinity validation
- Oversized input protection
- Docker non-root execution
- Health/readiness checks
- Runtime error monitoring

> Only trusted PyTorch model files should be loaded with full-object deserialization.

---

# 📡 API Endpoints

```text
GET  /health
GET  /health/live
GET  /health/ready
GET  /monitoring

POST /upload
POST /optimize
POST /approve/{thread_id}

GET  /status/{thread_id}
GET  /report

POST /inference/onnx
POST /inference/onnx/benchmark
GET  /download/onnx
```

---

# 📁 Project Structure

```text
Neuromorphic-Ops/
│
├── backend/
│   ├── agents/
│   │   ├── analyzer.py
│   │   ├── strategy.py
│   │   ├── pruner.py
│   │   ├── quantizer.py
│   │   ├── evaluator.py
│   │   ├── accuracy_agent.py
│   │   ├── latency_agent.py
│   │   ├── memory_agent.py
│   │   ├── refiner.py
│   │   ├── exporter.py
│   │   ├── reporter.py
│   │   └── approval.py
│   │
│   ├── graph/
│   ├── models/
│   ├── services/
│   ├── main.py
│   └── state.py
│
├── frontend/
│   ├── streamlit_app.py
│   └── app.py
│
├── demo/
│   ├── demo_model.py
│   ├── create_demo_model.py
│   ├── train_demo_model.py
│   └── trained_demo_model.pt
│
├── tests/
│
├── models/
├── outputs/
├── reports/
├── runs/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

---

# 🎯 Why This Project?

Neuromorphic-Ops is designed to demonstrate the complete **post-training model optimization lifecycle**.

Instead of building only:

```text
Train Model
```

the project focuses on:

```text
Train
  ↓
Analyze
  ↓
Optimize
  ↓
Evaluate
  ↓
Refine
  ↓
Export
  ↓
Validate
  ↓
Approve
  ↓
Deploy
  ↓
Monitor
```

This brings together AI/ML engineering, backend engineering, model optimization, deployment, observability, and MLOps concepts in a single system.

---
Bhai, tumhare **Neuromorphic-Ops** project ko run karne ka exact process ye hai:

### 1. Project folder open karo

cd "C:\Users\harsh\Desktop\Neuromorphic-Ops Autonomous Edge-AI Model Quantization & Pruning Engine"


### 2. Virtual environment activate karo


.\.venv\Scripts\Activate.ps1


Prompt mein `(.venv)` aa jana chahiye.

### 3. Docker Desktop start karo

Docker Desktop **running** hona chahiye.

### 4. Project build + start karo

```powershell
docker compose up -d --build
```

Ye backend aur frontend dono start karega.

### 5. Containers check karo

powershell
docker compose ps


Expected:

```text
neuromorphic-ops-backend
neuromorphic-ops-frontend
```

Dono `Running` hone chahiye.

### 6. Backend check karo

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Expected:

```text
status version
------ -------
ok     1.0.0
```

### 7. Frontend open karo

Browser mein:

```text
http://localhost:8501
```

### 8. Demo model upload karo

Dashboard mein:

```text
demo\trained_demo_model.pt
```

select karo.

Phir:

```text
Upload Model
```

### 9. Optimization start karo

Strategy choose karo:

```text
Automatic
Pruning
Quantization
Hybrid
```

Accuracy threshold set karo, phir:

```text
Start Autonomous Optimization
```

### 10. Workflow complete karo

Flow:

```text
Upload
   ↓
Analyze
   ↓
Optimize
   ↓
Evaluate
   ↓
ONNX Export
   ↓
Human Approval
   ↓
Deployment
```

### 11. Output dekho

Successful run ke baad:

```text
optimized_model.onnx
optimization_report.json
```

download kar sakte ho.

---

### Project band karna ho

```powershell
docker compose down
```

### Dobara start karna ho

```powershell
docker compose up -d
```

### Code change ke baad

```powershell
docker compose up -d --build
```

### Logs dekhne ho

```powershell
docker compose logs -f backend
```

**Sabse important command:**

```powershell
docker compose up -d --build
```

Phir `http://localhost:8501` kholo.

# 👨‍💻 Author

**Harsh Singh**

Computer Engineering

AI / ML · Backend Engineering · AI Systems

---

# ⭐ Keywords

```text
Artificial Intelligence
Machine Learning
Deep Learning
PyTorch
LangGraph
Model Optimization
Model Pruning
Quantization
ONNX
ONNX Runtime
Edge AI
FastAPI
Streamlit
Docker
MLOps
AI Engineering
Backend Engineering
Model Deployment
```
