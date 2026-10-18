# DC-Ops: On-Device Data Center Operations Assistant

**Qualcomm x Meta ExecuTorch Hackathon — June 27–28, 2026**

Real-time camera-based tool for data center technicians. Point your phone at server racks to detect components — compute trays, network ports, LEDs, cables, fans, drive bays, and more — with on-device AI. Runs **live on the Samsung Galaxy S25 Ultra**, fully offline, with detection selectable between the CPU and the Snapdragon **Hexagon NPU**.

The [Alif E8 port](alif-e8/README.md) also includes a [live English→Spanish sign demo](alif-e8/translation/live/README.md): camera OCR on two Ethos-U55 NPUs, transparent Spanish overlays for 60 common signs, and an experimental ExecuTorch/U85 neural translator.

## The Problem

Over **1,000 server racks are built every week**, assembled at volume by labor crews cross-referencing paper guides. Roughly **80% are installed incorrectly the first time** — cables in the wrong ports, components misidentified. DC-Ops turns any Snapdragon phone into a real-time inspection and assembly-guidance tool.

## Why On-Device?

- **Air-gapped environments**: Many data centers prohibit cloud connectivity in server rooms
- **Privacy**: Serial numbers, rack topology, and infrastructure status are sensitive/classified
- **Latency**: Technicians need instant visual feedback walking through aisles
- **Offline-first**: Even connected DCs have dead zones between rack rows
- **Power efficiency**: Continuous live detection on the NPU sips battery vs. the CPU

## Architecture

```
Camera (CameraX)
 │
 ▼
YOLOv8n-seg  (fine-tuned for 16 DC component classes)
 ├─ ExecuTorch → QNN HTP → Snapdragon NPU   (INT8, 3.2 MB)   ← runs on the Hexagon NPU
 └─ ExecuTorch → XNNPACK → CPU              (FP32, 13 MB)    ← CPU path / fallback
 │
 ├─ NPU model emits raw conv feature maps; YOLO anchor-decode (DFL + NMS)
 │  runs on-device in ModelManager.parseRawYoloOutput()
 │
 ▼
Two modes (live on the S25)
 ├─ Server Scan  — multi-class detection with color-coded overlays
 └─ Cable Match  — "Mission Control" AR workflow that highlights the exact
                   port for the next cable, guiding correct rack assembly
 │
 ▼
RAG Knowledge Overlay (CLIP ViT-B-32 + FAISS, 160 KB, on-device)
 └─ Detected component → specs, LED-status meanings, troubleshooting, maintenance
```

## Models

| Model | Backend | Size | Status |
|---|---|---|---|
| **YOLOv8n-seg** (fine-tuned) | **QNN HTP (NPU)** | **3.2 MB** | ✅ Working — runs on the Hexagon NPU |
| YOLOv8n-seg (fine-tuned) | XNNPACK (CPU) | 13 MB | ✅ Working — proven live on S25 |
| CLIP ViT-B-32 + FAISS | CPU | 160 KB | ✅ RAG knowledge retrieval |
| RetinaNet-ResNet50-FPN | QNN HTP | 36 MB | ⚠️ Compiles to NPU; anchor-decode integration WIP |

The NPU model is the same fine-tuned YOLOv8n-seg, INT8-quantized and compiled for the
Hexagon HTP. Because the QNN backend can't compile YOLO's dynamic anchor decode, we
export only the **raw convolution feature maps** (pure convs → clean NPU lowering) and
run the lightweight decode (DFL → anchors → boxes → NMS) on-device in the app.

## 16 DC Component Classes

| ID | Class | What it detects |
|---|---|---|
| 0 | server rack | Full rack enclosure |
| 1 | compute tray | GPU/CPU trays (NVL72: 18 per rack) |
| 2 | NVLink switch tray | NVLink switches (NVL72: 9 per rack) |
| 3 | network switch | TOR / Ethernet / InfiniBand switches |
| 4 | power shelf | PSU banks (NVL72: 8 per rack, 6x 5.5kW each) |
| 5 | cable | Copper, fiber, power cables |
| 6 | network port | OSFP, QSFP, Ethernet, USB, HDMI ports |
| 7 | LED indicator | Status LEDs on trays/switches |
| 8 | label | Serial numbers, asset tags |
| 9 | fan | Cooling fans |
| 10 | cooling manifold | Liquid cooling pipes/connections |
| 11 | cable cartridge | NVLink copper cable cartridge |
| 12 | power connector | Power plugs, bus bar connectors |
| 13 | drive bay | NVMe drive slots |
| 14 | management port | BMC RJ45, serial console |
| 15 | DPU | BlueField DPU / ConnectX NIC |

See [MODEL_CLASSES.md](MODEL_CLASSES.md) for source-class mapping, indexing notes, and per-class detection reliability.

## Training Data

**2,036 human-labeled images** from 4 Roboflow datasets (CC BY 4.0), bootstrapped with a
**BrightData** web-scraping + Grounding DINO/SAM auto-labeling pipeline, augmented for
moiré patterns, screen glare, and brightness (so it survives real-world camera conditions):

| Dataset | Images | Source Classes |
|---|---|---|
| [Server Vision](https://universe.roboflow.com/hunters-workspace-kqclz/server-vision) | 1,293 | 30 (CPU sockets, DIMM slots, fans, PSUs, drives, ports) |
| [PC Ports](https://universe.roboflow.com/ports-gmmmp/pc-ports) | 327 | 4 (HDMI, RJ45, USB-A, USB-C) |
| [Ports](https://universe.roboflow.com/ym-pffnw/ports-yutnj) | 138 | 7 (DP, ethernet, power, USB variants) |
| [Server Detection](https://universe.roboflow.com/fujitsu-ih8ei/server-detection) | 54 | 9 (SFP, XFP, patch panels, servers) |

All source classes mapped to our 16 DC-Ops classes via `scripts/merge_datasets.py`.

## RAG Knowledge Base

On-device retrieval-augmented generation for component information:

- **80 knowledge chunks** covering all 16 component classes (description, specs, troubleshooting, LED meanings, maintenance)
- **NVIDIA GB200 NVL72**-specific documentation included
- CLIP ViT-B-32 text embeddings + FAISS nearest-neighbor search
- **160 KB** total index — fully on-device, no cloud needed

Example: detect "compute tray" → RAG returns:
> *"1RU compute tray: 2x Grace CPUs, 4x B200 GPUs, 192GB HBM3e. Amber LED = check BMC log. Verify coolant flow rate."*

## Tech Stack

| Component | Tool | Runtime |
|---|---|---|
| Detection (NPU) | Fine-tuned YOLOv8n-seg (INT8) | ExecuTorch → QNN HTP (Snapdragon NPU) |
| Detection (CPU) | Fine-tuned YOLOv8n-seg (FP32) | ExecuTorch → XNNPACK |
| On-device decode | DFL + anchors + NMS (Kotlin) | App-side post-processing |
| RAG retrieval | CLIP ViT-B-32 + FAISS | On-device |
| Android app | Kotlin + CameraX + ExecuTorch AAR | Galaxy S25 Ultra |
| Data pipeline | Roboflow + BrightData + Grounding DINO/SAM | Training-time |
| Model format | .pte (ExecuTorch) | Compiled for SM8750 |

## Target Device

- **Samsung Galaxy S25 Ultra** (SM-S938U1)
- **Chipset**: Snapdragon 8 Elite for Galaxy (SM8750-AC)
- **Hexagon**: v79 — **NPU**: Hexagon Tensor Processor (HTP)

## Project Structure

```
DC-Ops/
├── android-app/              # Android application (Kotlin, CameraX, ExecuTorch)
│   └── app/src/main/
│       ├── assets/           # .pte models (YOLO CPU + YOLO NPU + RetinaNet)
│       └── java/com/dcops/ar/
│           ├── inference/ModelManager.kt   # ExecuTorch inference + on-device YOLO decode
│           └── overlay/PolygonOverlayView.kt
├── data/knowledge_base/      # RAG index + component documentation
├── models/                   # Trained weights + .pte files (gitignored; on HF)
├── notebooks/                # Colab notebooks for training + QNN export
│   ├── train_merged_colab.ipynb          # Train YOLOv8n-seg on 2,036 images
│   ├── export_yolo_qnn_colab.ipynb       # YOLO → QNN HTP export (raw-map approach)
│   └── build_rag_colab.ipynb             # RAG index builder
├── scripts/                  # merge_datasets, build_rag_index, benchmark, setup
├── presentation/             # Pitch deck (PPTX + PDF)
├── MODEL_CLASSES.md          # Class reference + decode/indexing notes
└── README.md
```

## Setup

### Prerequisites
- Python 3.12, JDK 17, Android NDK v26, QAIRT SDK 2.46

### Quick Start
```bash
source scripts/setup_env.sh
```

### Build Android App
```bash
cd android-app
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

### Export YOLO → QNN HTP (NPU)
Upload `notebooks/export_yolo_qnn_colab.ipynb` to Colab (Linux required for QNN
compilation), set runtime, and **Run all**. Produces `dc_ops_yolo_qnn.pte` (3.2 MB).
Drop it into `android-app/app/src/main/assets/` — the app's `parseRawYoloOutput()`
decoder auto-activates for the NPU toggle.

## Demo Flow

1. Point phone at a server rack / mini PC → **Server Scan** draws live colored overlays
2. **Cable Match** mode → app highlights the exact port for the next cable
3. Tap a component → RAG info panel: specs, LED meanings, troubleshooting
4. Toggle **CPU ↔ NPU** live → same model, different hardware
5. Airplane mode → everything still works (air-gapped proof)

## Performance

### Fine-tuned YOLOv8n-seg (2,036 images)
| Class | mAP50 |
|---|---|
| compute tray | **0.924** |
| network port | **0.912** |
| drive bay | **0.912** |
| cooling manifold | **0.833** |
| power shelf | **0.832** |
| DPU / LED indicator | **0.808** |
| server rack | 0.683 |
| cable | 0.660 |
| **Overall (box)** | **0.749** |

Confusion matrices published on Hugging Face (`results/`).

### NPU vs CPU
| | XNNPACK (CPU) | QNN HTP (NPU) |
|---|---|---|
| Hardware | ARM Kryo cores | Hexagon Tensor Processor v79 |
| Precision | FP32 | INT8 |
| Size | 13 MB | **3.2 MB** |
| Speed / power | Baseline | Faster, more efficient (NPU-accelerated) |

## Models on Hugging Face

[abhijitbetigeri/dc-ops-dataset](https://huggingface.co/datasets/abhijitbetigeri/dc-ops-dataset)

| File | Description |
|---|---|
| `models/dc_ops_yolo_qnn.pte` | **YOLOv8n-seg QNN HTP — Snapdragon NPU (INT8, 3.2 MB)** |
| `models/dc_ops_yolov8n_seg_v3.pte` | YOLOv8n-seg XNNPACK — CPU (13 MB) |
| `models/dc_ops_yolov8n_seg_v3.pt` | YOLOv8n-seg PyTorch source weights (6.4 MB) |
| `models/dc_ops_retinanet_qnn.pte` | RetinaNet QNN HTP (36 MB, experimental) |
| `results/` | Confusion matrices |

## Team

| Role | Focus |
|---|---|
| Model Engineer | YOLOv8n-seg fine-tuning, QNN HTP export, on-device decode, RAG |
| Android Developer | CameraX, ExecuTorch AAR, Mission Control UI, backend toggle |
| Data + Demo | Roboflow datasets, 3D NVL72 model, presentation |

---

Built with **ExecuTorch + Qualcomm QNN HTP + Snapdragon 8 Elite** — real-time data-center inspection that never touches the cloud.
