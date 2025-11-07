# P-QCEA: Predictive Quality-Conscious Energy-Aware Routing Simulator

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**A cutting-edge network routing simulation framework comparing predictive vs. reactive algorithms**

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Architecture](#architecture) • [Documentation](#documentation)

</div>

---

## 🌟 Overview

**P-QCEA** (Predictive Quality-Conscious Energy-Aware) is an advanced network routing simulation platform that compares three routing paradigms:

1. **📊 Dijkstra (Baseline)** - Traditional shortest-path routing
2. **⚡ QCEA (Reactive)** - Multi-objective QoS-aware routing 
3. **🔮 P-QCEA (Predictive)** - Novel proactive routing with link quality prediction

### Why P-QCEA?

Traditional routing protocols react to network conditions **after** they occur. P-QCEA introduces a **predictive layer** that:

- 🎯 **Anticipates congestion** before it happens
- 🔋 **Optimizes energy consumption** across network nodes
- 📡 **Balances multiple QoS metrics** (latency, bandwidth, packet loss)
- 🧠 **Learns from historical patterns** using time-series prediction

---

## ✨ Features

### 🎮 Interactive Dashboard
- **Real-time visualization** of network topology with live link utilization
- **Comparative analytics** across all three algorithms side-by-side
- **Customizable simulation parameters** (topology size, traffic mix, routing weights)
- **Traffic-aware routing** supporting VoIP, Video, and Best-Effort flows

### 🧪 Advanced Simulation Engine
- **Scale-free topologies** (Barabási-Albert model) for realistic networks
- **Dynamic network conditions** with bandwidth fluctuations, delay jitter, and loss bursts
- **Energy-aware routing** with per-node residual energy tracking
- **Heterogeneous link capacities** and QoS parameters

### 📈 Comprehensive Metrics
- **Latency** - Average end-to-end path delay
- **Packet Delivery Rate (PDR)** - Success rate under realistic loss conditions
- **Energy Efficiency** - Total network energy consumption tracking
- **Jitter** - Latency variance for real-time traffic quality
- **Prediction Accuracy** - Confidence and usage rate of predictions (P-QCEA specific)

### 🔬 Research-Grade Features
- **Reproducible experiments** with random seed control
- **Configurable traffic classes** with distinct QoS requirements
- **Exponential smoothing** for noise reduction
- **Linear trend detection** for prediction
- **Confidence scoring** for prediction reliability

---

## 🚀 Installation

### Prerequisites
- **Python 3.8+**
- pip package manager

### Step 1: Clone the Repository

### Step 2: Create Virtual Environment (Recommended)

### Step 3: Install Dependencies

---

## 🎯 Quick Start

### Launch the Dashboard

Then open your browser to: **http://127.0.0.1:8050/**

### Run Your First Simulation

1. **Click "Simulation Controls"** to expand the parameter panel
2. **Configure simulation**:
   - Time Steps: 100
   - Number of Nodes: 25
   - Flows per Step: 5
3. **Set traffic mix** (VoIP/Video/Data distribution)
4. **Select algorithm**: P-QCEA (Predictive)
5. **Click "► Start"** and watch the real-time visualization!

### Understanding the Dashboard

#### 📊 KPI Grid (4 rows × 3 columns)
- **Row 1**: Average Path Latency (ms) - Dijkstra | QCEA | P-QCEA
- **Row 2**: Total Residual Energy (J) - Energy remaining in network
- **Row 3**: Packet Delivery Rate (%) - Successful packet delivery
- **Row 4**: Average Jitter (ms) - Latency variance for QoS

#### 📈 Visualization Charts
- **Network Topology** - Interactive graph with live link utilization (green = low, yellow = medium, red = congested)
- **Latency Over Time** - Comparative plot showing all three algorithms
- **Energy Consumption** - Residual energy tracking across simulation
- **Packet Delivery Rate** - Success rate trends
- **Prediction Usage** - P-QCEA-specific: percentage of paths using predictions
- **Prediction Confidence** - Reliability scores for predictions (0-100%)

---

## 🧠 How It Works

### The P-QCEA Algorithm

#### 1️⃣ **Prediction Layer**
Uses **exponential smoothing** combined with **linear regression** to predict future link conditions:
- **Smoothing** reduces noise in measurements
- **Trend detection** identifies patterns (improving/degrading links)
- **Confidence scoring** determines when predictions are reliable enough to use

#### 2️⃣ **Multi-Objective Cost Function**
Combines five weighted metrics:
- **Latency (wl)** - Propagation delay
- **Bandwidth (wb)** - Link capacity
- **Packet Loss (wp)** - Loss probability
- **Energy (we)** - Node residual energy
- **Cost (wc)** - Transmission cost (inverse bandwidth)

#### 3️⃣ **Confidence Scoring**
Only uses predictions when confidence exceeds threshold (default: 70%):
- Low variance in history → High confidence
- High variance → Falls back to current measurements
- Prevents bad predictions from harming performance

### Traffic-Aware Routing

The system automatically adjusts weights based on traffic class:

| Class | Priority | Characteristics | Use Case |
|-------|----------|----------------|----------|
| **VoIP** | 🔴 High | Low latency, loss-sensitive | Real-time voice calls |
| **Video** | 🟡 Medium | High bandwidth, moderate latency | Streaming services |
| **Best-Effort** | 🟢 Low | Energy-aware, cost-conscious | File transfers, emails |

---

## 📁 Project Structure
qcea-sim/
├── src/ # Core simulation engine
│ ├── topology.py # Network graph generation (scale-free)
│ ├── traffic.py # Traffic flow generation with classes
│ ├── predictor.py # Time-series prediction module
│ ├── pqcea_routing.py # P-QCEA algorithm (predictive)
│ ├── qcea_routing.py # QCEA algorithm (reactive)
│ └── baseline_dijkstra.py # Dijkstra baseline
├── frontend/ # Dash web dashboard
│ ├── app.py # Dash app initialization
│ ├── layout.py # UI components
│ ├── callbacks.py # Interactive logic
│ └── assets/
│ └── style.css # Dark theme styling
├── config/ # Configuration files
│ ├── pqcea_config.yaml # Default simulation config
│ └── example_config.yaml # Example configuration
├── requirements.txt # Python dependencies
├── run_dashboard.py # Dashboard entry point
└── README.md # This file


---

## 🔧 Configuration

### Simulation Parameters

Edit `config/pqcea_config.yaml` to customize:

**Topology Settings:**
- Number of nodes (10-100)
- Topology type (scale-free)
- Random seed for reproducibility

**Simulation:**
- Time steps (simulation duration)
- Number of flows per step
- Traffic class distribution (VoIP/Video/Best-Effort)

**Prediction (P-QCEA):**
- Prediction horizon (steps ahead)
- Smoothing factor alpha (0-1)
- Confidence threshold (0-1)

**Routing Weights:**
- Can use AUTO mode (traffic-specific) or MANUAL mode (global)
- Five weight parameters: wl, wb, wp, we, wc

**Dynamic Network:**
- Bandwidth variation
- Delay jitter
- Loss burst probability

---

## 📊 Research Context

### Problem Statement

Modern networks face three major challenges:

1. **QoS Degradation** - Traditional routing ignores latency, bandwidth, and loss
2. **Energy Inefficiency** - No consideration for battery-powered nodes (IoT, mobile)
3. **Reactive Nature** - Protocols respond to problems *after* they occur

### Our Solution: P-QCEA

We introduce a **predictive routing protocol** that:

- ✅ **Predicts** future link conditions using lightweight time-series analysis
- ✅ **Optimizes** multiple objectives (latency, bandwidth, energy, loss)
- ✅ **Adapts** routing decisions based on traffic class requirements
- ✅ **Learns** from historical patterns without heavy ML overhead

### Key Innovations

1. **Exponential Smoothing + Linear Regression** - Simple yet effective prediction
2. **Confidence-Based Fallback** - Only use predictions when reliable
3. **Traffic-Aware Weight Adaptation** - Automatic optimization per flow type
4. **Energy Tracking** - Realistic battery drain simulation

### Expected Results

Based on the simulation framework, P-QCEA enables comparison of:

- Latency reduction vs. traditional shortest-path
- Energy efficiency improvements vs. reactive QoS routing
- Packet delivery rate under dynamic network conditions
- Prediction accuracy and confidence trends

---

## 🎨 Dashboard Customization

### Color Scheme

The dashboard uses a modern fintech-inspired dark theme:

- **P-QCEA**: Vibrant Magenta (#E040FB)
- **QCEA**: Lime Green (#BEF264)
- **Dijkstra**: Light Blue (#38BDF8)

Colors can be customized in `frontend/assets/style.css` under CSS variables.

### Extending the Dashboard

Add new metrics by:
1. Computing the metric in `run_simulation_step()` in `callbacks.py`
2. Adding a KPI card in `layout.py`
3. Creating a new chart figure
4. Linking with a callback

---

## 🧪 Advanced Usage

### Custom Topology

Generate your own topology programmatically and save for later use.

### Traffic Mix Experiments

Test different traffic distributions:
- **VoIP-heavy** (70% VoIP, 20% Video, 10% Data)
- **Video-heavy** (20% VoIP, 70% Video, 10% Data)
- **Mixed** (33% each)

### Weight Sensitivity Analysis

Compare performance with different weight configurations:
- **Latency-focused**: wl=0.7, others=0.075
- **Energy-focused**: we=0.7, others=0.075
- **Balanced**: all weights=0.2

---

## 🐛 Troubleshooting

### Dashboard won't start
- Check if port 8050 is already in use
- Verify Python path includes project root
- Ensure all dependencies are installed

### Slow predictions
If prediction usage rate is low (< 30%):
1. Lower confidence threshold: 0.7 → 0.5
2. Increase history size: 15 → 30
3. Adjust smoothing: alpha=0.3 → alpha=0.5 (more reactive)

### Memory issues with large topologies
For > 100 nodes:
1. Reduce time steps: 200 → 50
2. Decrease history size: 15 → 10
3. Limit number of flows

---

## 🤝 Contributing

We welcome contributions! Areas for enhancement:

- 🎯 Additional routing algorithms (OSPF, BGP variants)
- 📊 More sophisticated prediction models (ARIMA, LSTM)
- 🌐 Real-world topology datasets (CAIDA, Internet2)
- 📈 Statistical significance testing framework
- 🎨 Enhanced visualizations (3D graphs, heatmaps)

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **NetworkX** - Graph manipulation library
- **Plotly/Dash** - Interactive visualization framework
- **Research community** - For inspiration from papers on QoS-aware and energy-efficient routing

---

## 📬 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/qcea-sim/issues)
- **Discussions**: For questions and feature requests

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ for the networking research community

[Back to Top](#p-qcea-predictive-quality-conscious-energy-aware-routing-simulator)

</div>