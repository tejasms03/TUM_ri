# TUM_ri — Reinforcement Learning for Robotic Locomotion

This repository contains code and documentation supporting research on applying **Reinforcement Learning (RL)** algorithms for **motion planning and locomotion** on legged robots — specifically **hexapod** and **quadruped** platforms. The work evolved from simulations using PyBullet to custom hardware design and real‑world testing.

---

## 🚀 Project Overview

The goal of this research is to develop robust locomotion strategies using reinforcement learning that can be transferred from simulation environments to physical robots.

Main contributions include:

- A **simulated hexapod locomotion framework** using PyBullet.
- Custom design and development of a **quadruped robot** body.
- Code and tools to train, evaluate, and analyze RL policies for dynamic gait generation.
- Supporting data and visual results demonstrating locomotion performance.

---

## 🧠 Key Features

- 🔁 **Reinforcement Learning (RL) integration** for adaptive robot motion control.
- 🐜 **Hexapod simulation environment** for fast iteration.
- 🐾 **Quadruped real‑world implementation** in progress.
- 📊 Included media and documentation to illustrate motion results.
- 📝 Contains thesis report and supporting materials.

---

## 📁 Repository Structure

```
TUM_ri/
├── gym_hexapod_zoo.py            # Main RL training and simulation code
├── dachshund parts.zip           # 3D model assets / design files for robot
├── rob-dynam_description.zip     # Robot dynamics or description data
├── hexapod-movement-project.mp4  # Video of simulation/robot movement
├── motor control.mov             # Another demonstration video
├── thesis report (1).pdf         # Research thesis / final write‑up
├── README.md                     # This file
└── other assets / visuals        # Additional media
```

> ❗ *Note:* The main executable code relevant to training and running locomotion policies is in `gym_hexapod_zoo.py`. Other files include design assets, videos, and documentation.

---

## 🛠 Getting Started

### Prerequisites

Install the following on your machine:

- Python ≥ 3.7  
- PyBullet (`pip install pybullet`)  
- RL libraries such as Stable Baselines3 or custom frameworks  
- Standard scientific Python libraries (`numpy`, `gym`, etc.)

### Usage

1. **Clone this repository**

   ```bash
   git clone https://github.com/tejasms03/TUM_ri.git
   cd TUM_ri
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run training/simulation**

   ```bash
   python gym_hexapod_zoo.py
   ```

   *Refer to inline comments in `gym_hexapod_zoo.py` for details on simulation configuration.*

---

## 📘 Example

The `hexapod-movement-project.mp4` contains a demonstration of locomotion learned through reinforcement learning in simulation. Adjust parameters in the script to customize robot morphology, RL algorithms, or reward functions.

---

## 📄 Research & Documentation

Included in the repository:

- **`thesis report (1).pdf`** — detailed write‑up of problem statement, methodology, experiments, and conclusions.
- Design zip files for **robot assets** and **dynamics descriptions**.

---

## 🎯 Future Directions

- Transfer learned policies from simulation to real quadruped hardware.
- Improve robustness to environment variations.
- Integrate additional sensors and perception for adaptive navigation.

---

## 🧑‍💻 Contributing

Contributions are welcome! To contribute:

1. Fork this repository.
2. Create your feature branch: `git checkout -b my-feature`
3. Commit changes: `git commit -m "Add feature"`
4. Push branch: `git push origin my-feature`
5. Open a Pull Request.

For major changes, please open an issue first for discussion.

---

## 📜 License

Currently not specified — you may want to add an open‑source license such as MIT or BSD.

---

## ❓ Questions

If you have questions about the project structure or research goals, feel free to open an issue or contact the maintainer.

