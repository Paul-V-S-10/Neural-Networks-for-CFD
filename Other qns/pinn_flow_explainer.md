# 🌊 Flow Past an Ellipse Using PINNs — Simple Guide

---

## 📌 What Are We Solving?

We are simulating how a **fluid (like air or water) flows around an ellipse-shaped object**.

Think of it like placing an egg-shaped rock in a river and watching how water moves around it.

We use a special kind of neural network called a **PINN (Physics-Informed Neural Network)** to solve this — instead of traditional mesh-based solvers, the neural network *learns* the solution by respecting the laws of physics.

---

## 📌 Block 1 — Imports and Setup

```python
# Block 1: Imports and device setup
```

### What is happening here?

- We import **PyTorch** — the deep learning library used to build and train the neural network
- We import **NumPy and Matplotlib** — for math and plotting
- We check if a **GPU is available** — if yes, training happens on the GPU (much faster)

### Why GPU?

The neural network needs to do millions of math operations. A GPU (like your RTX 4060) can do thousands of these operations *at the same time*, making training ~10x faster than a CPU.

---

## 📌 Block 2 — The Neural Network (PINN Architecture)

```python
# Block 2: NavierStokesPINN class
```

### What is this network doing?

The network takes **two inputs**: the x and y coordinates of a point in the flow domain.

It outputs **three values** at that point:
- `u` → how fast the fluid moves horizontally (left-right)
- `v` → how fast the fluid moves vertically (up-down)
- `p` → the pressure at that point

### Simple analogy

Imagine you could poke any location in the river with a stick and instantly know the speed and pressure of the water there. That is exactly what the network learns to do — for every (x, y) point in the domain.

The network has **8 hidden layers** with 80 neurons each. More layers = better at learning complex flow patterns.

---

## 📌 Block 3 — Sampling Points

```python
# Block 3: Sampling helper functions
```

### What is happening here?

We randomly pick thousands of points inside our flow domain (the rectangular region around the ellipse). These are called **collocation points** — the locations where we will enforce the physics equations.

We also pick points on:
- The **ellipse surface** → to enforce no-slip (fluid doesn't move at the wall)
- The **inlet** (left edge) → to enforce incoming flow speed = 1
- The **outlet** (right edge) → to enforce zero pressure
- The **top/bottom walls** → to enforce slip condition (fluid glides along the wall)

### Simple analogy

Think of it like placing sensors all over the river — some in the water, some on the rock surface, some at the entry point. We train the network to give correct readings at all these sensor locations.

---

## 📌 Block 4 — The Physics (Loss Function)

```python
# Block 4: compute_loss function
```

### This is the most important block. Here's the concept:

A regular neural network learns by minimizing the difference between its predictions and known answers.

A **PINN learns by minimizing how much it violates the laws of physics.**

The physics here are the **Navier-Stokes equations** — the fundamental equations that govern all fluid flow:

| Equation | What it means in simple terms |
|---|---|
| **Continuity** | Fluid doesn't appear or disappear — what flows in must flow out |
| **Momentum (x)** | Newton's second law for fluid in the horizontal direction |
| **Momentum (y)** | Newton's second law for fluid in the vertical direction |

The **total loss = PDE error + boundary condition errors**

The network is penalized whenever:
- It violates the Navier-Stokes equations at collocation points
- It predicts non-zero velocity on the ellipse surface (no-slip condition)
- It predicts the wrong inlet velocity
- It predicts wrong pressure at the outlet

### Simple analogy

Imagine teaching someone to draw water flow by punishing them every time they draw water flowing uphill, or appearing out of nowhere. That's exactly what the loss function does — it punishes physically impossible predictions.

---

## 📌 Block 5 — Training Loop

```python
# Block 5: train_pinn function
```

### What is happening here?

This is where the network actually **learns**. Here's the process step by step:

1. Pick random points in the domain
2. Run them through the network → get predicted u, v, p
3. Compute how badly physics is violated (the loss)
4. Use **backpropagation** to adjust the network weights slightly
5. Repeat thousands of times until the loss is small

We use the **Adam optimizer** — a smart algorithm that adjusts how big each weight update step is.

We also use a **cosine learning rate scheduler** — the step size starts moderate, then gradually shrinks as training progresses (like taking big steps when far from the answer, small careful steps when close).

### Why do we resample collocation points every 500 epochs?

To avoid the network memorizing specific points. By using fresh random points, we make sure it learns the physics everywhere in the domain.

---

## 📌 Block 6 & 7 — Evaluation and Recirculation Length

```python
# Block 6 & 7: evaluate_model and measure_recirculation
```

### Evaluation

After training, we create a fine grid of points covering the entire domain and ask the network to predict u, v, p at every grid point. This gives us smooth flow fields that we can plot.

### What is Recirculation Length (Lr)?

When fluid flows around the ellipse, it separates at the rear and creates a **"dead zone"** directly behind it — a pocket where fluid actually flows *backward*.

**Lr = the length of this backward-flow pocket**, measured from the rear of the ellipse to the point where forward flow resumes on the centreline (y = 0).

```
Ellipse rear ←————— Lr —————→ point where u = 0 again
```

- **Small Lr** = flow reattaches quickly = smooth, well-behaved wake
- **Large Lr** = long recirculation bubble = flow wants to become unsteady

---

## 📌 Block 8 — Plotting

```python
# Block 8: plot_results function
```

### What are the four plots?

| Plot | What it shows |
|---|---|
| **u (streamwise velocity)** | How fast fluid moves left-right. Red = fast forward, Blue = slow/backward |
| **v (cross-stream velocity)** | Up-down motion. Shows how fluid deflects around the ellipse |
| **p (pressure)** | High pressure at front of ellipse (fluid being pushed), low pressure at rear |
| **Streamlines** | The actual paths fluid particles follow — like tracing ink drops in a river |

---

## 📌 Block 9 — Running All Reynolds Numbers

```python
# Block 9: Main loop — Re = 10, 50, 100, 500, 1000
```

### What is Reynolds Number (Re)?

Reynolds number tells you the **ratio of inertia forces to viscous forces** in the flow.

| Re | Behaviour | Real-world analogy |
|---|---|---|
| **Re = 10** | Very smooth, fully attached flow | Honey flowing around an object |
| **Re = 50** | Small symmetric wake bubble | Slow water around a pebble |
| **Re = 100** | Longer wake, approaching instability | Water around a stone in a stream |
| **Re = 500** | Unsteady in reality, PINN gives steady approximation | Fast river around a rock |
| **Re = 1000** | Strongly unsteady (vortex shedding) in reality | Wind around a building |

> **⚠️ Note for Re = 500 and 1000:**
> In real life, the flow at these Reynolds numbers is **unsteady** — vortices peel off the ellipse alternately from top and bottom (called a Kármán vortex street), like a flag flapping in the wind.
>
> The PINN has **no time dimension**, so it finds the steady solution that *mathematically exists* but would be unstable in a real experiment. Think of it like finding the exact position where a pencil balances on its tip — valid mathematically, impossible to hold in practice.

---

## 📌 Block 10 — Summary Table

```python
# Block 10: Print recirculation length table
```

### What to expect

| Re | Lr (approx) | Interpretation |
|---|---|---|
| 10 | ~0 | Almost no recirculation |
| 50 | ~0.5–1.0 | Small bubble |
| 100 | ~1.5–2.5 | Moderate bubble |
| 500 | ~4–7 | Long bubble (steady PINN approx.) |
| 1000 | ~7–12 | Very long bubble (steady PINN approx.) |

The trend is clear: **higher Re = longer wake = more complex flow**.

---

## 🔑 Key Takeaways

1. **PINNs** solve fluid equations by training a neural network to satisfy physics — no mesh needed
2. **Reynolds number** controls whether flow is smooth (low Re) or chaotic with vortices (high Re)
3. **Recirculation length Lr** measures the size of the backward-flow pocket behind the ellipse
4. At **Re = 500 and 1000**, real flow sheds vortices periodically — the PINN gives the steady-state approximation which is a mathematically valid but physically unstable solution
5. Your **RTX 4060 GPU** will handle all of this efficiently via CUDA — total runtime ~30–40 minutes
