# FedBuff Codebase: A Complete Learning Guide

---

## 1. What Is This Project?

Imagine you want to train a machine learning model, but the training data is spread across millions of phones — and you can't move that data to a central server (for privacy). Instead, the server sends the model to phones, each phone trains locally on its own data, and sends back only the *changes* (called a **gradient** or **update**).

This is **Federated Learning (FL)**.

Now, the problem with classic FL: the server has to **wait for every phone to finish** before it can update the model. If one phone is slow, everyone waits. This is the "straggler problem."

This code reproduces the paper **"Federated Learning with Buffered Asynchronous Aggregation"** (FedBuff, AISTATS 2022). The paper's solution: don't wait for everyone. Instead, **collect K updates in a buffer**, then aggregate — no waiting. This is *asynchronous* FL.

The code implements FedBuff alongside several baselines (FedAvg, FedAsync, FedProx, FedAvgM) so you can compare them fairly.

---

## 2. Files — What to Focus On and What to Ignore First

### Focus on these first (the core)
```
run_experiment.py       ← entry point, how experiments start
config.py               ← all settings/hyperparameters in one place
simulator.py            ← the engine that runs everything
algorithms/base.py      ← defines what a "client" and "server" are
algorithms/fedbuff.py   ← the main algorithm from the paper
```

### Read second (the details)
```
algorithms/fedavg.py    ← synchronous baselines for comparison
algorithms/fedasync.py  ← fully async baseline
utils/delay.py          ← how client slowness is simulated
utils/metrics.py        ← how results are measured and saved
```

### Leave for later (important but not foundational)
```
utils/dp.py             ← differential privacy (advanced topic)
utils/models.py         ← neural network architectures
datasets/cifar10.py     ← how CIFAR-10 is split across clients
datasets/leaf.py        ← Sent140 and CelebA datasets
```

---

## 3. How the Program Starts — Step by Step

When you run:
```bash
python run_experiment.py --algorithm fedbuff --dataset cifar10 --concurrency 100 --K 10
```

Here is exactly what happens:

**Step 1 — `run_experiment.py:195` `main()`**
Parses command-line arguments. Routes to either a single experiment or a batch mode.

**Step 2 — `run_experiment.py:252` `get_config(...)`**
Loads `ExperimentConfig` from `config.py` with all hyperparameters — learning rates, buffer size, delay distribution, etc. The pre-tuned values from the paper's Table 4 are applied automatically.

**Step 3 — `run_experiment.py:34` `run_single(config)`**
Creates a `FLSimulator` object, calls `simulator.run()`, saves results to a JSON file.

**Step 4 — `simulator.py:40` `FLSimulator.__init__`**
Sets up everything:
- Loads the dataset (e.g., CIFAR-10 split across 5000 clients)
- Creates the neural network model
- Creates the server (e.g., `FedBuffServer`)
- Creates the delay simulator
- Creates staleness tracker

**Step 5 — `simulator.py:160` `simulator.run()`**
Evaluates the model once at the start, then calls either `_run_sync()` or `_run_async()` depending on the algorithm.

---

## 4. Connecting the Paper to the Code

| Paper Concept | Where in Code |
|---|---|
| "Client trips" as the evaluation metric | `simulator.py:164` `total_trips` counter |
| K = buffer size | `config.py:23` `buffer_size`, `fedbuff.py:30` |
| Staleness τᵢ(t) = current\_version − download\_version | `utils/delay.py:62` `compute_staleness()` |
| Staleness scaling s(τ) = 1/(1+τ)^α | `utils/delay.py:70` `staleness_scaling()` |
| Algorithm 1 (FedBuff) | `algorithms/fedbuff.py:54` `receive_update()` + `_aggregate_and_step()` |
| Server update w\_{t+1} = w\_t − η\_g × Δ\_t | `algorithms/base.py:142` `apply_update()` |
| Server momentum (FedAvgM) | `algorithms/base.py:152-169` |
| FedProx proximal term | `algorithms/base.py:95-99` in `local_train()` |
| LR normalization (Section 5) | `algorithms/base.py:85-88` |
| DP-FTRL tree aggregation | `utils/dp.py:61` `TreeAggregation` |
| Half-normal delay distribution | `utils/delay.py:34` |
| Table 4 hyperparameters | `config.py:61` `BEST_HYPERPARAMS` |

---

## 5. One Complete Training Cycle — Traced Through the Code

Let's follow a single client update in FedBuff from start to finish.

---

### Phase A: Client Downloads the Model

In `simulator.py:246`, the async loop creates a new client slot:

```python
download_version = self.staleness_tracker.record_download()  # records version number
model_snapshot = self.server.get_model_copy()               # deep copy of global model
```

`record_download()` (`delay.py:57`) just returns the current server version number (e.g., version 5). This is stamped onto the client so we can measure staleness later.

---

### Phase B: Client Trains Locally

In `simulator.py:262`, after the client "finishes" (simulated by popping the earliest finish time):

```python
delta = client.local_train(earliest['model_snapshot'], lr=cfg.client_lr, ...)
```

Inside `algorithms/base.py:48` `FLClient.local_train()`:

1. **Save initial weights**: `initial_params = {name: p.data.clone() for ...}` (line 68)
2. **Run one epoch of SGD** over the client's local data (lines 74–109):
   - Forward pass → compute loss
   - (Optional) add FedProx proximal penalty to keep client weights close to global weights
   - Backward pass → compute gradients
   - Manual SGD step: `p.data -= effective_lr * p.grad`
3. **Compute delta** = initial weights − trained weights (line 111–114):
   ```python
   delta[name] = initial_params[name] - p.data
   ```

> **Why subtract?** The "delta" is the *direction the client wants to move the global model*. If the client's training moved its weights in some direction, we want the server to move in that same direction. So: initial − final gives the update vector.

---

### Phase C: Client Sends Update Back

Back in `simulator.py:269`, a `ClientUpdate` is created:

```python
update = ClientUpdate(
    client_id=...,
    delta=delta,            # the weight changes
    num_samples=...,        # how much data this client has
    download_version=...,   # what version of the model it started from
    staleness=staleness,    # how many server updates happened since download
    delay=...,
)
```

`staleness` is computed at `delay.py:62`:
```python
return self.current_server_version - download_version
```
If the server updated 3 times while this client was training (e.g., server is now at version 8, client downloaded version 5), staleness = 3.

---

### Phase D: FedBuff Server Receives the Update

In `simulator.py:280`:
```python
updated = self.server.receive_update(update)
```

In `fedbuff.py:54` `receive_update()`:
1. (Optional) Clip the update for DP privacy
2. Append to `self.buffer`
3. Check: `if len(self.buffer) >= self.buffer_size:` (K updates collected?)
4. If yes → call `_aggregate_and_step()`

---

### Phase E: Aggregation (the Core of FedBuff)

In `fedbuff.py:79` `_aggregate_and_step()`:

```python
for update in self.buffer:
    weight = staleness_scaling(update.staleness, self.staleness_alpha)
    # weight = 1 / (1 + staleness)^0.5
    # Fresh updates get weight 1.0, stale updates get lower weight

    for name, delta in update.delta.items():
        aggregated[name] += weight * delta   # weighted sum of all K updates

for name in aggregated:
    aggregated[name] /= K   # divide by K to get average
```

This implements the paper's formula:

**Δ\_t = (1/K) × Σᵢ s(τᵢ) × Δᵢ**

where s(τ) = 1/(1+τ)^α is the staleness penalty.

---

### Phase F: Server Model Is Updated

`_aggregate_and_step()` calls `self.apply_update(aggregated)` which lives in `algorithms/base.py:142`:

```python
# Without momentum:
for name, p in self.model.named_parameters():
    p.data -= self.server_lr * aggregated_delta[name]

# With momentum (FedAvgM):
momentum_buffer[name] = beta * momentum_buffer[name] + aggregated_delta[name]
p.data -= self.server_lr * momentum_buffer[name]
```

This is: **w\_{t+1} = w\_t − η\_g × Δ\_t**

Then `self.server_version += 1` and `self.buffer.clear()` — ready for the next K updates.

---

## 6. Architecture Diagrams

### Overall Architecture

```
run_experiment.py
        │
        ▼
   get_config()          ← config.py: all hyperparameters
        │
        ▼
  FLSimulator            ← simulator.py: the engine
  ┌─────────────────────────────────────────────┐
  │  FLDataset  (datasets/)                      │
  │  Model      (utils/models.py)                │
  │  FLServer   (algorithms/)                    │
  │  DelaySimulator (utils/delay.py)             │
  │  StalenessTracker (utils/delay.py)           │
  │  ExperimentMetrics (utils/metrics.py)        │
  └─────────────────────────────────────────────┘
        │
        ▼
  simulator.run()
  ├── _run_sync()   for FedAvg, FedAvgM, FedProx
  └── _run_async()  for FedBuff, FedAsync
```

---

### Client-Server Interaction (Async / FedBuff)

```
SERVER                          CLIENT(s)
  │                                │
  │──── send model snapshot ──────▶│  (records download_version)
  │                                │
  │                                │  local_train() — one epoch SGD
  │                                │  compute delta = initial - trained
  │                                │
  │◀─── ClientUpdate(delta, ───────│
  │      staleness, delay)         │
  │                                │
  │  [buffer.append(update)]       │
  │                                │
  │  if len(buffer) >= K:          │
  │    aggregate K updates         │
  │    apply_update()              │
  │    server_version += 1         │
  │    buffer.clear()              │
  │                                │
  (repeat)
```

---

### Synchronous vs Asynchronous Mode

```
SYNCHRONOUS (FedAvg / FedAvgM / FedProx)
─────────────────────────────────────────────────────────────
Round 1: [client1, client2, ..., clientN] → server waits → aggregate all → update
Round 2: [client1, client2, ..., clientN] → server waits → aggregate all → update
         ▲ server blocks until ALL N clients finish ▲

ASYNCHRONOUS (FedBuff)
─────────────────────────────────────────────────────────────
Timeline:
  client1 ──trains──▶ arrives (staleness=0) → buffer [1]
  client3 ──────trains──▶ arrives (staleness=1) → buffer [1,2]
  client2 ──────────trains──▶ arrives (staleness=2) → buffer [1,2,3]
                                                  ↓ buffer full (K=3)!
                                              aggregate → update server
  (no waiting — clients upload as they finish)
```

---

### Data Flow

```
datasets/cifar10.py
    CIFAR-10 (50K images)
         │
         │ Dirichlet(0.1) split
         ▼
  5000 client datasets (non-IID)
         │
         │ get_client_dataset(client_id)
         ▼
   FLClient.local_train()
         │
         │ delta = initial_weights - trained_weights
         ▼
   ClientUpdate (delta, staleness, num_samples)
         │
         │ receive_update()
         ▼
   FedBuffServer.buffer (collect K)
         │
         │ _aggregate_and_step()
         ▼
   apply_update() → global model updated
         │
         │ evaluate_model()
         ▼
   ExperimentMetrics → saved as JSON
```

---

### Training Workflow

```
simulator.run()
      │
      ├─── [sync algorithms] ──────────────────────────────────────────┐
      │         _run_sync()                                             │
      │         ┌────────────────────────────────────────────────────┐ │
      │         │ while total_trips < budget:                        │ │
      │         │   for each client in cohort:                       │ │
      │         │     _train_client() → ClientUpdate                 │ │
      │         │   server.process_updates(all_updates)              │ │
      │         │   total_trips += cohort_size                       │ │
      │         │   evaluate periodically                            │ │
      │         └────────────────────────────────────────────────────┘ │
      │                                                                 │
      └─── [async algorithms] ─────────────────────────────────────────┘
                _run_async()
                ┌────────────────────────────────────────────────────┐
                │ while total_trips < budget:                        │
                │   fill slots up to concurrency:                    │
                │     assign each client a finish_time = delay       │
                │     snapshot current model                         │
                │   pop earliest-finishing client                    │
                │   local_train(snapshot) → delta                    │
                │   compute staleness                                │
                │   server.receive_update(ClientUpdate)              │
                │     → if buffer full: aggregate + update model     │
                │   total_trips += 1                                 │
                │   evaluate periodically                            │
                └────────────────────────────────────────────────────┘
```

---

## 7. Most Important Classes and Functions

| Class / Function | File | What It Does |
|---|---|---|
| `FLSimulator` | `simulator.py:30` | Top-level orchestrator. Controls the entire training loop. |
| `FLSimulator._run_async()` | `simulator.py:221` | The async training loop — manages timing, client slots, ordering by finish time. |
| `FLClient.local_train()` | `algorithms/base.py:48` | A client does one epoch of SGD and returns the delta. The heart of FL. |
| `FedBuffServer.receive_update()` | `fedbuff.py:54` | Adds an update to the buffer; triggers aggregation when buffer is full. |
| `FedBuffServer._aggregate_and_step()` | `fedbuff.py:79` | Weighted averaging of K updates with staleness scaling; updates global model. |
| `FLServer.apply_update()` | `algorithms/base.py:142` | Applies aggregated delta to model with optional server momentum. |
| `staleness_scaling()` | `utils/delay.py:70` | s(τ) = 1/(1+τ)^α — penalizes old updates. |
| `StalenessTracker` | `utils/delay.py:47` | Tracks which server version each client downloaded; computes staleness. |
| `get_config()` | `config.py:101` | Creates a config pre-filled with best hyperparameters from the paper. |
| `evaluate_model()` | `utils/metrics.py` | Computes val accuracy and loss to track progress. |

### How They Interact

```
FLSimulator
  ├── uses FLClient.local_train()   to simulate each client
  ├── uses DelaySimulator           to assign finish times
  ├── uses StalenessTracker         to record download versions
  ├── calls server.receive_update() or server.process_updates()
  │       └── FedBuffServer._aggregate_and_step()
  │               ├── staleness_scaling()       for weights
  │               └── FLServer.apply_update()   for model update
  └── calls evaluate_model()        periodically to log progress
```

---

## 8. The 20% of Code That Explains 80% of the System

Read these in this order — this is the essential core:

**1. `algorithms/base.py:48-116` — `FLClient.local_train()`**
This is where learning actually happens. One client, one epoch, returns a delta.

**2. `algorithms/base.py:142-171` — `FLServer.apply_update()`**
This is where the global model is updated. One formula: `w -= lr * delta`.

**3. `algorithms/fedbuff.py:54-116` — `receive_update()` + `_aggregate_and_step()`**
This is the entire FedBuff algorithm. Buffer K updates, average them with staleness weights, update model.

**4. `utils/delay.py:70-76` — `staleness_scaling()`**
This single function implements the key idea for handling stale updates.

**5. `simulator.py:221-287` — `_run_async()`**
This is the main simulation loop. All the timing, ordering, and orchestration lives here.

---

## 9. Key Concepts Explained Simply

**Delta / Pseudo-gradient**
Instead of sending raw gradients, clients send the *difference* between their starting weights and their trained weights. This represents "how far and in which direction the client moved."

**Staleness**
If the server has updated 5 times since a client downloaded the model, that client's update is 5 versions stale. It's like submitting homework based on last week's assignment when the teacher has already updated it 5 times.

**Staleness scaling**
Old updates are less trustworthy, so we multiply them by a smaller number. A fresh update gets weight 1.0; an update that is 3 versions stale gets weight 1/(1+3)^0.5 ≈ 0.5.

**Client trips**
The paper counts "client trips" (each time a client downloads + trains + uploads = 1 trip) instead of wall-clock time. This makes comparisons fair across different hardware.

**Buffer size K**
The key hyperparameter. K=1 means update after every single client (FedAsync). K=1000 means collect 1000 updates before aggregating (closer to synchronous). K=10 is the sweet spot in the paper.

**Concurrency**
How many clients are training in parallel at any moment. In async FL, you can set this independently of K. In sync FL, concurrency equals cohort size.

**Non-IID data**
"Non-IID" means each client has a different distribution of data — one phone might have only cat photos, another only dog photos. This is realistic and makes FL harder. The code simulates this using a Dirichlet(0.1) split in `datasets/cifar10.py`.

**Server learning rate (η\_g) vs client learning rate (η\_l)**
Clients use η\_l when doing local SGD. The server uses η\_g when applying the aggregated delta to the global model. Both are tuned separately (see `BEST_HYPERPARAMS` in `config.py`).

---

## 10. Learning Roadmap

### Read First — understand the core loop
1. `config.py` — understand what all the settings mean
2. `algorithms/base.py` — `FLClient.local_train()` then `FLServer.apply_update()`
3. `algorithms/fedbuff.py` — `receive_update()` and `_aggregate_and_step()`
4. `utils/delay.py` — `staleness_scaling()` and `StalenessTracker`

### Read Second — understand the simulation
5. `simulator.py` — `_run_async()` (most important), then `_run_sync()`
6. `run_experiment.py` — `run_single()` and `main()`
7. `algorithms/fedavg.py` — to understand the synchronous baseline

### Read Later — fill in the details
8. `utils/metrics.py` — how results are measured and displayed
9. `datasets/cifar10.py` — how data is split non-IID across clients
10. `utils/models.py` — the neural network architectures
11. `utils/dp.py` — differential privacy (only if you care about privacy)
12. `datasets/leaf.py` — Sent140 and CelebA (only if using those datasets)

---

### Questions to Answer Once You Understand the Codebase

**Conceptual**
- What is the difference between FedBuff and FedAsync? (Hint: buffer size K)
- Why does staleness matter, and how does the code handle it?
- Why is "client trips" a better metric than wall-clock time for comparing algorithms?
- What is the "straggler problem" and how does async FL solve it?
- Why can FedBuff use SecAgg (Secure Aggregation) but FedAsync cannot?

**Code-level**
- What does `delta = initial_params[name] - p.data` represent, and why is it `initial - final` not `final - initial`?
- What happens in `_run_async()` when `len(active_clients) < cfg.concurrency`?
- What does `server_version` track, and where is it incremented?
- How does `StalenessTracker.compute_staleness()` work, and where is it called?
- What is the difference between `server_lr` and `client_lr`, and why does FedBuff need both?

**Experimental**
- Why does the paper test `K = 1, 10, 100`? What tradeoff does K control?
- What does `concurrency=1000` mean in the async setting vs. the sync setting?
- How would you add a new algorithm to this codebase? (Hint: subclass `FLServer`, implement `process_updates()`)
