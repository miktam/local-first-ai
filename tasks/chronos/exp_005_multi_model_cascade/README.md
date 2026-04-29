# Feature Brief: Multi-Model Cascade ("Dicer/Describer") Inference Pipeline

## 1. Overview & Objective
**Goal:** Enable rapid, fully local analysis of large, structured data payloads without triggering the hardware memory bandwidth bottleneck identified in recent telemetry testing.
**Approach:** Implement a deterministic Map-Reduce (Cascade) architecture using a lightweight Python orchestrator and a local inference engine. We will utilize a sub-10B parameter model to chunk and extract key data ("Dice"), and a >20B parameter model to synthesize the final insights ("Describe").

## 2. The Problem Statement (The Bandwidth Cliff)
Raw data exports (such as large JSON payloads, system logs, or dense structured reports) naturally exceed 75,000 tokens. Feeding this directly into a high-parameter model pushes the active KV cache well past the ~25,000-token safe zone. This triggers a super-quadratic prefill latency spike ($O(N^2)$), starving the compute engines of memory bandwidth and resulting in system hangs or severe performance degradation.

## 3. System Architecture & Environment
* **Target Hardware:** Edge Node / Apple Silicon Pro-tier (e.g., 20-core GPU, 64GB Unified Memory, ~273 GB/s bandwidth).
* **Inference Engine:** Local API server (e.g., Ollama, vLLM) exposing standard endpoints.
* **The "Dicer" Model:** Sub-10B parameter model (Fast, highly compute-bound, low memory footprint).
* **The "Describer" Model:** >20B parameter reasoning model (High-tier logic, larger memory footprint).
* **Memory Management Strategy:** Both models are locked concurrently in VRAM. The combined active footprint fits comfortably within the allocated GPU unified memory pool, maintaining a strict zero-SSD-swap execution state.

## 4. Implementation Milestones

### Phase 1: Data Sanitization (Pre-computation)
* Write a lightweight script to parse the raw data structure and strip unnecessary metadata (e.g., coordinate mappings, UI states, raw timestamp hashes, empty arrays).
* Isolate only the high-value text nodes.
* *Expected outcome:* Substantial (70-80%) reduction in raw token count before inference begins.

### Phase 2: The Dicer Layer (Map)
* Segment the sanitized payload into logical, discrete chunks (e.g., max 4,000-token batches).
* Route chunks sequentially or concurrently to the fast sub-10B model.
* **Prompt objective:** Extract critical entities, summarize status states, and compress the chunk's core information into a dense format.

### Phase 3: The Describer Layer (Reduce)
* Concatenate the compressed outputs from the Dicer layer into a single summary document.
* Route this aggregated document to the high-parameter reasoning model.
* **Prompt objective:** Provide high-level trajectory analysis, identify cross-chunk correlations, and generate final strategic insights.

### Phase 4: Telemetry & Validation
* Wrap the orchestrator in the existing test harness.
* Log hardware wattage, ms/token prefill latency, and total pipeline execution time to formally validate that the system remains in a strictly compute-bound regime.

## 5. Success Metrics
* **Zero Swapping:** Telemetry confirms 0 bytes of SSD swap used during the entire pipeline execution.
* **Latency Ceilings:** The heavy reasoning model maintains a prefill rate well below the previously established "cliff" threshold (e.g., < 13.5 ms/token).
* **Data Sovereignty:** 100% of the data ingestion, chunking, and generation executes on bare metal without external API calls.
