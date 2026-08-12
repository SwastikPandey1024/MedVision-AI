# ADR-010: Single-Instance Distribution Strategy Reuse Across Model Factory and Training Engine

## Status
**ACCEPTED**

## Context
During multi-GPU execution on Kaggle (`2 × Tesla T4` GPUs), TensorFlow 2 / Keras 3 raised the following runtime error:
```text
RuntimeError: Mixing different tf.distribute.Strategy objects:
<MirroredStrategy object A> is not <MirroredStrategy object B>
```

### Root Cause
1. `scripts/train.py` called `get_distribution_strategy()` to instantiate `MirroredStrategy` object A.
2. `build_model()` inside `src/medvision/models/factory.py` independently called `get_distribution_strategy()` a second time, instantiating `MirroredStrategy` object B.
3. Attempting to scope tensor creation and model compilation under `Strategy B` scope while the execution script was under `Strategy A` scope resulted in a strategy object mismatch exception.

## Decision
1. **Single Strategy Instantiation**: `scripts/train.py` creates `strategy` **exactly once** at entrypoint via `strategy, gpu_count = get_distribution_strategy()`.
2. **Strategy Parameter Injection**: `build_model()` signature in `src/medvision/models/factory.py` accepts `strategy: Optional[tf.distribute.Strategy] = None`.
3. **Explicit Reuse**: If `strategy` is supplied, `active_strategy = strategy` is reused directly. Factory never creates a secondary `MirroredStrategy`.
4. **Validation assertion**: In smoke test mode, `strategy.num_replicas_in_sync == (gpu_count if gpu_count > 0 else 1)` and `Strategy object identity / reuse: PASS` are logged and asserted.

## Consequences
- ✅ Completely eliminates `RuntimeError: Mixing different tf.distribute.Strategy objects` on Kaggle multi-GPU kernels.
- ✅ Preserves CPU fallback (`_DefaultDistributionStrategy`) for fast local unit testing.
- ✅ Guarantees thread safety and GPU sync across training and evaluation loops.
