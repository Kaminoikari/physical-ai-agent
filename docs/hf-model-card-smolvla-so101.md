# HF Model Card 存底 — `Kaminoikari/smolvla-so101-pickplace-ft`

> 這份是已發佈到 Hugging Face 的 model card 存底，作為學習軌跡的一部分。
> 線上版：<https://huggingface.co/Kaminoikari/smolvla-so101-pickplace-ft>
> 訓練流程見 [`kaggle-finetune-smolvla-so101.md`](./kaggle-finetune-smolvla-so101.md)。

---

# SmolVLA · SO-101 Pick-and-Place (fine-tuned)

Fine-tune of [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) (450M VLA)
on the real-robot SO-101 dataset
[`lerobot/svla_so101_pickplace`](https://huggingface.co/datasets/lerobot/svla_so101_pickplace)
(50 teleoperated episodes, 2 cameras `up`/`side`, 6-DoF state).

## Training

| | |
|---|---|
| Base | `lerobot/smolvla_base` |
| Dataset | `lerobot/svla_so101_pickplace` (50 ep / 11,939 frames) |
| Steps | 2,000 (batch size 8) |
| GPU | single T4 (~78 min) |
| Camera mapping | `up→camera1`, `side→camera2` via `--rename_map` |
| Loss | 0.410 → 0.141 (monotonic) |

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --rename_map='{"observation.images.up":"observation.images.camera1","observation.images.side":"observation.images.camera2"}' \
  --batch_size=8 --steps=2000 --policy.device=cuda
```

## Intended use & honest limitations

This is a **pipeline-validation / learning run**, not a production policy.

- ✅ Demonstrates the full real-robot imitation-learning loop: load a real
  teleoperation dataset → fine-tune a pretrained VLA → converge → ship a checkpoint.
- ⚠️ Only 2,000 steps (~1.3 epochs). The SmolVLA paper uses ~20k steps; expect
  this checkpoint to under-perform a fully trained one.
- ⚠️ **No closed-loop success rate.** Evaluation on the physical SO-101 arm
  (`lerobot-record`) was not run — no hardware. Reported signal is training-loss
  convergence only, which proves the model is learning, not real-world task success.

## Load

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
policy = SmolVLAPolicy.from_pretrained("Kaminoikari/smolvla-so101-pickplace-ft")
```
