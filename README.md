# CIFAR-10 with 500 images: robustness to distribution shift

Train a small Inception-style CNN **from scratch** on only 400 CIFAR-10 images (plus 100 for validation) and test it on a shifted version of the CIFAR-10 test set (rotation, brightness/contrast, blur/noise, ~25% occlusion, and one hidden corruption).

## Results (mean ± std over 3 seeds: 0, 1, 2)

| Model | Val acc | Clean test | Shifted test |
|---|---|---|---|
| No augmentation (baseline) | 44.67 ± 4.11 | 45.41 ± 0.07 | 16.28 ± 0.90 |
| **Final model** | 45.67 ± 0.47 | 41.26 ± 1.93 | **23.03 ± 0.52** |

Chance is 10%. Augmentation gives +6.7 points on the shifted set but costs about 4 points on clean images. The biggest single improvement came from **geometric augmentation** (+2.5 points alone; +4.5 together with photometric augmentation).

Ablation (shifted test): no aug 16.28 → geometric 18.81 → photometric 15.72 → geometric + photometric 20.80 → + Cutout 21.64 → + Mixup/CutMix + label smoothing 21.20 → final (tuned weight decay) 23.03.

## Final configuration

- **Model:** conv stem, two Inception modules (1×1, 1×1→3×3, 1×1→5×5, pool→1×1), conv head, global average pooling, dropout 0.3, linear layer. 332,714 trainable parameters (limit: 1,000,000).
- **Training:** SGD + Nesterov (momentum 0.9), OneCycle LR (max 0.05), weight decay 0.005, batch 64, 60 epochs, best-validation checkpoint, label smoothing 0.1.
- **Augmentation (on the fly, training data only):** rotation ±25°, translation ±10%, scale 0.9–1.1, horizontal flip, brightness/contrast/saturation jitter, blur, noise, rare channel shuffle, Cutout, Mixup/CutMix (probability 0.6).
- **Hyper-parameters** (weight decay, dropout, mix probability) were chosen on the validation split only.

## Data and seeds

| Item | Value |
|---|---|
| train_small | 500 images, 50 per class, `RandomState(0)` |
| Split | 400 train / 100 validation (40 / 10 per class) |
| Training seeds reported | 0, 1, 2 |
| Hyper-parameter search seed | 100 |
| Shifted-test seed | 2024 |
| Shifted-validation seed | 777 |

The hidden corruption was not specified, so it is implemented as random grayscale or random channel swap (`hidden_corruption()` in the notebook). Other test-set details (rotation ±25°, 16×16 black box = 25% of the image) follow the assignment text. Blur and noise are mild and seeded.

## How to reproduce

**Option A: Google Colab (recommended, GPU)**
1. Open [colab.research.google.com](https://colab.research.google.com), then File → Upload notebook, and select `cifar10_robust_small_data.ipynb`.
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → **Run all**. CIFAR-10 downloads automatically. Expect roughly 20–40 minutes, plus the download time.

**Option B: local machine**
```bash
git clone <this-repo-url>
cd cifar10-robust-small-data
pip install -r requirements.txt
jupyter notebook cifar10_robust_small_data.ipynb     # then Run All
```
A CUDA GPU is strongly recommended. On CPU the full run will be very slow.

**Option C: plain script** (same code, no notebook)
```bash
python cifar10_robust.py       # writes results.txt and figures to ./figs/
```

Keep `SMOKE = False` (first code cell). `SMOKE = True` is only for a quick code check on fake data. All randomness is seeded, but GPU results can differ slightly between hardware and library versions (the reported run used torch 2.11 with CUDA 13).

## Files

| File | Content |
|---|---|
| `cifar10_robust_small_data.ipynb` | Completed, executed notebook (Parts A to F, all outputs) |
| `cifar10_robust.py` | Same code as a single script |
| `summary_one_page.docx` | One-page summary |
| `figures/` | Plots exported from the notebook |
| `requirements.txt` | Python packages |

## Notebook contents

- **A:** baseline without augmentation, learning curves, diagnosis (overfitting + covariate shift)
- **B:** augmentation design and answers (vertical flip, ±90° rotation, too much augmentation)
- **C:** regularisation and training design
- **D:** ablation table (3 seeds)
- **E:** per-class accuracy, confusion matrix, 12 misclassified images, hidden-corruption-only test
- **F (bonus):** test-time augmentation (no gain: 23.03 → 22.48 at 8 views, 8.3× slower)

## Notes

- Test sets are never used for training, validation or hyper-parameter choice. They are used once per run, after the best checkpoint is fixed, for reporting.
- Pretrained weights are not used. Augmented images are never saved to disk.
