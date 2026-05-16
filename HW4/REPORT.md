# HW4 All-in-One Image Restoration — 完整技術報告

> **任務**：單一模型同時還原 Rain / Snow 兩種退化  
> **模型**：PromptIR 拓樸 + NAFBlock 作為 encoder / decoder 內部 block  
> **框架**：PyTorch 2.1 + AMP + EMA + W&B logging  
> **執行腳本**：`best_run.sh`

---

## 目錄

1. [整體流程概覽](#1-整體流程概覽)
2. [資料前處理](#2-資料前處理)
3. [訓練資料增強 (Augmentation)](#3-訓練資料增強-augmentation)
4. [模型架構](#4-模型架構)
   - 4.1 [整體 U-Net 拓樸](#41-整體-u-net-拓樸)
   - 4.2 [OverlapPatchEmbed](#42-overlappatchembed)
   - 4.3 [NAFBlock（內部 encoder/decoder block）](#43-nafblock內部-encoderdecoder-block)
   - 4.4 [Downsample / Upsample](#44-downsample--upsample)
   - 4.5 [PromptGenBlock（PromptIR 核心貢獻）](#45-promptgenblockpromptir-核心貢獻)
   - 4.6 [Prompt 注入與通道變換](#46-prompt-注入與通道變換)
   - 4.7 [Refinement 與輸出](#47-refinement-與輸出)
5. [訓練流程](#5-訓練流程)
6. [Inference 流程](#6-inference-流程)
7. [超參數總覽](#7-超參數總覽)
8. [附錄：資料流維度追蹤](#8-附錄資料流維度追蹤)

---

## 1. 整體流程概覽

```
原始資料
  HW4/dataset/train/degraded/{rain,snow}-{1..1600}.png
  HW4/dataset/train/clean/   {rain,snow}_clean-{1..1600}.png
  HW4/dataset/test/degraded/ {0..99}.png   (均為 256×256 RGB)
        │
        ▼
src/dataset.py : HW4TrainDataset
  ├── 掃描 degraded/ 配對 clean/，分別計入 rain (de_id=0) / snow (de_id=1)
  ├── 以 seed 固定的 per-type shuffle 切出 val_ratio (default 5%) 作驗證集
  └── 訓練 sample：random crop + flip + rot90 (+ 可選 RGB shuffle / MixUp)
        │
        ▼  (× epochs)
train.py
  ├── PromptIRNAF（dim=48, blocks=[4,6,6,8], refine=4, ~22.85M params）
  ├── AdamW + 線性 warmup 15ep + cosine 衰減 200ep
  ├── Charbonnier 主損失（可切換 l1 / l2 / psnr，可疊加 SSIM / FFT）
  ├── AMP fp16 + GradScaler + grad-clip(1.0)
  ├── EMA (decay 0.999) 同步維護影子權重
  ├── 每個 epoch 做 val PSNR（all / rain / snow）
  └── 儲存 last.pt / best.pt / ema_best.pt + args.json
        │
        ▼
inference.py
  ├── 從 ckpt 讀取 args.json 重建相同架構
  ├── 預設使用 ema_best 權重
  ├── 對每張 test PNG 做 8-way TTA self-ensemble
  ├── 可選 sliding-window patch inference（Gaussian 加權拼接）
  └── 輸出 pred.npz：{"0.png": (3,H,W) uint8, ..., "99.png": ...}
```

---

## 2. 資料前處理

### 2.1 輸入格式

```
HW4/dataset/
├── train/
│   ├── degraded/
│   │   ├── rain-1.png      … rain-1600.png      (1600 張)
│   │   └── snow-1.png      … snow-1600.png      (1600 張)
│   └── clean/
│       ├── rain_clean-1.png … rain_clean-1600.png
│       └── snow_clean-1.png … snow_clean-1600.png
└── test/
    └── degraded/
        └── 0.png … 99.png                       (100 張，皆 256×256)
```

每張 train 圖均為 256×256 RGB；test 經助教 example 解壓驗證亦均為 256×256。

### 2.2 影像讀取（`src/dataset.py`）

```python
deg   = np.array(Image.open(deg_path).convert("RGB"))      # uint8 HWC
clean = np.array(Image.open(clean_path).convert("RGB"))    # uint8 HWC
…
_to_tensor(arr):    # HWC uint8 → CHW float32 in [0, 1]
    return torch.from_numpy(
        np.ascontiguousarray(arr.astype(np.float32) / 255.0)
                            .transpose(2, 0, 1))
```

- 一律 `convert("RGB")` 避免 RGBA / Palette 進入網路
- `np.ascontiguousarray` 確保翻轉 / 旋轉後仍是 contiguous（PyTorch 才能直接 `from_numpy`）
- 數值 normalize 到 `[0, 1]`，與 NAFNet / PromptIR 慣例一致

### 2.3 配對與 Train/Val 切分

```python
# _pair_paths：掃描 degraded/，根據前綴 rain-/snow- 推算 clean 檔名
for p in glob("train/degraded/*.png"):
    name = basename(p)
    if name.startswith("rain-"):
        clean = "train/clean/" + name.replace("rain-", "rain_clean-")
        de_id = 0
    elif name.startswith("snow-"):
        clean = "train/clean/" + name.replace("snow-", "snow_clean-")
        de_id = 1

# _split_pairs：per-type seeded shuffle → 取 val_ratio
rng = random.Random(seed)
for de_id, lst in by_type.items():
    rng.shuffle(lst)
    n_val = round(len(lst) * val_ratio)
    val   += lst[:n_val]
    train += lst[n_val:]
```

- **per-type 切分**：rain 與 snow 各自獨立切，避免某一類全部落在 train 或 val
- **deterministic**：`random.Random(seed)`（不污染全域 RNG）
- 預設 `val_ratio=0.05` → 每類 80 張 val，共 160 張；train 共 3040 張

---

## 3. 訓練資料增強 (Augmentation)

訓練 pipeline（`HW4TrainDataset.__getitem__`）依序執行：

```
原圖 (256×256 BGR uint8 numpy)
      │
      ▼  1. _crop()
         random crop to patch_size×patch_size (default 128)
         若 patch_size > 原圖：先 reflect pad
         val 時改為 center crop（patch_size=0 表全圖）

      │
      ▼  2. flip H (50%)  ← --aug_flip
      │     deg, clean = deg[:, ::-1], clean[:, ::-1]

      ▼  3. flip V (50%)  ← --aug_flip
      │     deg, clean = deg[::-1, :], clean[::-1, :]

      ▼  4. rot90 × k, k∈{0,1,2,3}  ← --aug_rot90
      │     等價於 dihedral D₄ 群除翻轉外的其他元素，總計 8 種等價方位

      ▼  5. RGB channel shuffle (50%, 可選)  ← --aug_rgb_shuffle
      │     deg / clean 同步以同一 permutation 重排 RGB
      │     讓模型對通道分佈更穩健

      ▼  6. ToTensor → float32 [0,1] CHW

      ▼  7. MixUp（可選，機率 --aug_mixup_p）
            從同一退化類型隨機抽另一張，做 λ × A + (1−λ) × B
            λ ~ Beta(0.2, 0.2)
            degraded / clean 用同一 λ → 像素級線性混合
```

| 操作 | 預設 | 目的 |
|------|------|------|
| RandomCrop (128) | ✓ | 縮小 GPU 記憶體佔用 + 增加位置多樣性 |
| Flip H + Flip V | ✓ | rain streaks 多為斜向，水平翻轉能涵蓋兩個方向 |
| Rot90 × {0,1,2,3} | ✓ | 雪花為近似各向同性，4 個方向都合理 |
| RGB Channel Shuffle | ✗ | 對顏色不敏感的還原任務有時能加強泛化 |
| MixUp | ✗ | 退化還原任務的 MixUp 較少見，列為實驗開關 |

驗證 pipeline 無隨機操作：全圖（256×256） → `_to_tensor`。

---

## 4. 模型架構

### 4.1 整體 U-Net 拓樸

```
Input: (B, 3, H, W)  RGB float in [0, 1]
                            ↓
┌────────────────────────────────────────────────────────────┐
│  4.2  OverlapPatchEmbed    3×3 conv  →  c1=48ch            │
└────────────────────────────────────────────────────────────┘
                            ↓
        ┌─── 4 × NAFBlock (encoder_level1, c1=48) ─────────┐
        ↓                                                   │
   Downsample (PixelUnshuffle, ×½)                          │ skip
        ↓                                                   │
        ┌─── 6 × NAFBlock (encoder_level2, c2=96) ─────────┤
        ↓                                                   │
   Downsample (×¼)                                          │ skip
        ↓                                                   │
        ┌─── 6 × NAFBlock (encoder_level3, c3=192) ────────┤
        ↓                                                   │
   Downsample (×⅛)                                          │ skip
        ↓                                                   │
        ┌─── 8 × NAFBlock (latent,        c4=384) ─────────┤
        ↓                                                   │
   ── PromptGenBlock 3  →  concat pd3=320  →  704ch ──      │
   ── NAFBlock(704)  →  1×1 Conv  →  c3=192       ──        │
        ↓                                                   │
   Upsample (PixelShuffle, ×2)  →  c2=96  ←─── skip enc3 ───┘
        ↓     concat (96+192=288)  →  1×1 Conv  →  c3=192
        ┌─── 6 × NAFBlock (decoder_level3, c3=192) ────────┐
        ↓                                                   │
   ── PromptGenBlock 2  →  concat pd2=128  →  320ch ──      │ skip
   ── NAFBlock(320)  →  1×1 Conv  →  c3=192       ──        │
        ↓                                                   │
   Upsample (×½)  →  c2=96  ←─── skip enc2 ─────────────────┘
        ↓     concat (96+96=192)  →  1×1 Conv  →  c2=96
        ┌─── 6 × NAFBlock (decoder_level2, c2=96) ─────────┐
        ↓                                                   │
   ── PromptGenBlock 1  →  concat pd1=64  →  160ch ──       │ skip
   ── NAFBlock(160)  →  1×1 Conv  →  c2=96        ──        │
        ↓                                                   │
   Upsample (×½)  →  c1=48  ←─── skip enc1 ─────────────────┘
        ↓     concat (48+48=96)  =  c2
        ┌─── 4 × NAFBlock (decoder_level1, c2=96) ─────────┐
        ↓                                                   │
        ┌─── 4 × NAFBlock (refinement,     c2=96) ─────────┘
                            ↓
                3×3 Conv → 3ch
                            ↓
       Output = restored + input_image  (residual)
```

**主要設計選擇**：

- 沿用 PromptIR 四層 U-Net 拓樸、3 個 PromptGenBlock 注入點與最終 residual
- **將 PromptIR 原本的 TransformerBlock（MDTA + GDFN）全部替換為 NAFBlock**
  → 沒有 self-attention、沒有 GELU；只用 SimpleGate 和 SCA 提供非線性與 channel 互動
  → 在相同寬度下 FLOPs 較低，可在單張 4090 上跑到 `dim=48` + `[4,6,6,8]` blocks
- 通道寬度由 `dim` 和 `prompt_dims` **動態計算**（不像原 PromptIR 把 192/512/224/64 寫死）
  → 任意 `dim` 都能正常前向 (`promptir_naf.py` 內均寫成 `c1, c2, c3, c4` 表達式)
- 預設模型 `~22.85M` 參數（dim=48），輕量版 `dim=32, [2,3,3,4]` 約 `10.3M` 用於 smoke test

---

### 4.2 OverlapPatchEmbed

```python
self.patch_embed = nn.Conv2d(3, dim=48, kernel_size=3, stride=1, padding=1, bias=False)
```

- 不做下採樣，僅將 RGB（3ch）投影到 `dim` 維特徵空間
- 採用 3×3 而非 1×1，能在 patch 邊界保留鄰域資訊

---

### 4.3 NAFBlock（內部 encoder/decoder block）

NAFBlock 是 NAFNet (ECCV 2022) 的核心 building block，**特色為：無 self-attention、無 ReLU / GELU、僅用 LayerNorm2d + 1×1 / 3×3 conv + SimpleGate + SCA**。整個 block 由兩個帶 β / γ 縮放的 residual 組成：

```
inp  ─┬─────────────────────────────────────────────┐  (residual 1)
      │                                              │
      ▼                                              │
  LayerNorm2d(c)                                     │
      ▼                                              │
  Conv2d(c → 2c, 1×1)            ← conv1 (expand)   │
      ▼                                              │
  DepthwiseConv2d(2c → 2c, 3×3)  ← conv2 (DW conv)  │
      ▼                                              │
  SimpleGate:  chunk(2) → x1 * x2  →  c channels    │
      ▼                                              │
  SCA: AdaptiveAvgPool(1) → Conv1×1(c → c) → x*sca  │
      ▼                                              │
  Conv2d(c → c, 1×1)             ← conv3 (project) │
      ▼                                              │
  Dropout (default 0)                                │
      ▼                                              │
  × β  (learnable, init 0)                           │
      └──── y = inp + x*β ─────────────────────────►┘

y ─┬───────────────────────────────────────────────┐  (residual 2)
   ▼                                                │
LayerNorm2d(c)                                      │
   ▼                                                │
Conv2d(c → 2c, 1×1)            ← conv4 (expand)    │
   ▼                                                │
SimpleGate:  chunk(2) → x1 * x2  →  c channels     │
   ▼                                                │
Conv2d(c → c, 1×1)             ← conv5 (project)   │
   ▼                                                │
Dropout                                             │
   ▼                                                │
× γ  (learnable, init 0)                            │
   └──── out = y + x*γ ──────────────────────────►┘
```

**子模組細節**：

#### (a) `LayerNorm2d`（自定義 autograd）

對 `(N, C, H, W)` 沿 **channel 維**做 normalize（每個 pixel 對其自己的通道做正規化），帶 learnable affine `weight (C,)`, `bias (C,)`。  
為什麼用自定義 `torch.autograd.Function`：直接寫成 `permute + LayerNorm + permute` 在大 batch + 大解析度下 backward 較慢；NAFNet 釋出的自定義版本約快 1.3×。

#### (b) `SimpleGate`

```python
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
```

把通道一半當作 gate，**完全沒有 ReLU / GELU 等飽和函數**。NAFNet paper 顯示其效果優於 GELU + GLU 變體，同時計算成本更低。  
→ 因為 SimpleGate 會把 channel 數對半，所以網路中所有寬度都需是偶數（PromptIR-NAF 中所有寬度 48/96/160/192/288/320/384/704 都滿足）。

#### (c) Simplified Channel Attention (SCA)

```python
self.sca = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Conv2d(c // 2, c // 2, 1),      # 不再用 SE 的 fc-relu-fc bottleneck
)
x = x * self.sca(x)
```

把傳統 SE 的兩層 fc + ReLU 簡化為單一 1×1 conv —— 仍保留「以 global statistics 重新縮放每個通道」的能力，但參數和算力更省。

#### (d) β / γ 縮放 residual

```python
self.beta  = nn.Parameter(torch.zeros((1, c, 1, 1)))   # init = 0
self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))
…
y = inp + x * self.beta
out = y + x * self.gamma
```

兩個 residual 各乘一個 **per-channel 可學參數，初始化為 0**。  
→ 訓練開始時 block 等價於 identity，整個網路從 input 直接到 output（殘差 0），  
→ 隨訓練 β/γ 逐漸學到非零值，每個 channel 自己決定要不要打開該 residual。  
→ 相當於 Stochastic Depth 的 soft 版本，能穩定深層 stack 的訓練（PromptIR 共 ~50 個 NAFBlock）。

---

### 4.4 Downsample / Upsample

兩者皆採用 **PixelShuffle 家族**，比 strided conv 對齊像素更精確、無棋盤格 artifact：

```python
# Downsample(n)  輸入 (B, n, H, W) → (B, 2n, H/2, W/2)
nn.Sequential(
    nn.Conv2d(n, n // 2, 3, 1, 1, bias=False),
    nn.PixelUnshuffle(2),         # (B, n/2, H, W) → (B, 2n, H/2, W/2)
)

# Upsample(n)    輸入 (B, n, H, W) → (B, n/2, 2H, 2W)
nn.Sequential(
    nn.Conv2d(n, n * 2, 3, 1, 1, bias=False),
    nn.PixelShuffle(2),           # (B, 2n, H, W) → (B, n/2, 2H, 2W)
)
```

→ Down 把 ch 翻倍、解析度減半；Up 把 ch 減半、解析度翻倍 —— 完整對稱，方便 skip connection。

---

### 4.5 PromptGenBlock（PromptIR 核心貢獻）

PromptIR 的關鍵設計是**讓網路在不知道退化類別的情況下，自動從輸入特徵推斷出 task-specific 的「prompt」並注入 decoder**。

```python
class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim, prompt_len, prompt_size, lin_dim):
        # prompt_param: (1, prompt_len, prompt_dim, prompt_size, prompt_size)
        #   = 5 個 learnable prompt embedding，每個都是 prompt_dim × ps × ps 的小特徵圖
        self.prompt_param = nn.Parameter(
            torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3      = nn.Conv2d(prompt_dim, prompt_dim, 3, padding=1, bias=False)

    def forward(self, x):                              # x: (B, lin_dim, H, W)
        emb = x.mean(dim=(-2, -1))                     # (B, lin_dim)     global avg
        w   = F.softmax(self.linear_layer(emb), -1)    # (B, prompt_len)  soft 選擇權重
        # 把 5 個 prompt 用 w 做加權平均
        prompt = (w[:, :, None, None, None] *
                  self.prompt_param.expand(B, -1, -1, -1, -1)).sum(1)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear")  # 拉到 feature 同尺寸
        prompt = self.conv3x3(prompt)                            # 再 refine
        return prompt                                  # (B, prompt_dim, H, W)
```

**設計直覺**：
- 5 個 prompt 像「字典條目」，網路可以依據輸入特徵動態混合（soft attention）
- 對 rain 輸入會學到一組權重，對 snow 又是另一組；連續退化空間中 prompt 也能線性插值
- 因為混合是可微的，整個 PromptGenBlock 完全 end-to-end 一起學

**三組 PromptGenBlock 參數**（與 PromptIR 原設定一致）：

| 位置 | `prompt_dim` | `prompt_len` | `prompt_size` | `lin_dim`（feature 寬度） |
|------|------|------|------|------|
| `prompt3`（latent，1/16 解析度） | 320 | 5 | 16 | 384 (= c4) |
| `prompt2`（decoder_level3，1/8） | 128 | 5 | 32 | 192 (= c3) |
| `prompt1`（decoder_level2，1/4） | 64  | 5 | 64 | 96  (= c2) |

→ 越靠近輸出，prompt_dim 越小、prompt_size 越大（更貼近原始解析度）。

---

### 4.6 Prompt 注入與通道變換

每個 prompt 都會與當層 feature **沿 channel 維 concat**，再經一個 NAFBlock 與 1×1 conv 壓回原寬度：

```python
# 以 prompt3 為例（latent 寬度 c4=384, prompt_dim=320 → concat=704）
dec3 = self.prompt3(latent)                              # (B, 320, 16, 16)
latent = torch.cat([latent, dec3], dim=1)                # (B, 704, 16, 16)
latent = self.noise_level3(latent)                       # NAFBlock(704)
latent = self.reduce_noise_level3(latent)                # 1×1 conv: 704 → 192 (= c3)
```

三組注入點對應的寬度：

| 注入點 | feature in | + prompt | 注入後 NAFBlock 寬度 | reduce 至 |
|--------|-----------|----------|---------------------|----------|
| latent (prompt3) | 384 | +320 | **704** | 192 |
| decoder_level3 (prompt2) | 192 | +128 | **320** | 192 |
| decoder_level2 (prompt1) | 96 | +64 | **160** | 96 |

所有 concat 後的寬度均為偶數 → SimpleGate 不會出錯。`reduce_noise_level3` 把 latent 從 c4 壓到 c3，方便緊接的 `up4_3 = Upsample(c3)`；其餘兩個 reduce 在 prompt fusion 後 channel 不變（c3→c3 / c2→c2）。

> 註：若 `--decoder_prompt=false`（ablation），程式會以一個 `latent_reduce = Conv2d(c4, c3, 1)` 取代 prompt3 路徑，確保 `up4_3` 仍能正確接 c3 輸入。

---

### 4.7 Refinement 與輸出

```python
# decoder_level1 之後（96ch）：
self.refinement = nn.Sequential(*[NAFBlock(96) for _ in range(num_refinement_blocks)])  # default 4
self.output     = nn.Conv2d(96, 3, kernel_size=3, padding=1, bias=False)

# Forward 末段
out = self.output(self.refinement(out_dec_level1)) + inp_img    # residual prediction
```

- 多疊一段 NAFBlock 做 detail refine（不接 skip / prompt）
- 最後 3×3 conv → 3ch
- **整個網路預測的是 residual**：`output = ConvHead(features) + degraded_input`  
  → 訓練早期 residual 近 0，模型自動退化為恆等映射 + 微小修正  
  → 對 PSNR 任務這個 residual learning 是標準做法

---

## 5. 訓練流程

### 5.1 Optimizer：AdamW

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-4,
    betas=(0.9, 0.9),       # 二階動量也用 0.9，與 NAFNet recipe 對齊
)
```

- **betas=(0.9, 0.9)**（非預設 (0.9, 0.999)）：來自 NAFNet 對 deblurring 任務的調校結論，二階動量視窗較短能讓 loss 對最近 batch 更敏感
- **weight_decay=1e-4**：弱正則化，避免 NAFBlock 的 β/γ 被過早壓回 0

### 5.2 Learning Rate Schedule（`src/schedulers.py`）

```
Epoch 0 → 15   Linear warmup
                LR: 0 → 2e-4 （線性）
Epoch 15 → 200 Cosine annealing
                LR: 2e-4 → 0.0  （半週期 cosine）
```

實作於 `LinearWarmupCosineAnnealingLR`：

```python
if e < warmup_epochs:
    lr = base_lr * e / warmup_epochs
else:
    progress = (e - warmup_epochs) / (max_epochs - warmup_epochs)
    lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(π * progress))
```

→ 15 個 epoch 緩升、185 個 epoch 餘弦下降；warmup 對 EMA 初期穩定特別重要。

### 5.3 Loss：Charbonnier + 可選輔助項

主損失預設使用 Charbonnier loss（L1 的 smooth 版）：

```python
class CharbonnierLoss(nn.Module):
    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))   # eps=1e-3
```

- `eps=1e-3` 在小誤差時近似 L2、大誤差時近似 L1，兼具梯度平滑與離群點魯棒
- 可切換為 `--loss {l1,l2,charbonnier,psnr}`；其中 `psnr` 是 NAFNet 的 `-10·log10(MSE)` 版本

**可疊加的輔助損失**（預設權重 0）：

| 名稱 | 公式 | 何時打開 |
|------|------|---------|
| SSIM | `1 - SSIM_window11(pred, target)` | `--ssim_weight 0.1`：對結構性失真敏感 |
| FFT  | `‖FFT(pred) − FFT(target)‖₁`       | `--fft_weight 0.05`：強化高頻細節 |

整體 loss 由 `CompositeLoss` 加權彙整：

```python
total = primary(pred, clean)
      + ssim_weight * (1 - SSIM(pred, clean))
      + fft_weight  * ‖FFT(pred) − FFT(clean)‖₁
```

### 5.4 Gradient Clipping

```python
scaler.unscale_(optimizer)
total_norm = nn.utils.clip_grad_norm_(model.parameters(),
                                      max_norm=args.grad_clip or 1e9)
```

- 預設 `grad_clip=1.0`（L2 norm）
- `clip_grad_norm_` 回傳 **裁剪前** 的總範數，所以即便 `grad_clip=0`（用 1e9 不裁）也能拿到當前梯度大小 → 寫入 `train/grad_norm` 監控訓練穩定性

### 5.5 Mixed Precision Training (AMP)

```python
scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

with torch.cuda.amp.autocast(enabled=args.amp):
    pred = model(deg)
    loss = loss_fn(pred, clean)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
nn.utils.clip_grad_norm_(...)
scaler.step(optimizer)
scaler.update()
```

- 前向 / backward fp16；參數更新 fp32
- GradScaler 動態調整 loss scale，避免 fp16 underflow
- NAFBlock 沒有 GRN 之類數值敏感操作，AMP 對它幾乎沒有 NaN 風險

### 5.6 EMA（Exponential Moving Average）

```python
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    def update(self, model):
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(decay).add_(model.state_dict()[k].detach(), alpha=1-decay)
```

- 每個 optimizer step 都呼叫 `ema.update(model)`
- Validation 與最終 inference 都使用 EMA 權重（`--ckpt_kind ema`，預設）
- 在影像還原任務經驗上 EMA 通常能多帶來 +0.1 ~ +0.3 dB

### 5.7 Validation

```python
@torch.no_grad()
def validate(eval_model, val_loader, device):
    psnr_by_type = {0: [], 1: []}
    for deg, clean, de_id in val_loader:
        pred = eval_model(deg).clamp(0, 1)
        psnr = -10 * log10(MSE(pred, clean))      # via psnr_torch()
        psnr_by_type[de_id].append(psnr)
    return {
        "psnr_all":  mean(all),
        "psnr_rain": mean(psnr_by_type[0]),
        "psnr_snow": mean(psnr_by_type[1]),
    }
```

- 全圖（256×256）直接 forward，不做 TTA、不切 tile → 同時是真實使用情境
- 拆 rain / snow 報告 → 能立刻看出哪個 task 拖後腿

### 5.8 Checkpoint 與 W&B logging

每個 epoch：
- 始終會寫 `last.pt`（每 `save_every` epoch）；
- 若 `psnr_all > best_psnr` 則寫 `best.pt`（同時 EMA 寫 `ema_best.pt`）；
- W&B 記錄：

| 群組 | metric |
|------|--------|
| `train/` | `loss`, `lr`, `grad_norm`, `time_s`, `samples_per_s`, `gpu_mem_gb` |
| `val/`   | `psnr_all`, `psnr_rain`, `psnr_snow`, `best_psnr` |
| samples  | 固定 4 張 val triplet `(degraded \| pred \| clean)`，每 `wandb_image_every` epoch 或 PSNR 創新高時更新 |
| summary  | `params_M`, `train_pairs`, `val_pairs`, `best_psnr`, `final_best_psnr` |

### 5.9 訓練配置總結

| 設定 | 值 |
|------|----|
| epochs | 200 |
| batch_size | 8 |
| patch_size | 128 |
| optimizer | AdamW |
| lr (base) | 2e-4 |
| betas | (0.9, 0.9) |
| weight_decay | 1e-4 |
| warmup_epochs | 15 |
| LR schedule | linear warmup → cosine to 0 |
| grad_clip | 1.0 |
| amp | True |
| ema_decay | 0.999 |
| loss | Charbonnier (eps=1e-3) |
| val_ratio | 0.05 (per type) |
| save_every | 5 epochs |
| val_every | 1 epoch |
| seed | 42 |
| num_workers | 4 |

---

## 6. Inference 流程

### 6.1 單張推論

```
test/degraded/{i}.png   (256×256 RGB)
      │
      ▼  HW4TestDataset.__getitem__
         Image.open → np uint8 HWC
         _to_tensor → CHW float32 [0, 1]
      │
      ▼  to device (cuda), (1, 3, 256, 256)
      │
      ▼  restore_image(model, x, tile=0, overlap=32, use_tta=True)
         ├── pad_to_multiple(16)  → (1, 3, 256, 256)   (256 已被 16 整除，no-op)
         ├── 若 tile>0  → sliding window 拼接（見 6.3）
         │   否則     → 直接 forward
         └── 若 use_tta → 8-way TTA（見 6.2）
      │
      ▼  clamp [0, 1] → ×255 → round → uint8 → (3, H, W)
      │
      ▼  寫入 out_dict[basename(p)] = arr
      │
      ▼  np.savez(args.output, **out_dict)
```

最終 `pred.npz`：100 個 key（`"0.png" ... "99.png"`），每個 value 為 `uint8 (3, 256, 256)`，與助教 `example/example_submmision.zip` 結構完全一致。

### 6.2 Test-Time Augmentation (8-way self-ensemble)

對應 `src/utils.py:tta_forward`：

```
原圖 x
  ├── identity                 → model → ŷ₀
  ├── flip H                   → model → ŷ₁ → flip H
  ├── flip V                   → model → ŷ₂ → flip V
  ├── flip H + flip V          → model → ŷ₃ → flip V → flip H
  ├── rot90                    → model → ŷ₄ → rot−90
  ├── rot90 + flip H           → model → ŷ₅ → flip H → rot−90
  ├── rot90 + flip V           → model → ŷ₆ → flip V → rot−90
  └── rot90 + flip H + flip V  → model → ŷ₇ → flip V → flip H → rot−90
        ↓
   mean({ŷ₀ … ŷ₇}) → output
```

→ 覆蓋 dihedral D₄ 的全部 8 個對稱方位；對隨機 augmentation 訓練過的模型平均能再 +0.1~0.3 dB。  
→ Forward 次數 ×8，但 256×256 上單張只要 ~50ms ×8 = 400ms / 圖，仍很快。

### 6.3 Sliding Window（可選，預設關閉）

由 `--tile T --overlap O` 啟用。流程：

```
input (H, W)
  ├── 計算切窗位置 ys = 0, T-O, 2(T-O), ... 並補上 H-T
  ├── 對每個 (y, x) 取 patch[y:y+T, x:x+T]
  │     → model → pred patch
  │     → 加上 Gaussian 權重 win=(T,T) （中央高、邊緣低）
  │     → 累加到 out, weight
  └── out / weight  → 最終結果
```

→ 對於遠大於訓練 patch_size 的 test image 有用；本題 test = 256 與訓練 crop = 128 同尺度，**預設 `--tile 0`**（全圖 forward）。

### 6.4 載入 checkpoint 細節（`inference.py:load_model`）

```python
blob = torch.load(args.ckpt)
cfg  = blob["args"]            # 訓練時存在 ckpt 中的 argparse 字典
                               # 也可手動以 --config_json 覆蓋
model = PromptIRNAF(
    dim=cfg["dim"],
    num_blocks=tuple(cfg["num_blocks"]),
    num_refinement_blocks=cfg["num_refinement_blocks"],
    dw_expand=cfg["dw_expand"], ffn_expand=cfg["ffn_expand"],
    drop_path=0.0,             # 推論時關 dropout
    decoder=cfg["decoder_prompt"],
    prompt_dims=tuple(cfg["prompt_dims"]),
    prompt_len=cfg["prompt_len"],
    prompt_sizes=tuple(cfg["prompt_sizes"]),
)
sd = blob.get("ema", blob["model"])   # --ckpt_kind ema (default) / raw
model.load_state_dict(sd)
```

→ 訓練什麼架構，推論就是什麼架構，**完全不依賴 argparse 預設值**。

---

## 7. 超參數總覽

### 7.1 模型超參數

| 超參數 | 預設 | 意義 |
|--------|------|------|
| `--dim` | 48 | Stage-1 通道寬度；後續為 96/192/384 |
| `--num_blocks` | 4 6 6 8 | encoder 各層 NAFBlock 數量 |
| `--num_refinement_blocks` | 4 | 最後 refinement 段 NAFBlock 數量 |
| `--dw_expand` | 2 | NAFBlock 內 1×1 expand 倍率（影響中間通道） |
| `--ffn_expand` | 2 | NAFBlock 內 FFN 段 expand 倍率 |
| `--drop_path` | 0.0 | NAFBlock 內 dropout（0 = 關閉） |
| `--decoder_prompt` | True | 啟用 3 個 PromptGenBlock 注入 |
| `--prompt_dims` | 64 128 320 | 三個 prompt 的 channel 數 |
| `--prompt_len` | 5 | 每個 PromptGenBlock 的字典大小 |
| `--prompt_sizes` | 64 32 16 | 三個 prompt 的空間解析度 |

### 7.2 訓練超參數

| 超參數 | 預設 | 意義 |
|--------|------|------|
| `--epochs` | 200 | 總訓練輪數 |
| `--batch_size` | 8 | 受 GPU VRAM 限制；4090 24GB 可上 8 |
| `--patch_size` | 128 | 訓練時 random crop 大小 |
| `--lr` | 2e-4 | AdamW base learning rate |
| `--wd` | 1e-4 | weight decay |
| `--beta1`, `--beta2` | 0.9, 0.9 | AdamW 動量（兩者皆 0.9，NAFNet recipe） |
| `--warmup_epochs` | 15 | 線性 warmup |
| `--grad_clip` | 1.0 | L2 gradient clipping |
| `--amp` | True | FP16 訓練 |
| `--seed` | 42 | 全域 random seed |

### 7.3 Loss 超參數

| 超參數 | 預設 | 意義 |
|--------|------|------|
| `--loss` | charbonnier | 主損失：`l1 / l2 / charbonnier / psnr` |
| `--charbonnier_eps` | 1e-3 | Charbonnier 平滑常數 |
| `--psnr_toY` | False | PSNRLoss 是否先轉到 Y 通道 |
| `--ssim_weight` | 0.0 | `(1 - SSIM)` 輔助損失權重 |
| `--fft_weight` | 0.0 | FFT L1 輔助損失權重 |

### 7.4 Augmentation 超參數

| 超參數 | 預設 | 意義 |
|--------|------|------|
| `--aug_flip` | True | random horizontal / vertical flip |
| `--aug_rot90` | True | random rotation k×90° |
| `--aug_rgb_shuffle` | False | 同步隨機 RGB 通道重排 |
| `--aug_mixup_p` | 0.0 | MixUp 機率（Beta(0.2,0.2) λ） |
| `--val_ratio` | 0.05 | per-type validation hold-out 比例 |

### 7.5 EMA 超參數

| 超參數 | 預設 | 意義 |
|--------|------|------|
| `--use_ema` | True | 是否維護 EMA 權重 |
| `--ema_decay` | 0.999 | EMA 衰減率 |

### 7.6 Inference 超參數

| 超參數 | 預設 | 意義 |
|--------|------|------|
| `--ckpt` | — | 必填；訓練輸出的 `*.pt` |
| `--ckpt_kind` | auto | `auto`（優先 EMA）/ `ema` / `raw` |
| `--tta` | True | 8-way self-ensemble |
| `--tile` | 0 | 0 = 全圖 forward；>0 啟用 sliding window |
| `--overlap` | 32 | sliding window 重疊像素 |
| `--batch_size` | 1 | 推論 batch size |

### 7.7 W&B logging 超參數

| 超參數 | 預設 | 意義 |
|--------|------|------|
| `--wandb_project` | "" | 空字串 = 完全停用 W&B |
| `--wandb_entity` | "" | team / user 名 |
| `--run_name` | "" | 預設用 `basename(ckpt_dir)` |
| `--wandb_tags` | [] | 多個 tag |
| `--wandb_log_images` | True | 是否記錄 val triplet 影像 |
| `--wandb_image_count` | 4 | 取多少張 val sample 視覺化 |
| `--wandb_image_every` | 10 | 每幾個 epoch 重新算一次 triplet |
| `--wandb_watch_freq` | 0 | `wandb.watch` log_freq（0 = 不開） |

---

## 8. 附錄：資料流維度追蹤

以預設 `dim=48` + 訓練輸入 `(B, 3, 128, 128)`、測試輸入 `(1, 3, 256, 256)` 為例。

### 訓練 (128×128 input)

```
Input        : (B, 3, 128, 128)
patch_embed  : (B,  48, 128, 128)
encoder_lv1  : (B,  48, 128, 128)
down1_2      : (B,  96,  64,  64)
encoder_lv2  : (B,  96,  64,  64)
down2_3      : (B, 192,  32,  32)
encoder_lv3  : (B, 192,  32,  32)
down3_4      : (B, 384,  16,  16)
latent       : (B, 384,  16,  16)
+ prompt3    : (B, 704,  16,  16)   ← concat with PromptGenBlock(320)
noise_lv3    : (B, 704,  16,  16)
reduce_noise3: (B, 192,  16,  16)
up4_3        : (B,  96,  32,  32)
cat skip_lv3 : (B, 288,  32,  32)
reduce_chan3 : (B, 192,  32,  32)
decoder_lv3  : (B, 192,  32,  32)
+ prompt2    : (B, 320,  32,  32)   ← concat with PromptGenBlock(128)
noise_lv2    : (B, 320,  32,  32)
reduce_noise2: (B, 192,  32,  32)
up3_2        : (B,  96,  64,  64)
cat skip_lv2 : (B, 192,  64,  64)
reduce_chan2 : (B,  96,  64,  64)
decoder_lv2  : (B,  96,  64,  64)
+ prompt1    : (B, 160,  64,  64)   ← concat with PromptGenBlock(64)
noise_lv1    : (B, 160,  64,  64)
reduce_noise1: (B,  96,  64,  64)
up2_1        : (B,  48, 128, 128)
cat skip_lv1 : (B,  96, 128, 128)
decoder_lv1  : (B,  96, 128, 128)
refinement   : (B,  96, 128, 128)
output conv  : (B,   3, 128, 128)
+ input      : (B,   3, 128, 128)
```

### 推論 (256×256 input)

每一層的 spatial 都翻倍，channel 維完全不變：

```
latent          : (1, 384, 16, 16)  →  (1, 384, 32, 32)   (now 1/8 of 256)
其他層解析度同步加倍；prompt 因 F.interpolate 自動拉伸至 feature 同尺寸，
故同一份 prompt_param 可同時支援 128 / 256 / 任意大小的輸入。
```

### 參數量

| 配置 | 參數量 |
|------|--------|
| **default** (dim=48, blocks=[4,6,6,8], refine=4) | **22.85 M** |
| ablation (decoder=False) | 14.82 M |
| lightweight (dim=32, blocks=[2,3,3,4]) | 10.29 M |

### 引用

- **PromptIR**: Potlapalli, Zamir, Khan, Khan. *PromptIR: Prompting for All-in-One Blind Image Restoration*. NeurIPS 2023. arXiv:2306.13090.
- **NAFNet**: Chen, Chu, Zhang, Sun. *Simple Baselines for Image Restoration*. ECCV 2022. arXiv:2204.04676.
- **Restormer**: Zamir, Arora, Khan, Hayat, Khan, Yang. *Restormer: Efficient Transformer for High-Resolution Image Restoration*. CVPR 2022.
