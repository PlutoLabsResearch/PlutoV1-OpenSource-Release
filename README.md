# PlutoV1

Back in 2023, we had a pretty crazy idea: build a single technique that could handle *any* input and *any* output—text, images, audio, video—and actually get *better* over time. Not just incrementally better. The kind of better that unlocks entirely new capabilities.

Most models back then were specialized. Stable Diffusion did images. Audio models did sound. If you wanted music with lyrics, you were stitching together half a dozen different systems. We wanted one model that could do it all.

PlutoV1 was our first real swing at this. It worked. It worked *really* well. Well enough that we built an entire generation stack on top of it—music with lyrics, images, all from the same core. Video was the one that got away; we just didn’t have the data to make it *perfect*. But we did get it working—sort of. Here’s a demo from back then:

https://github.com/user-attachments/assets/2e8baa59-ef82-4683-9503-da22acdbc7ee


*Drone shoot through a forest. 55 seconds. Pixelated, lagging, raw. This is what happens when you try to do video with a model trained on limited data. It’s not pretty, but it’s proof that the technique *can* handle it—given enough compute and data, it would have worked.*

But for everything else? This thing delivered.

Now we’ve moved on to bigger and better things internally (PlutoV3 builds on these ideas), so we’re opening this up. Consider it a foundation, a proof point, and—if you’re paying attention—a hell of a starting point for your own work.

---

## What This Thing Actually Is

PlutoV1 is a **unified generative model** built around a few core ideas:

1. **A codec that actually understands structure.** Most codecs just compress. Ours learns hierarchical representations—residual vector quantization with multiple levels, so it captures both the fine details and the big picture. This means it doesn’t just compress; it *represents* the data in a way that’s meaningful across modalities.

2. **A backbone that picks its battles.** We use a hybrid architecture: SSM blocks (think S4D-style state spaces) for the heavy lifting, with attention sprinkled in only where it matters. This keeps the quadratic cost in check while still nailing the long-range dependencies. The result is a model that can handle sequences most transformers would choke on.

3. **Diffusion that doesn’t waste time.** Our decoder uses rectified flow—straight paths, minimal steps, fast sampling. No fancy scheduling, just good math. This means you can generate high-quality outputs without waiting forever.

4. **A training loop that doesn’t quit.** Adversarial training, spectral losses, reinforcement learning fine-tuning—we throw everything at it until it sticks. The model doesn’t just learn; it *adapts*.

The result? A single system that can take in raw bytes, audio waveforms, or pixel grids, compress them into discrete tokens, and generate new ones that actually make sense—*across multiple modalities*.

---

## What We Built With It

| Domain | Status | Notes |
|--------|--------|-------|
| **Images** | ✅ Worked great | The codec + diffusion combo handles textures, compositions, and fine details without falling apart. More importantly, it does this as part of a unified system—not a siloed image model. |
| **Music (with lyrics)** | ✅ Surprisingly good | This is where things get interesting. The same architecture that handles images also *natively* understands structured audio. Feed it a prompt like "jazz piano with vocals about a rainy day" and it delivers both the music *and* the lyrics, coherent and in key. Try getting SDXL to do that. |
| **Video** | ❌ Not quite | We tried. The technique was sound, but we just didn’t have the compute or the data to train it properly. Video at scale is a different animal—you need orders of magnitude more firepower. |

---

## The Code: What’s Actually in Here

We’re not giving you the full training pipeline (some secrets stay secret), but we *are* giving you the guts of the system. Here’s what you’ll find:

```
PlutoV1-OpenSource-Release/
├── core/
│   ├── backbone.py    # Hybrid SSM + Attention backbone
│   ├── codec.py       # Residual VQ codec (the magic)
│   ├── diffusion.py   # Rectified flow diffusion decoder
│   ├── model.py       # The main transformer model
│   ├── train.py       # Training loops (codec, model, RL)
│   ├── data.py        # Data loading and preprocessing
│   ├── vocab.py       # Token vocabulary (bytes, codec tokens, control codes)
│   ├── quantize.py    # Vector quantization implementation
│   ├── critic.py      # Adversarial critic for training
│   ├── improve.py     # Fine-tuning and improvement utilities
│   └── verify.py      # Verification and evaluation tools
└── adapters/
    └── speech2text.py # Audio preprocessing for speech-to-text
```

---

## Training It Yourself

Yes, you *can* train this. No, it won’t be cheap. But if you’re serious about understanding how this stuff works, here’s how we did it.

### Step 1: The Codec

The codec is the foundation. It compresses your raw data (audio waveforms, image pixels, whatever) into discrete tokens. Train this first.

```python
from core.codec import Codec
from core.train import Trainer
from core.data import chunk_1d, load_file
import torch

# For audio (1D)
codec = Codec(
    ndim=1,           # 1D for audio, 2D for images
    in_channels=1,    # Mono audio
    channels=32,      # Start small, scale up
    latent=128,       # Latent dimension
    strides=(2,4,4,5), # Downsampling strides
    codebook_size=512, # Number of codebook entries
    levels=4          # Residual VQ levels
)

# Load your data (audio files, normalized to [-1, 1])
# This is a simplified example - you'll need real data loading
audio_files = [...]  # List of paths
data = torch.cat([chunk_1d([load_file(f)], length=65536) for f in audio_files])

# Train it
trainer = Trainer(lr=3e-4, weight_decay=0.05, wave_weight=50.0)
codec = trainer.fit_codec(codec, data, steps=6000, batch=16, use_spectral=True)
```

**What’s happening here:**
- The codec uses **residual vector quantization**—each level refines the error from the previous one. This means it captures hierarchical structure, not just flat compression.
- `strides` control how much it downsamplees. `(2,4,4,5)` means it compresses by 2*4*4*5 = 160x in the time dimension.
- `use_spectral=True` adds a spectral loss (STFT-based) for audio. For images, you’d skip this.
- The `wave_weight=50.0` keeps the waveform reconstruction loss dominant. The spectral loss is a regularizer, not the main objective.

**Pro tip:** Start with small data. Train the codec on 10-20 hours of audio first, get it working, then scale up. If it can’t reconstruct a single sample well, it’ll never work at scale.

### Step 2: The Model

Once you have a trained codec, you can tokenize your data and train the generative model.

```python
from core.model import Model
from core.vocab import vocab_size, sequence, encode_bytes, AUDIO_MARK

# Tokenize your data with the trained codec
def tokenize_audio(audio, codec):
    # audio: (1, 1, N) tensor
    with torch.no_grad():
        idx, grid = codec.encode(audio)
    # Convert codec indices to tokens
    tokens = encode_codec(idx, codec.quantizer.size, codec.quantizer.levels)
    # Add modality marker
    return sequence(AUDIO_MARK, tokens)

# Build the model
model = Model(
    codebook=512,      # Must match codec
    levels=4,          # Must match codec
    dim=512,           # Model dimension
    depth=12,          # Number of layers
    attn_every=4,      # Attention every 4th layer
    heads=8,           # Attention heads
    state=64,          # SSM state size
    dropout=0.1,       # Dropout rate
    max_len=8192       # Max sequence length
)

# Train it
trainer = Trainer(lr=3e-4, weight_decay=0.05)
model = trainer.fit_model(
    model,
    sequences,        # List of token sequences
    steps=60000,       # Training steps
    batch=12,          # Batch size
    ctx=512,           # Context length
    held=held_out,     # Held-out validation set
    patience=8         # Early stopping patience
)
```

**What’s happening here:**
- The model is a **transformer with a hybrid backbone**—SSM blocks for most layers, with attention inserted every `attn_every` layers.
- `dim=512` is a good starting point. Bigger = better but slower.
- `depth=12` with `attn_every=4` means 9 SSM layers and 3 attention layers. Only 25% of layers are quadratic.
- The tokenizer prepends modality markers (`AUDIO_MARK`, `IMAGE_MARK`, etc.) so the model knows what kind of data it’s dealing with.

**Pro tip:** Use a learning rate scheduler. We use `OneCycleLR`—it warms up, peaks, then decays. Works better than constant LR for this architecture.

### Step 3: Adversarial Training (Optional but Recommended)

The codec and model will work without it, but adversarial training sharpens everything up.

```python
from core.critic import Critic

# Build the critic
critic = Critic(
    scales=3,      # Multi-scale discrimination
    channels=16,   # Critic channels
    lr=2e-4,       # Critic learning rate
    adv_weight=1.0,
    fm_weight=2.0,  # Feature matching weight
    warmup=2000     # Steps before critic is active
)

# Train codec with adversarial loss
trainer = Trainer(critic=critic, lr=3e-4)
codec = trainer.fit_codec(codec, data, steps=6000, batch=16, use_spectral=True)
```

**What’s happening here:**
- The critic is a **multi-scale discriminator**—it looks at the data at different resolutions.
- `warmup=2000` gives the codec a chance to learn basic reconstruction before the critic starts pushing back.
- `fm_weight=2.0` means feature matching loss is twice as important as the adversarial loss. This stabilizes training.

**Warning:** Adversarial training is finicky. If your codec isn’t already reconstructing well, the critic will just make things worse. Get the basics working first.

### Step 4: Reinforcement Learning Fine-Tuning (Advanced)

This is where things get interesting. Once you have a trained model, you can fine-tune it with RL to optimize for specific metrics.

```python
from core.verify import Verifier

# Define a verifier (reward function)
class MyVerifier(Verifier):
    def collect(self, model, problems, samples=8, max_new=320):
        # Generate samples and compute rewards
        # Return (kept_samples, total_attempts)
        ...

verifier = MyVerifier()

# Fine-tune with RL
trainer.reinforce(
    model,
    problems,        # List of prompts/problems
    verifier,
    samples=8,       # Samples per problem
    rounds=3,        # RL rounds
    max_new=320      # Max new tokens per sample
)
```

**What’s happening here:**
- The verifier defines what "good" means. For music, maybe it’s melodic coherence. For images, maybe it’s aesthetic quality.
- The model generates samples, the verifier scores them, and the model is fine-tuned to maximize those scores.
- This is where the model starts to *really* shine—but it’s also where things can go off the rails if your reward function is bad.

---

## The Architecture Deep Dive

### Codec

The codec is a **convolutional encoder-decoder with residual vector quantization**:

```
Input -> Encoder (strided convs) -> Latent -> ResidualVQ -> Tokens
Tokens -> ResidualVQ.decode -> Latent -> Decoder (strided transposed convs) -> Output
```

- **Encoder:** Downsample with strided convolutions, increasing channels at each level.
- **Decoder:** Upsample with strided transposed convolutions, decreasing channels.
- **ResidualVQ:** Multiple levels of vector quantization. Each level quantizes the residual from the previous level. This captures hierarchical structure.

**Why residual?** Because a single VQ level can’t capture both fine details and coarse structure. The first level gets the big picture, the second level refines it, and so on.

### Backbone

The backbone is where the magic happens. It’s a **hybrid SSM + attention architecture**:

```
SSMBlock -> SSMBlock -> SSMBlock -> Attention -> SSMBlock -> ...
```

- **SSMBlock:** State space model with a convolutional gating mechanism. Think of it as a more efficient, longer-range RNN.
- **CausalSelfAttention:** Standard transformer attention, but only used every `attn_every` layers.
- **FeedForward:** Gated linear unit (GLU) style feed-forward network.

**Why hybrid?** Because attention is quadratic in sequence length. If every layer is attention, you’re limited to short sequences. By using SSM for most layers, we get linear scaling with sequence length, and only pay the quadratic cost for a fraction of layers.

### Diffusion Decoder

For raw data generation (like audio waveforms), we use a **conditioned diffusion decoder**:

```
Input (noise) + Conditioning -> TimeEmbedding -> ConditionedBlock x N -> Output
```

- **TimeEmbedding:** Encodes the timestep into a conditioning vector.
- **ConditionedBlock:** Residual block with time and class conditioning.
- **Rectified Flow:** A diffusion variant where the probability flow is a straight line. This means we can sample in very few steps (8-16) without losing quality.

---

## Training Tips (From Our Scars)

1. **Start small.** Train the codec on a tiny dataset first. Get it working. Then scale up.

2. **Monitor the codebook usage.** If your VQ codebook has dead entries, your model is missing out on expressive power. The `usage()` method tells you what percentage of the codebook is actually being used. If it’s below 80%, something’s wrong.

3. **Spectral loss is your friend for audio.** L1/MSE loss on waveforms can sound "blurry." Spectral loss (STFT magnitude) sharpens it up.

4. **Adversarial training is a double-edged sword.** It can make your outputs *amazing*, or it can make your training *unstable*. Warm it up, keep the weights reasonable, and don’t let it dominate the loss.

5. **The SSM state size matters.** `state=64` is a good default, but bigger states remember longer-range dependencies. If your data has very long-range structure (like music), consider `state=128` or `state=256`.

6. **Mixed precision is your friend.** Use `bfloat16` if your GPU supports it. The SSM blocks need `float32` for numerical stability, but most of the model can run in `bfloat16`.

7. **Gradient clipping is non-negotiable.** We clip at 1.0. Without it, the SSM blocks will explode.

8. **Learning rate scheduling helps.** We use `OneCycleLR` with a warmup phase. It’s simpler and more effective than cosine annealing for this architecture.

9. **Batch size matters more than you think.** Too small, and the model overfits. Too large, and you’ll run out of memory. For the codec, 16-32 is a good range. For the model, 8-12 works well.

10. **Patience is a virtue.** These models take *time* to train. The codec might take 10K-100K steps to converge. The model might take 100K-1M steps. Don’t expect miracles overnight.

---

## Hardware Requirements

| Component | Minimum | Recommended | "We Used This" |
|-----------|---------|-------------|----------------|
| **GPU** | 1x A100 (40GB) | 4x A100 (80GB) | 8x A100 (80GB) |
| **VRAM** | 40GB | 80GB+ | 640GB total |
| **CPU** | 8 cores | 16+ cores | 64 cores |
| **RAM** | 32GB | 64GB+ | 256GB |
| **Storage** | 1TB NVMe | 10TB NVMe | 100TB NVMe + Lustre |

**Can you train on less?** Sure. You’ll need to reduce model size, batch size, and sequence length. But don’t expect to beat SDXL on a single RTX 3090.

---

## Data Requirements

| Domain | Minimum | Good | "We Used This" |
|--------|---------|------|----------------|
| **Images** | 10K images | 100K+ images | 100M+ images |
| **Audio** | 10 hours | 100+ hours | 10K+ hours |
| **Music** | 50 hours | 500+ hours | 5K+ hours |

**Quality matters more than quantity.** A well-curated 100K image dataset will beat a noisy 10M dataset every time.

---

## What We’re Not Giving You

This is an open-source release, but it’s not a turnkey solution. Here’s what we’re *not* including:

1. **The full training pipeline.** We’re giving you the core components, but not the distributed training infrastructure, data loaders, or preprocessing pipelines.

2. **Pre-trained weights.** You’ll need to train this yourself. Sorry.

3. **The video training code.** We tried. It didn’t work. The code for it is a mess. We’re not subjecting you to that.

4. **PlutoV3.** That’s still our secret sauce. This is V1—great, but not our best.

---

## Why We’re Open-Sourcing This

A few reasons:

1. **It’s old news to us.** We’ve moved on. PlutoV3 is where our focus is now.

2. **It’s still better than most.** Even though it’s "old," this technique beats a lot of what’s out there today. If you’re building generative models, there’s a lot to learn here.

3. **We want to see what you do with it.** The best ideas come from unexpected places. Maybe you’ll take this in a direction we never considered.

4. **The community deserves good tools.** Too much of the generative AI space is locked behind corporate walls. We’re doing our part to change that.

---

## License

This code is released under the **MIT License**. Do with it what you will. We’d love to hear about what you build, but we’re not requiring it.

---

## Final Thoughts

This was our first real swing at building a unified generative model. It worked. It worked *really* well. And now it’s yours.

Will it match SDXL on images out of the box? Probably not—it’s a generalist, not a specialist. But that’s the point. SDXL does one thing incredibly well. This does *many* things well. And if you train it right, it can do things SDXL simply can’t.

So yes, you’ll need to train it, tune it, and probably fight with it a bit. But if you put in the work, it’ll reward you with capabilities most models can’t touch.

And who knows? Maybe you’ll be the one to take this to the next level.

-- The Pluto Labs Team

---

*This README was generated with the help of AI, but the story, the code, and the lessons learned are all real.*
