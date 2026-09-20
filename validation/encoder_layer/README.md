# Encoder validation through WeightLoader

Place `weights.json` beside the twelve existing `ref_weight_*.bin` files and
the two `ref_encoder_layer_input.bin` / `ref_encoder_layer_output.bin` files
from the same Colab export. No new reference export is needed.

The manifest describes the existing configuration: model width 64, eight heads,
feed-forward width 256, pre-normalization, GELU, and evaluation mode.

The validator requires an installed funlib containing `WeightLoader` (both the
updated headers and library). Its existing CMake file links that installed library.
After rebuilding/installing funlib and rebuilding the validator yourself, invoke:

```bash
./validate_encoder_layer /path/to/reference_directory
```

The directory defaults to the current working directory. All parameter files are
resolved relative to `weights.json`. The loader preserves the PyTorch parameter
names and shapes; the validator checks expected shapes, transposes linear weights,
splits packed Q/K/V, expands biases, and uploads tensors to the CUDA queue.
Input and expected output still use the reference binary reader.

The encoder calculations and strict max-absolute-difference threshold of `1e-3`
are unchanged. A successful run prints the loaded tensor count, PyTorch/funlib
output values, maximum difference, and `PASS`. This validates the host loading
path followed by the validator's uploads; it does not exercise the loader's
optional direct-to-device constructor.
