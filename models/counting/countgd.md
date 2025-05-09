
I looked at some counting models this one shouldnt
be all to hard to implement.
one could also do it via the github repo 
https://github.com/niki-amini-naieni/CountGD

here is the paper:
https://arxiv.org/pdf/2407.04619

here is the leaderboard:
https://paperswithcode.com/sota/object-counting-on-fsc147?p=countr-transformer-based-generalised-visual

but I checked that it ios also integreated into huggingface
## Setup

```python
from transformers import CountGDProcessor, CountGDForCounting
import torch

model_name = "nikigoli/CountGD"
device     = "cuda"
processor = CountGDProcessor.from_pretrained(model_name)
model     = CountGDForCounting.from_pretrained(model_name).to(device)
```

## Inference

```python
# image: PIL.Image or numpy array
# prompt: e.g. "How many apples?"
inputs  = processor(images=image, text=prompt, return_tensors="pt", padding=True).to(device)
outputs = model(**inputs)
count   = int(outputs.count)
```

## Integration Sketch

Create `countgd_model.py` alongside your other evaluators:

```python
from transformers import CountGDProcessor, CountGDForCounting
from .base import Evaluator, BaseDataset

class CountGDModel(Evaluator):
    def __init__(self, model_name="nikigoli/CountGD", device="cpu"):
        super().__init__(system_prompt=None)
        self.processor = CountGDProcessor.from_pretrained(model_name)
        self.model     = CountGDForCounting.from_pretrained(model_name).to(device)

    def __str__(self):
        return "countgd"

    def eval(self, dataset_dir: Path, result_file: Path, batch_size=1):
        df      = pd.read_csv(dataset_dir / "dataset.csv")
        dataset = BaseDataset(df, image_dir=dataset_dir / "images")
        with open(result_file, "w") as f:
            f.write("idx,count\n")
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i : i + batch_size]
                imgs  = [b["image"] for b in batch]
                txts  = [b["text"]  for b in batch]
                inp   = self.processor(images=imgs, text=txts, return_tensors="pt", padding=True).to(self.model.device)
                out   = self.model(**inp)
                for j, c in enumerate(out.count.detach().cpu()):
                    f.write(f"{i+j},{int(c)}\n")
```

---

**Notes:**

* No extra config files needed.
* Use `padding=True` for batching.
* Keep device handling consistent.


