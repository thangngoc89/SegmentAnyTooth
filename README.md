# SegmentAnyTooth

An open-source deep learning framework for tooth enumeration and segmentation in intraoral photos.

---

## 📢 Update (2025-01-17)

We are pleased to announce that our paper has been accepted in the **Journal of Dental Sciences**.

**Citation**:
> Nguyen, K. D., Hoang, H. T., Doan, T.-P. H., Dao, K. Q., Wang, D.-H., & Hsu, M.-L. (2025).  
> *SegmentAnyTooth: An open-source deep learning framework for tooth enumeration and segmentation in intraoral photos.*  
> Journal of Dental Sciences, S1991790225000030. https://doi.org/10.1016/j.jds.2025.01.003

<p align="center">
  <img src="https://github.com/thangngoc89/SegmentAnyTooth/raw/refs/heads/main/SegmentAnyTooth_text.webp" alt="SegmentAnyTooth" width="600"/>
</p>

---

## 🚀 Installation

We use [`uv`](https://github.com/astral-sh/uv) as the dependency manager.

All dependencies are specified in `pyproject.toml`.  
Please prefer using `pyproject.toml` directly to manage the environment.

Example installation:

```bash
pip install uv
uv pip install -r pyproject.toml
```

(Optionally) Install development dependencies:

```bash
uv pip install -r pyproject.toml --group dev
```
## Weights download

The weight is released under [SegmentAnyTooth Non-Commercial License](./SegmentAnyTooth_license_agreement.pdf).

To obtain the weight, please sign the agreement form and email the signed version to [hi+segmentanytooth@khoanguyen.me](mailto:hi+segmentanytooth@khoanguyen.me). I will email the link for weight download in working days.

## 🚀 Usage Example

After installing the dependencies, you can run predictions:

```python
from segmentanytooth import predict

# Predict tooth masks from an intraoral image
mask = predict(
    image_path="path/to/your/image.png",
    view="upper",            # one of: "upper", "lower", "left", "right", "front"
    weight_dir="./weights",  # path to downloaded model weights
    sam_batch_size=10,       # optional: adjust batch size for faster SAM inference
)

# Save the predicted mask
import cv2
cv2.imwrite("predicted_mask.jpg", mask * 5)  # Scale mask for visualization if needed
```

Notes:
-	Model weights must be placed in the weights/ directory (or specify another directory).
-	Views correspond to different photographic angles of intraoral images.
-	Output mask is a NumPy array with pixel values representing **FDI** tooth numbers.

## 📜 License

-	Code is licensed under the MIT License.
You are free to use, modify, and distribute the code with attribution.
-	Pretrained model weights are provided under a Non-Commercial Use License.
Commercial use of the weights is prohibited without explicit permission.

Please see:
-	[LICENSE](./LICENSE) for full terms of the MIT License.
-	[WEIGHT_LICENSE](./SegmentAnyTooth_license_agreement.pdf) for the model weights licensing terms.

## 📚 Citation

If you use SegmentAnyTooth in your work, please cite:

```bibtex
@article{Nguyen2025SegmentAnyTooth,
  title={SegmentAnyTooth: An open-source deep learning framework for tooth enumeration and segmentation in intraoral photos},
  author={Nguyen, Khoa D. and Hoang, Huy T. and Doan, Thi-Phuong-Hoa and Dao, Kim-Quyen and Wang, Ding-Han and Hsu, Min-Ling},
  journal={Journal of Dental Sciences},
  year={2025},
  doi={10.1016/j.jds.2025.01.003}
}
```