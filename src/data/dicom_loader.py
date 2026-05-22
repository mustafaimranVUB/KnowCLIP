"""DICOM image loading and preprocessing for MIMIC-CXR.

Reads DICOM files via ``pydicom``, applies windowing, and converts to
normalised tensors suitable for CLIP-family encoders.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

RASTER_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ---------------------------------------------------------------------------
# Default windowing for CXR (centre / width)
# ---------------------------------------------------------------------------
DEFAULT_WINDOW_CENTER = 40
DEFAULT_WINDOW_WIDTH = 400


def _apply_windowing(
    pixel_array: np.ndarray,
    window_center: float,
    window_width: float,
) -> np.ndarray:
    """Apply VOI LUT windowing to a 2-D pixel array.

    Args:
        pixel_array: Raw pixel data (int16/uint16).
        window_center: DICOM window centre.
        window_width: DICOM window width.

    Returns:
        Windowed image scaled to [0, 1] float32.
    """
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    img = np.clip(pixel_array.astype(np.float32), lower, upper)
    img = (img - lower) / max(window_width, 1e-6)
    return img


class DICOMLoader:
    """Load and preprocess DICOM chest X-ray images.

    Parameters:
        image_size: Target spatial size (square).
        use_clahe: Whether to apply CLAHE contrast enhancement.
    """

    # CLIP-family normalisation stats (ImageNet)
    IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
    IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        image_size: int = 224,
        use_clahe: bool = False,
    ) -> None:
        self.image_size = image_size
        self.use_clahe = use_clahe

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_image(self, image_path: Path | str) -> Optional[PILImage.Image]:
        """Load either a DICOM or a raster image into a preprocessed PIL image.

        This keeps the dataset path resolution flexible so the same training
        pipeline can ingest native MIMIC-CXR DICOMs or the MIMIC-CXR-JPG
        release, which preserves the same study/image identifiers.
        """
        path = Path(image_path)
        if path.suffix.lower() in RASTER_IMAGE_SUFFIXES:
            return self.load_raster_image(path)
        return self.load_dicom(path)

    def load_dicom(self, dicom_path: Path | str) -> Optional[PILImage.Image]:
        """Load a single DICOM file and return a preprocessed PIL Image.

        Args:
            dicom_path: Path to a ``.dcm`` file.

        Returns:
            RGB PIL Image of size ``(image_size, image_size)`` or ``None``
            if the file cannot be read.
        """
        try:
            import pydicom  # type: ignore

            ds = pydicom.dcmread(str(dicom_path))
            pixel_array = ds.pixel_array.astype(np.float32)

            # Photometric Interpretation — invert if MONOCHROME1
            pi = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
            if pi == "MONOCHROME1":
                pixel_array = pixel_array.max() - pixel_array

            # Windowing
            wc = getattr(ds, "WindowCenter", DEFAULT_WINDOW_CENTER)
            ww = getattr(ds, "WindowWidth", DEFAULT_WINDOW_WIDTH)
            # Multi-valued windowing: take first
            if isinstance(wc, (list, pydicom.multival.MultiValue)):
                wc = float(wc[0])
            if isinstance(ww, (list, pydicom.multival.MultiValue)):
                ww = float(ww[0])
            wc, ww = float(wc), float(ww)

            img = _apply_windowing(pixel_array, wc, ww)

            # Optional CLAHE
            if self.use_clahe:
                img = self._apply_clahe(img)

            # Resize
            img = self._resize(img, self.image_size)

            # Convert to PIL RGB image (transforms handle ToTensor + Normalize)
            pil_img = self._to_pil(img)
            return pil_img

        except Exception as exc:
            logger.warning("Failed to load DICOM %s: %s", dicom_path, exc)
            return None

    def load_raster_image(self, image_path: Path | str) -> Optional[PILImage.Image]:
        """Load a JPG/PNG-style image and normalise it to the dataset PIL format.

        MIMIC-CXR-JPG images are already derived from the corresponding DICOMs,
        so we preserve their grayscale content and align the output shape with the
        DICOM code path by resizing and converting back to RGB.
        """
        try:
            with PILImage.open(image_path) as image:
                grayscale = image.convert("L")
                resized = grayscale.resize(
                    (self.image_size, self.image_size),
                    resample=PILImage.Resampling.BILINEAR,
                )
                return resized.convert("RGB")
        except Exception as exc:
            logger.warning("Failed to load raster image %s: %s", image_path, exc)
            return None

    def load_preprocessed(self, tensor_path: Path | str) -> Optional[torch.Tensor]:
        """Load a pre-processed ``.pt`` tensor from cache.

        Args:
            tensor_path: Path to a ``.pt`` file saved by :meth:`save_preprocessed`.

        Returns:
            Tensor of shape ``(3, image_size, image_size)`` or ``None``.
        """
        try:
            return torch.load(str(tensor_path), map_location="cpu", weights_only=True)
        except Exception as exc:
            logger.warning("Failed to load cached tensor %s: %s", tensor_path, exc)
            return None

    @staticmethod
    def save_preprocessed(tensor: torch.Tensor, out_path: Path | str) -> None:
        """Save a preprocessed tensor to disk."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, str(out_path))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resize(img: np.ndarray, size: int) -> np.ndarray:
        """Resize a 2-D image to (size, size) using skimage."""
        from skimage.transform import resize as sk_resize  # type: ignore

        return sk_resize(img, (size, size), anti_aliasing=True, preserve_range=True).astype(
            np.float32
        )

    @staticmethod
    def _to_pil(img: np.ndarray) -> PILImage.Image:
        """Convert a [0,1] grey-scale numpy image to an RGB PIL Image."""
        uint8_img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        return PILImage.fromarray(uint8_img, mode="L").convert("RGB")

    def _to_normalised_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert a [0,1] grey-scale image to a normalised 3-channel tensor."""
        # (H, W) → (3, H, W)
        img_3ch = np.stack([img, img, img], axis=0)
        tensor = torch.from_numpy(img_3ch)

        # Per-channel normalisation (ImageNet stats used by CLIP)
        mean = torch.tensor(self.IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(self.IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor

    @staticmethod
    def _apply_clahe(img: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement."""
        from skimage.exposure import equalize_adapthist  # type: ignore

        return equalize_adapthist(img, clip_limit=0.01).astype(np.float32)


# ---------------------------------------------------------------------------
# CLI for batch DICOM pre-processing
# ---------------------------------------------------------------------------

def _preprocess_batch(
    input_dir: Path,
    output_dir: Path,
    image_size: int = 224,
    num_workers: int = 4,
) -> None:
    """Pre-process all DICOM files in *input_dir* and save as .pt tensors."""
    import concurrent.futures

    loader = DICOMLoader(image_size=image_size)
    dcm_files = sorted(input_dir.rglob("*.dcm"))
    logger.info("Found %d DICOM files in %s", len(dcm_files), input_dir)

    def _process_one(dcm_path: Path) -> Tuple[str, bool]:
        rel = dcm_path.relative_to(input_dir)
        out_path = output_dir / rel.with_suffix(".pt")
        if out_path.exists():
            return str(rel), True
        pil_img = loader.load_dicom(dcm_path)
        if pil_img is None:
            return str(rel), False
        tensor = loader._to_normalised_tensor(
            np.array(pil_img.convert("L"), dtype=np.float32) / 255.0
        )
        loader.save_preprocessed(tensor, out_path)
        return str(rel), True

    success = 0
    fail = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_one, p): p for p in dcm_files}
        for future in concurrent.futures.as_completed(futures):
            _rel, ok = future.result()
            if ok:
                success += 1
            else:
                fail += 1
            if (success + fail) % 5000 == 0:
                logger.info("Processed %d / %d (failed: %d)", success + fail, len(dcm_files), fail)

    logger.info(
        "Preprocessing complete: %d success, %d failed out of %d total",
        success,
        fail,
        len(dcm_files),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch preprocess DICOM to .pt")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    _preprocess_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
