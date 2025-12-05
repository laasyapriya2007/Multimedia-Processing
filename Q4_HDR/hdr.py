# hdr.py — FINAL version with correct tone mapping (Drago)
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

IN_DIR = Path("images")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

def list_images():
    files = sorted(list(IN_DIR.glob("*.jpg")) + list(IN_DIR.glob("*.png")))
    if not files:
        raise SystemExit("❌ No images found in 'images/' folder.")
    return [str(f) for f in files]

def save_crf_plot(response_curve, out_file):
    g = response_curve.reshape(256, 3)
    x = np.arange(256)
    plt.figure(figsize=(7, 4))
    plt.plot(x, g[:, 2], label="R")
    plt.plot(x, g[:, 1], label="G")
    plt.plot(x, g[:, 0], label="B")
    plt.xlabel("Pixel value (Z)")
    plt.ylabel("g(Z)")
    plt.title("Camera Response Curves")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()

def save_irradiance_vis(hdr, out_file):
    L = 0.2126 * hdr[:, :, 2] + 0.7152 * hdr[:, :, 1] + 0.0722 * hdr[:, :, 0]
    L_log = np.log(L + 1e-8)
    L_norm = (L_log - L_log.min()) / (L_log.max() - L_log.min() + 1e-12)
    vis = (L_norm * 255).astype(np.uint8)
    cv2.imwrite(str(out_file), vis)

def main():
    print("\n📥 Loading images...")
    paths = list_images()

    # Load images and resize to same shape
    raw = [cv2.imread(p) for p in paths]
    h, w = raw[0].shape[:2]
    imgs = [cv2.resize(im, (w, h)) for im in raw]
    print(f"✔ Loaded {len(imgs)} images of size {h}×{w}")

    # Manual exposure times
    print("⚠ Using manual exposure values...")
    times = np.array([1/60, 1/30, 1/15, 1/8], dtype=np.float32)

    # Save exposure times
    with open(OUT_DIR / "exposure_times.txt", "w") as f:
        for p, t in zip(paths, times):
            f.write(f"{Path(p).name}\t{float(t)}\n")

    # CRF
    print("📈 Estimating CRF...")
    calibrate = cv2.createCalibrateDebevec()
    response = calibrate.process(imgs, times)
    save_crf_plot(response, OUT_DIR / "crf_curves.png")

    # HDR merge
    print("🌈 Merging HDR...")
    merge = cv2.createMergeDebevec()
    hdr = merge.process(imgs, times, response)
    cv2.imwrite(str(OUT_DIR / "hdr_output.hdr"), hdr.astype(np.float32))

    # Tone mapping — DRAGO (best choice)
    print("🎨 Tone mapping (Drago)...")
    tonemap = cv2.createTonemapDrago(gamma=1.0, saturation=1.0, bias=0.85)

    ldr = tonemap.process(hdr)
    ldr = np.clip(ldr * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(str(OUT_DIR / "ldr_output.jpg"), ldr)

    # Irradiance visualization
    save_irradiance_vis(hdr, OUT_DIR / "irradiance_vis.png")

    print("\n🎉 DONE! All results saved in 'outputs/' folder.\n")

if __name__ == "__main__":
    main()
