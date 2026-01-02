# HMC Computer Vision Project: Kalman-Based Bubble Tracking

> **Collaborative project with Thomas Wing**  
> Built at Harvey Mudd College’s Flow Imaging Lab (FILM)

This project explores computer vision and temporal tracking techniques to improve the detection and removal of air bubbles in underwater fin-motion videos. The work was motivated by real research needs in biomechanics and fluid-flow analysis, where bubbles interfere with downstream techniques such as Particle Image Velocimetry (PIV).

---

## Motivation

When mechanical fins move underwater, they generate bubbles and air pockets that distort the surrounding flow. FILM researchers need to remove these air regions frame-by-frame so they can study water motion alone.

While existing bubble detection pipelines (thresholding, background subtraction, morphology) work reasonably well, they fail when:
- bubbles become transparent,
- multiple bubbles overlap,
- detections briefly disappear due to lighting or motion.

These failures require extensive manual correction and don’t scale to long, high-resolution videos.

Our goal was to **add temporal reasoning** to the pipeline so bubble detection could remain stable over time instead of treating each frame independently.

---

## Approach

We combined classical computer vision with **Kalman filter–based tracking** to improve robustness across frames.

### 1. Mask Optimizing Video Editor (MOVE)

To rapidly experiment with filter pipelines, we built a lightweight Python video editor that:
- loads `.mov` files using OpenCV,
- applies filters across all frames via a decorator-based API,
- allows fast visual inspection of intermediate results,
- saves processed videos back to disk.

This tool made it easy to prototype and tune filter sequences without rewriting code for every experiment.

---

### 2. Bubble & Fin Masking

We implemented custom pipelines for:
- **Waterline extraction** using grayscale conversion, frame averaging, Gaussian blurring, and k-means color quantization.
- **Fin and bubble extraction** using background subtraction, contrast enhancement, thresholding, and median filtering.

These steps produced first-pass masks that served as inputs to the tracking system.

---

### 3. Kalman Filter Tracking

To stabilize bubble detection over time, we modeled each bubble using a **constant-velocity Kalman filter** with state: [x, y, vx, vy]



At each frame, the tracker:
1. Predicts the bubble’s next position.
2. Updates the estimate when a detection is available.
3. Continues predicting during short detection failures.

This allows the system to:
- smooth noisy detections,
- bridge short disappearances,
- reduce sudden ID switches between bubbles.

---

### 4. Synthetic Video Evaluation

Before applying the tracker to messy real footage, we created **synthetic videos** of moving white circles to isolate tracking behavior under controlled conditions. These videos simulated:
- bubbles crossing paths,
- temporary disappearances,
- jitter and noise,
- stops and slowdowns.

This made it possible to evaluate tracking behavior precisely and understand failure modes.

---

## Results

- The Kalman tracker successfully maintained bubble identities through short disappearances and noisy motion.
- Motion smoothing significantly reduced frame-to-frame jitter.
- Identity switches occurred only during exact overlaps, highlighting the need for more advanced assignment strategies (e.g. Hungarian matching).

Overall, the results showed that **temporal tracking substantially improves bubble segmentation stability**, making the pipeline more scalable for long experiments.

---

## Tech Stack

- Python
- OpenCV
- NumPy / SciPy
- Kalman filtering for state estimation
- Custom video processing utilities

---

## Lessons Learned

- Temporal modeling is critical for stabilizing noisy computer vision pipelines.
- Synthetic data is extremely useful for debugging tracking logic before deploying on real-world footage.
- Simple Kalman models go a long way, but data association becomes the bottleneck in dense or overlapping scenarios.

---

## Future Work

- Improve track assignment using gated Hungarian matching.
- Tune Kalman parameters using statistics from real experimental data.
- Separate fin and bubble masks more robustly.
- Implement a dynamically moving waterline model.

---

## Notes

- Requires **Python < 3.14**
- A virtual environment is recommended:
  ```bash
  pip install -r requirements.txt
