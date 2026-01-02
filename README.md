
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
