# Event-camera datasets

Use **only for the generalisation section** ("also evaluated on external sequences"), never
for training. The task needs many clean repetitions of one path + a scripted break, with a
clean trajectory curve as ground truth — no public set is shaped like that. All give
bounding boxes or motion masks, mostly on different sensors, with non-repeating motion.

| Dataset | What | Sensor / res | GT | Note |
|---|---|---|---|---|
| **EV-IMO / EV-IMO2** | indoor, up to 3 fast-moving objects | DAVIS, low-res | motion masks + Vicon 6-DoF | closest to "multiple objects, real trajectory GT"; camera moves; motion-segmentation focus |
| **EventVOT** | single-object tracking benchmark | 1280×720 | bounding box | resolution closest to the DVXplorer; best external check |
| **FE108 / FELT** | single-object tracking, 108 / long sequences | DAVIS346 (346×260) | bbox @ 20–40 Hz | lower-res secondary check |
| **VisEvent** | RGB + event tracking | mixed | bbox | frame+event fusion benchmark |
| **COESOT** | category-wide event single-object tracking | — | bbox | large, category-labelled |
| **MTevent** | 6D pose + moving-object detection | — | pose + boxes | multi-task |
| **Event ping-pong** (arXiv 2506.07860) | egocentric ball trajectory | event + Aria | 3-D ball trajectory | closest task; public code; ballistic only |

If you want more scale with the right sensor model, **simulate** (`02-methods/simulation-with-v2e.md`)
rather than borrow a foreign dataset.
