# development: prediction and path memory

Median over clips, offset removed. Prediction error in px at each horizon; path = shape of the remembered cycle; period = memory's period over the true one.

| memory | input | pred 25 ms | pred 50 ms | pred 100 ms | pred 200 ms | path px | period |
|---|---|---|---|---|---|---|---|
| snn_phasemap | centroid | 10.9 | 11.7 | 13.5 | 19.9 | 12.9 | 1.00 |
| snn_phasemap | snn | 16.7 | 16.5 | 17.7 | 22.7 | 24.3 | 1.00 |
| phasemap | centroid | 11.4 | 13.0 | 14.5 | 19.0 | 13.1 | 1.00 |
| phasemap | snn | 16.4 | 16.7 | 18.5 | 20.7 | 15.1 | 1.00 |
| kalman | centroid | 10.7 | 11.9 | 13.7 | 16.2 | 14.1 | 1.01 |
| kalman | snn | 18.7 | 20.7 | 25.1 | 23.1 | 17.6 | 1.00 |
| harmonic | centroid | 21.4 | 22.1 | 23.4 | 24.5 | 10.0 | 1.01 |
| harmonic | snn | 20.6 | 21.5 | 22.5 | 22.1 | 13.1 | 1.00 |

# development: deviation detection

Ratcheting alarm, k = 25, chosen on the development clips. AUC and latency over the break clips; false alarms over every clip's quiet part.

| memory | input | break clips | AUC | latency s | false alarms/min |
|---|---|---|---|---|---|
| snn_phasemap | centroid | 2 | 0.98 | 0.23 | 0.00 |
| snn_phasemap | snn | 2 | 0.91 | 0.42 | 0.00 |
| phasemap | centroid | 2 | 1.00 | 0.15 | 0.00 |
| phasemap | snn | 2 | 0.98 | 0.37 | 0.00 |
| kalman | centroid | 2 | 0.86 | 0.42 | 39.93 |
| kalman | snn | 2 | 0.80 | 0.09 | 30.66 |
| harmonic | centroid | 2 | 0.72 | 0.14 | 0.00 |
| harmonic | snn | 2 | 0.72 | 0.28 | 0.00 |
