# Evaluation on the Cloud Dataset

## Class-wise Result of our KTDA
| Class         | IoU   | Acc   | Dice  | Fscore | Precision | Recall |
|--------------|-------|-------|-------|--------|-----------|--------|
| Clear        | 81.68 | 87.59 | 89.92 | 89.92  | 92.38     | 87.59  |
| Cloud Shadow | 0.0   | 0.0   | 0.01  | 0.01   | 66.67     | 0.0    |
| Thin Cloud   | 44.61 | 65.74 | 61.7  | 61.7   | 58.12     | 65.74  |
| Cloud        | 79.68 | 90.83 | 88.69 | 88.69  | 86.65     | 90.83  |

## Comparision Result of our KTDA and Existing Methods

| Method            | mIoU | OA | F1 |
|-------------------|----------------|--------------|--------------|
| MCDNet | 33.85         | 69.75        | 42.76        |
| SCNN   | 32.38         | 71.22        | 52.41        |
| CDNetv1 | 34.58         | 68.16        | 45.80        |
| KappaMask | 42.12         | 76.63        | 68.47        |
| UNetMobv2 | 47.76         | 82.00        | 56.91        |
| CDNetv2 | 43.63         | 78.56        | 70.33        |
| HRCloudNet | 43.51         | 77.04        | 71.36        |
| KTDA (Ours)             | **51.49**      | **83.55**     | **60.08**     |
