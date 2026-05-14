# Pytorch
## Loss Fuction
1. MSELoss
Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input x and target y
2. L1Loss
Creates a criterion that measures the mean absolute error (MAE) between each element in the input x and target y
3. Huber Loss
Creates a criterion that uses a squared term if the absolute element-wise error falls below delta and a delta-scaled L1 term otherwise. This loss combines advantages of both L1Loss and MSELoss
4. Cross Entropy Loss
This criterion computes the cross entropy loss between input logits and target.
