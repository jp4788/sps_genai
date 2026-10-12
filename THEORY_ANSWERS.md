
# Assignment 2 - Part 3: CNN Theory

## Question 1

**Question:** Given a 32×32×3 input image and 8 filters of size 5×5, stride 1, and no padding, what is the output size?

Formula:

Output size = floor((Input - Kernel + 2 × Padding) / Stride) + 1

Calculation:

(32 - 5 + 0) / 1 + 1 = 28

There are 8 filters, so the output has 8 channels.

**Answer: 28 × 28 × 8**

## Question 2

**Question:** How does the output size change if padding is changed to "same"?

With "same" padding and stride 1, the spatial size stays the same as the input.

Output height = 32

Output width = 32

Number of channels = 8

**Answer: 32 × 32 × 8**

## Question 3

**Question:** A 3×3 filter with stride 2 and no padding is applied to a 64×64 input. What is the output spatial size?

Formula:

Output size = floor((Input - Kernel + 2 × Padding) / Stride) + 1

Calculation:

floor((64 - 3 + 0) / 2) + 1

= floor(30.5) + 1

= 31

**Answer: 31 × 31**

## Question 4

**Question:** A 2×2 max-pooling layer with stride 2 is applied to a 16×16 feature map. What is the output size?

Calculation:

(16 - 2) / 2 + 1 = 8

Max pooling reduces the height and width by half.

**Answer: 8 × 8**

## Question 5

**Question:** A 128×128 image passes through two convolutional layers. Both use a 3×3 kernel, stride 1, and "same" padding. What is the output shape?

With "same" padding and stride 1, each convolution keeps the spatial size unchanged.

After the first convolution: 128 × 128

After the second convolution: 128 × 128

**Answer: 128 × 128 (spatial dimensions)**

## Question 6

**Question:** What happens if we remove model.train() before the training loop?

**Answer:** Removing model.train() does not always stop training, but it may cause incorrect training behavior if the model is in evaluation mode.
