# LSTM Numerical Example

This project demonstrates a **step-by-step numerical LSTM calculation** using plain Python.

The example uses a simple input sequence:

```python
X = [1, 2, 3]
```

and manually computes the LSTM gates, cell state, hidden state, and final predicted output for the next value.

## What this notebook/code does

The code performs the following steps:

1. Defines the **sigmoid** activation function.
2. Initializes the **input sequence**.
3. Sets the LSTM parameters:
   - Forget gate weights
   - Input gate weights
   - Candidate cell state weights
   - Output gate weights
4. Starts with:
   - Initial hidden state `h0 = 0`
   - Initial cell state `C0 = 0`
5. Processes the sequence values one by one.
6. Predicts the next value for `x4 = 4`.
7. Applies a linear output layer:

```python
y_hat = Wy * h4 + by
```

## Parameters used

### Forget gate
```python
Wf = 0.5
Whf = 0.1
bf = 0
```

### Input gate
```python
Wi = 0.6
Whi = 0.2
bi = 0
```

### Candidate cell state
```python
Wc = 0.7
Whc = 0.3
bc = 0
```

### Output gate
```python
Wo = 0.8
Who = 0.4
bo = 0
```

### Output layer
```python
Wy = 4
by = 0
```

## Main equations

For each time step:

### Forget gate
```python
f_t = sigmoid(Wf * x_t + Whf * h_(t-1) + bf)
```

### Input gate
```python
i_t = sigmoid(Wi * x_t + Whi * h_(t-1) + bi)
```

### Candidate cell state
```python
C_tilde_t = tanh(Wc * x_t + Whc * h_(t-1) + bc)
```

### Cell state update
```python
C_t = f_t * C_(t-1) + i_t * C_tilde_t
```

### Output gate
```python
o_t = sigmoid(Wo * x_t + Who * h_(t-1) + bo)
```

### Hidden state update
```python
h_t = o_t * tanh(C_t)
```

## Expected result

After processing the sequence and predicting the next value, the final output is approximately:

```python
Predicted next value = 3.796
```

So the model predicts a value close to:

```python
3.8
```

## Files

- `README.md` → explanation of the LSTM numerical example
- Colab notebook cells → implementation in separate blocks

## Notes

- This implementation is written using only Python and the `math` library.
- It is useful for understanding **how LSTM works numerically step by step**.
- It is not using TensorFlow, PyTorch, or any deep learning framework.

## Author note

This example is suitable for:
- assignments
- Colab demonstrations
- learning LSTM calculations manually
- explaining gate-by-gate computations
