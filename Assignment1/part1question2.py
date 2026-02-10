import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def squared_loss(y, y_pred):
   return .5*(y-y_pred)**2

def d_E_yhat(y, y_pred):
   return y_pred - y

def d_yhat_z(y_pred):
   return y_pred*(1-y_pred)

# initial data
x1 = [0,0,1,1]
x2 = [0,1,0,1]
y = [0,1,1,1]
a1 = []
a2 = []
ah1 = []
ah2 = []
z = []
yhat = []
err = []

# forward pass
for i in range(4):
    a1.append(x1[i] + x2[i])
    a2.append(x1[i] + x2[i])
    ah1.append(sigmoid(a1[i]))
    ah2.append(sigmoid(a2[i]))
    z.append(ah1[i] + ah2[i])
    yhat.append(sigmoid(z[i]))
    err.append(squared_loss(y[i], yhat[i]))


# Back Propagation

# Hidden Layer --> Output
w1_arr = []
w2_arr = []
for i in range(4):
    w1_arr.append(d_E_yhat(y[i], yhat[i]) * d_yhat_z(yhat[i]) * ah1[i])
    w2_arr.append(d_E_yhat(y[i], yhat[i]) * d_yhat_z(yhat[i]) * ah2[i])

w1 = np.mean(w1_arr)
w2 = np.mean(w2_arr)

# Input --> Hidden
w11_arr = []
w12_arr = []
w21_arr = []
w22_arr = []
for i in range(4):
   w11_arr.append(d_E_yhat(y[i], yhat[i]) * d_yhat_z(yhat[i]) * ah1[i]*(1-ah1[i]) * x1[i])
   w12_arr.append(d_E_yhat(y[i], yhat[i]) * d_yhat_z(yhat[i]) * ah1[i]*(1-ah1[i]) * x1[i])
   w21_arr.append(d_E_yhat(y[i], yhat[i]) * d_yhat_z(yhat[i]) * ah1[i]*(1-ah1[i]) * x2[i])
   w22_arr.append(d_E_yhat(y[i], yhat[i]) * d_yhat_z(yhat[i]) * ah1[i]*(1-ah1[i]) * x2[i])

w11 = np.mean(w11_arr)
w12 = np.mean(w12_arr)
w21 = np.mean(w21_arr)
w22 = np.mean(w22_arr)

# Output
print(f"w1: {w1}")
print(f"w2: {w2}")
print(f"w11: {w11}")
print(f"w12: {w12}")
print(f"w21: {w21}")
print(f"w22: {w22}")