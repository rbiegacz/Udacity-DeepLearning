# Create the input data
data = np.random.rand(16, 100).astype(np.float32)

# Create labels for the data as integers [0, 9].
label = (np.random.rand(16) * 10).astype(np.int32)

workspace.FeedBlob("data", data)
workspace.FeedBlob("label", label)

# Create model using a model helper
m = cnn.CNNModelHelper(name="my first net")
fc_1 = m.FC("data", "fc1", dim_in=100, dim_out=10)
pred = m.Sigmoid(fc_1, "pred")
[softmax, loss] = m.SoftmaxWithLoss([pred, "label"], ["softmax", "loss"])