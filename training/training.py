def train(model, optimizer, loss_fn, train_loader, epochs=20, device="cpu"):
    for epoch in range(epochs):
        training_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()

            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)
        print("Epoch: ", epoch, "training_loss: ", training_loss)
