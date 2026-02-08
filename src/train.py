import torch

def train_model(model, model_name, optimizer, criterion, train_loader, scaler, num_epochs=10, device=torch.device("cpu")):
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        running_original_loss = 0.0 
        
        for X_batch, timestamp_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            timestamp_batch = timestamp_batch.to(device)
            y_batch = y_batch.to(device)

            sequence_input = X_batch[:, :, :] 
            timestamp_input = timestamp_batch 
            optimizer.zero_grad()

            if model_name.lower() == "neural ode":
                outputs = model(sequence_input)
            else:
                outputs = model(sequence_input, timestamp_input)

            outputs_rescaled = scaler.inverse_transform(outputs.detach().cpu().numpy().reshape(-1, 1))
            y_batch_rescaled = scaler.inverse_transform(y_batch.detach().cpu().numpy().reshape(-1, 1))

            outputs_rescaled_tensor = torch.tensor(outputs_rescaled, dtype=torch.float32).to(device)
            y_batch_rescaled_tensor = torch.tensor(y_batch_rescaled, dtype=torch.float32).to(device)

            original_loss = criterion(outputs_rescaled_tensor, y_batch_rescaled_tensor)
            running_original_loss += original_loss.item()

            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Original Loss: {running_original_loss/len(train_loader):.4f}')
