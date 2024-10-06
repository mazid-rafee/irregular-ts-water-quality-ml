import torch

def train_model(model, optimizer, criterion, train_loader, scaler, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_original_loss = 0.0 
        
        for X_batch, timestamp_batch, y_batch in train_loader:
            sequence_input = X_batch[:, :, :] 
            timestamp_input = timestamp_batch 
            optimizer.zero_grad()
            outputs = model(sequence_input, timestamp_input)
            outputs_rescaled = scaler.inverse_transform(outputs.detach().cpu().numpy().reshape(-1, 1))
            y_batch_rescaled = scaler.inverse_transform(y_batch.cpu().numpy().reshape(-1, 1))
            original_loss = criterion(
                torch.tensor(outputs_rescaled, dtype=torch.float32),
                torch.tensor(y_batch_rescaled, dtype=torch.float32)
            )

            running_original_loss += original_loss.item()
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Original Loss: {running_original_loss/len(train_loader):.4f}')
