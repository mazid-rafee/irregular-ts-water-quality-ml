import torch

def train_model(model, optimizer, criterion, train_loader, scaler, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_original_loss = 0.0  # To store the loss on original values
        
        for X_batch, timestamp_batch, y_batch in train_loader:
            # X_batch contains the full sequence (100 data points with analytes, timestamps, and time differences)
            sequence_input = X_batch[:, :, :]  # All data (main analyte, associated analytes, timestamps, etc.)
            
            # timestamp_batch contains both the 101st timestamps and time differences (stacked together)
            timestamp_input = timestamp_batch  # This is already the combined 101st timestamp and time difference

            optimizer.zero_grad()

            # Forward pass: pass both the sequence and the combined timestamp + time difference to the model
            outputs = model(sequence_input, timestamp_input)

            # --- Calculate the loss on original (unscaled) values ---
            outputs_rescaled = scaler.inverse_transform(outputs.detach().cpu().numpy().reshape(-1, 1))
            y_batch_rescaled = scaler.inverse_transform(y_batch.cpu().numpy().reshape(-1, 1))
            
            # Calculate the loss on original scale
            original_loss = criterion(
                torch.tensor(outputs_rescaled, dtype=torch.float32),
                torch.tensor(y_batch_rescaled, dtype=torch.float32)
            )

            running_original_loss += original_loss.item()

            # Backpropagation still using the scaled loss
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

        # Log the original value loss
        print(f'Epoch [{epoch+1}/{num_epochs}], Original Loss: {running_original_loss/len(train_loader):.4f}')
