import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(log_file_path):
    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    # Extract data
    epochs = []
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    test_loss = None
    test_accuracy = None

    for line in lines:
        if 'Epoch:' in line:
            epoch = int(line.split('Epoch:')[1].split('/')[0].strip())
            epochs.append(epoch)

        if 'Phase: train' in line:
            train_loss = float(line.split('Loss:')[1].split('Acc:')[0].strip())
            train_accuracy = float(line.split('Acc:')[1].strip())
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

        if 'Phase: validation' in line:
            val_loss = float(line.split('Loss:')[1].split('Acc:')[0].strip())
            val_accuracy = float(line.split('Acc:')[1].strip())
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

        if 'Best test loss' in line:
            test_loss = float(line.split('Best test loss:')[1].split('|')[0].strip())
            test_accuracy = float(line.split('Best test accuracy:')[1].strip())

    return epochs, train_losses, train_accuracies, val_losses, val_accuracies, test_loss, test_accuracy

# Example usage:
classical_qnn_log_path = 'classical_qnn_log4bit.txt'
classical_epochs, classical_train_losses, classical_train_accuracies, classical_val_losses, classical_val_accuracies, classical_test_loss, classical_test_accuracy = parse_log_file(classical_qnn_log_path)

data = np.array([list(set(classical_epochs)), classical_train_losses, classical_train_accuracies, classical_val_losses, classical_val_accuracies]).T
header = "N\tTraining_Loss\tTraining_Accuracy\tValidation_Loss\tValidation_accuracy\n"
# np.savetxt('classical.dat', data, delimiter='\t', header=header, comments='')
file_path = "classical.dat"

# Write data to file
with open(file_path, "w") as file:
    # Write headers
    file.write(header)
    
    # Write data
    for row in data:
        row_data = "\t".join(map(str, row))
        file.write(row_data + "\n")

hybrid_qnn_log_path = 'hybrid_qnn_log4bit.txt'
hybrid_epochs, hybrid_train_losses, hybrid_train_accuracies, hybrid_val_losses, hybrid_val_accuracies, hybrid_test_loss, hybrid_test_accuracy = parse_log_file(hybrid_qnn_log_path)

data = np.array([list(set(hybrid_epochs)), hybrid_train_losses, hybrid_train_accuracies, hybrid_val_losses, hybrid_val_accuracies]).T
header = "N\tTraining_Loss\tTraining_Accuracy\tValidation_Loss\tValidation_accuracy\n"
np.savetxt('hybrid.dat', data, delimiter='\t', header=header, comments='')

# Ensure the same number of epochs for both classical and hybrid QNN models
common_epochs = list(set(classical_epochs) & set(hybrid_epochs))
# num_epochs = min(20, min(len(classical_test_losses), len(hybrid_test_losses)))
num_epochs = 20

import matplotlib.pyplot as plt

# Assuming you have already parsed the log files and obtained the following arrays:
# common_epochs, classical_train_losses, hybrid_train_losses
# common_epochs, classical_train_accuracies, hybrid_train_accuracies
# common_epochs, classical_val_losses, hybrid_val_losses
# common_epochs, classical_val_accuracies, hybrid_val_accuracies

# Plotting the losses and accuracies for the first 20 epochs
plt.figure(figsize=(18, 12))

# Train Losses
plt.subplot(2, 2, 1)
plt.plot(common_epochs[:num_epochs], classical_train_losses[:num_epochs], label='Classical QNN Train Loss', marker='o')
plt.plot(common_epochs[:num_epochs], hybrid_train_losses[:num_epochs], label='Hybrid QNN Train Loss', marker='o')
plt.title('Train Losses Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Train Accuracies
plt.subplot(2, 2, 2)
plt.plot(common_epochs[:num_epochs], classical_train_accuracies[:num_epochs], label='Classical QNN Train Accuracy', marker='o')
plt.plot(common_epochs[:num_epochs], hybrid_train_accuracies[:num_epochs], label='Hybrid QNN Train Accuracy', marker='o')
plt.title('Train Accuracies Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Validation Losses
plt.subplot(2, 2, 3)
plt.plot(common_epochs[:num_epochs], classical_val_losses[:num_epochs], label='Classical QNN Validation Loss', marker='o')
plt.plot(common_epochs[:num_epochs], hybrid_val_losses[:num_epochs], label='Hybrid QNN Validation Loss', marker='o')
plt.title('Validation Losses Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Validation Accuracies
plt.subplot(2, 2, 4)
plt.plot(common_epochs[:num_epochs], classical_val_accuracies[:num_epochs], label='Classical QNN Validation Accuracy', marker='o')
plt.plot(common_epochs[:num_epochs], hybrid_val_accuracies[:num_epochs], label='Hybrid QNN Validation Accuracy', marker='o')
plt.title('Validation Accuracies Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


