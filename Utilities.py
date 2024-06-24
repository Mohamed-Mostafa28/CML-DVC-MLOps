import numpy as np
import matplotlib.pyplot as plt
import pickle
import yaml






def load_params(params_path):
    """
    Load parameters from a YAML file.
    
    Parameters:
    - params_path: Path to the YAML file.
    
    Returns:
    - params: Dictionary containing the parameters.
    """
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    return params










def save_image(data, labels, model_name, model_params, file_path):
    """
    Save a portrait-oriented plot of the confusion matrix with model name and parameters.

    Parameters:
    - data: The data to plot (e.g., a confusion matrix).
    - labels: List of labels for the plot.
    - model_name: Name of the model.
    - model_params: Dictionary containing model parameters.
    - file_path: File path where the image will be saved.
    """
    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 10))  # Adjust figsize for portrait orientation
    im = ax.imshow(data, interpolation='nearest', cmap=plt.cm.Greens)  # Change colormap here (e.g., plt.cm.Blues, plt.cm.Oranges)

    # Title and colorbar
    ax.set_title(f'Confusion Matrix for {model_name}', fontsize=16, pad=20)  # Customize title with model name
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)  # Adjust colorbar font size

    # Adding labels
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=12)

    # Adding numbers inside the squares
    thresh = data.max() / 2.
    for i, j in np.ndindex(data.shape):
        ax.text(j, i, format(data[i, j], 'd'),
                ha="center", va="center",
                color="white" if data[i, j] > thresh else "black", fontsize=12)

    # Model parameters
    params_str = '\n'.join(f"{key}: {value}" for key, value in model_params.items())
    ax.text(0, -0.3, params_str, transform=ax.transAxes,
            va='top', ha='left', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Adjusting layout and saving the image
    ax.set_xlabel('Predicted label', fontsize=14, labelpad=20)  # Customize xlabel as needed
    ax.set_ylabel('True label', fontsize=14, labelpad=20)  # Customize ylabel as needed
    fig.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust layout with title and params space
    print("55555555555555555555555555555555555555555555555")
    fig.savefig(file_path, dpi=300)  # Save the plot as an image with higher resolution
    print("55555555555555555555555555555555555555555555555")
    plt.close(fig)  # Close the plot to free up memory (important in loops or scripts)

# Example usage:
# confusion_matrix = np.array([[10, 2],
#                              [3, 15]])
# class_labels = ['Class 0', 'Class 1']
# model_name = "Decision Tree Classifier"
# model_params = {
#     "max_depth": 5,
#     "min_samples_split": 2,
#     "criterion": "gini"
# }
# save_image(confusion_matrix, class_labels, model_name, model_params, 'confusion_matrix.png')





def save_model_info(model_name, accuracy, model_params, classification_report, file_path):
    """
    Save model information, accuracy, and classification report to a text file.

    Parameters:
    - model_name: Name or identifier of the model.
    - accuracy: Accuracy score of the model.
    - model_params: Dictionary containing model parameters.
    - classification_report: Textual representation of the classification report.
    - file_path: File path where the text file will be saved.
    """
    with open(file_path, 'w') as f:
        f.write(f"Model Name: {model_name}\n\n")
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write("Model Parameters:\n")
        for key, value in model_params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report)
        f.write("="*55+"\n")

# # Example usage:
# model_name = "Decision Tree Classifier"
# accuracy = 0.85
# model_params = {
#     "max_depth": 5,
#     "min_samples_split": 2,
#     "criterion": "gini"
# }
# classification_report = """
#               precision    recall  f1-score   support

#      Class 0       0.88      0.92      0.90       100
#      Class 1       0.78      0.70      0.74        50

#     accuracy                           0.85       150
#    macro avg       0.83      0.81      0.82       150
# weighted avg       0.85      0.85      0.85       150
# """
# save_model_info(model_name, accuracy, model_params, classification_report, 'model_info.txt')






def dump_model(model, file_path):
    """
    Save a trained model to a .pkl file using pickle.

    Parameters:
    - model: Trained machine learning model to be saved.
    - file_path: File path where the .pkl file will be saved.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
