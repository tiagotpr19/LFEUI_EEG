########### IMPORTS ###########

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from scipy.stats import norm
import os


########### FUNCTIONS ###########

def generate_confusion_matrix(file_path, actual_col, predicted_col):
    """
    Generates and displays a confusion matrix creating a dataset from an Excel file.
    Maps 'left' to 'No' and 'right' to 'Yes' for consistency.

    Parameters:
        file_path (str): Path to the Excel file.
        actual_col (str): Name of the column containing the actual labels.
        predicted_col (str): Name of the column containing the labels predicted by the participants.

    """
    try:
        # Load the Excel file and inspect columns
        temp_data = pd.read_excel(file_path, nrows=0)
        columns = temp_data.columns

        if actual_col not in columns or predicted_col not in columns:
            # Reload data skipping the first 9 rows (empty rows)
            data = pd.read_excel(file_path, skiprows=9)
            if actual_col not in data.columns or predicted_col not in data.columns:
                print(f"Columns '{actual_col}' and '{predicted_col}' do not exist after skipping rows.")
                return
        else:
            data = pd.read_excel(file_path)

        # Extract actual and predicted labels
        actual = data[actual_col]
        predicted = data[predicted_col]

        # Drop rows with missing values in either column
        clean_data = pd.DataFrame({actual_col: actual, predicted_col: predicted}).dropna()
        actual = clean_data[actual_col]
        predicted = clean_data[predicted_col]

        # Ensure all values are strings
        actual = actual.astype(str)
        predicted = predicted.astype(str)

        # Map 'left' -> 'No' and 'right' -> 'Yes'
        mapping = {"left": "No", "right": "Yes"}
        actual = actual.map(mapping)
        predicted = predicted.map(mapping)

        # Generate confusion matrix with updated labels
        cm = confusion_matrix(actual, predicted, labels=["No", "Yes"])

        # Display confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Absent", "Present"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.xlabel("Response")  # Predicted labels
        plt.ylabel("Sound")  # True labels

        # Save the confusion matrix as a PDF in the same directory as the Excel file
        output_path = os.path.splitext(file_path)[0] + "_confusion_matrix_VP.png"
        plt.savefig(output_path, format="png")
        plt.close()  # Close the plot to free memory

        print(f"Confusion matrix saved as PNG at: {output_path}")

        # Calculate signal detection theory metrics
        TN, FP, FN, TP = cm.ravel()

        hit_rate = TP / (TP + FN)
        correct_rejection_rate = TN / (TN + FP)
        false_alarm_rate = FP / (TN + FP)
        proportion_correct = (hit_rate + correct_rejection_rate) / 2

        print(f"\nhit_rate (True Positive Rate): {hit_rate:.4f}")
        print(f"false_alarm_rate (False Positive Rate): {false_alarm_rate:.4f}")
        print(f"correct_rejection_rate (True Negative Rate): {correct_rejection_rate:.4f}")
        print(f"proportion_correct: {proportion_correct:.4f}")

        # Calculate d' and criterion
        dprime = norm.ppf(hit_rate) - norm.ppf(false_alarm_rate)
        criterion = - 0.5 * (norm.ppf(hit_rate) + norm.ppf(false_alarm_rate))

        print(f"\ndprime: {dprime:.4f}")
        print(f"criterion (bias): {criterion:.4f}")

        if criterion < 0:
            print("This person is Liberal")
        else:
            print("This person is Conservative")

        return dprime, criterion

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def plot_gaussian_curves(dprime, criterion, file_path):
    """
    Plots and saves Gaussian curves for signal detection theory.

    Parameters:
        dprime (float): Sensitivity index (d').
        criterion (float): Criterion value.
        file_path (str): Path to the original Excel file.

    """
    try:
        # Create x-axis values
        x = np.linspace(-4, 4, 1000)

        # Gaussian distributions
        noise = norm.pdf(x, loc=0, scale=1)  # Noise distribution (mean=0)
        signal = norm.pdf(x, loc=dprime, scale=1)  # Signal distribution (mean=d')

        # Plot distributions
        plt.figure(figsize=(10, 6))
        plt.plot(x, noise, label="Noise", color="blue")
        plt.plot(x, signal, label="Signal", color="orange")

        # Add criterion
        plt.axvline(x=criterion, color="red", linestyle="--", label="Criterion")

        # Highlight areas
        plt.fill_between(x, noise, 0, color="blue", alpha=0.2)
        plt.fill_between(x, signal, 0, color="orange", alpha=0.2)

        # Plot dprime as the distance between the means of the two distributions
        plt.annotate('', xy=(dprime, 0.4), xytext=(0, 0.4),
                     arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        plt.text(dprime / 2, 0.42, f'd\' = {dprime:.2f}', color='green', ha='center')

        # Add labels and legend
        plt.title("Signal Detection Theory - Gaussian Distributions")
        plt.xlabel("Decision Axis")
        plt.ylabel("Probability Density")
        plt.legend(loc="upper right")
        plt.grid(True)

        # Expand the grid in the y direction
        plt.ylim(0, 0.5)

        # Save the Gaussian curves as a PNG in the same directory as the Excel file
        output_path = os.path.splitext(file_path)[0] + "_gaussian_curves_VP.png"
        plt.savefig(output_path, format="png")
        plt.close()  # Close the plot to free memory

        print(f"Gaussian curves saved as PNG at: {output_path}")

    except Exception as e:
        print(f"An error occurred while plotting Gaussian curves: {e}")

def compute_average_response_time(file_path, response_time_col):
    """
    Computes the average response time from an Excel file.

    Parameters:
        file_path (str): Path to the Excel file.
        response_time_col (str): Name of the column containing the response times.

    Returns:
        float: The average response time.
    """
    try:
        # Load the Excel file
        data = pd.read_excel(file_path)

        if response_time_col not in data.columns:
            print(f"Column '{response_time_col}' does not exist in the file.")
            return None

        # Extract response times and drop rows with missing values
        response_times = data[response_time_col].dropna()

        # Compute the average response time
        average_response_time = response_times.mean()

        print(f"Average response time: {average_response_time:.4f} seconds")
        return average_response_time

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def evaluate_response_consistency(file_path, audio_code_col, response_col):
    """
    Evaluates the consistency of responses for each audio type in the study.

    Parameters:
        file_path (str): Path to the Excel file.
        audio_code_col (str): Name of the column containing the audio codes.
        response_col (str): Name of the column containing the responses.

    """
    try:
        # Load the Excel file
        data = pd.read_excel(file_path)

        if audio_code_col not in data.columns or response_col not in data.columns:
            print(f"Columns '{audio_code_col}' and/or '{response_col}' do not exist in the file.")
            return

        # Extract audio codes and responses, drop rows with missing values
        clean_data = data[[audio_code_col, response_col]].dropna()

        # Group by audio types (e.g., 'animal', 'vehicle') and count the occurrences of each response
        clean_data[audio_code_col] = clean_data[audio_code_col].str.extract(r'([a-zA-Z]+)')[0]
        response_counts = clean_data.groupby([audio_code_col, response_col]).size().unstack(fill_value=0)

        # Calculate consistency as the proportion of the most frequent response for each audio type
        consistency = response_counts.max(axis=1) / response_counts.sum(axis=1)

        # Move 'speech' and 'voice' columns to the right end
        consistency = consistency.reindex(
            [col for col in consistency.index if col not in ['speech', 'voice']] + ['speech', 'voice']
        )

        # Calculate the average consistency
        average_consistency = consistency.mean()

        # Plot the consistency results with custom colors
        colors = ['skyblue' if audio_type not in ['speech', 'voice'] else 'orange' for audio_type in consistency.index]
        plt.figure(figsize=(12, 6))
        consistency.plot(kind='bar', color=colors)
        plt.title('Response Consistency for Each Audio Type')
        plt.xlabel('Audio Type')
        plt.ylabel('Consistency')
        plt.ylim(0, 1)

        # Save the consistency plot as a PNG in the same directory as the Excel file
        output_path = os.path.splitext(file_path)[0] + "_response_consistency_VP.png"
        plt.savefig(output_path, format="png")
        plt.close()  # Close the plot to free memory

        print(f"Response consistency plot saved as PNG at: {output_path}")
        print(f"Average consistency: {average_consistency:.4f}")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


########### MAIN ###########

result = generate_confusion_matrix('file_path', 'corrAns', 'response_VP.keys')
if result is not None:
    dprime, criterion = result
    plot_gaussian_curves(dprime, criterion, 'file_path')

# Time of response
average_response_time = compute_average_response_time('file_path', 'response_VP.rt')

# Evaluate response consistency
evaluate_response_consistency('file_path', 'smallName', 'response_VP.keys')