# Spam Detection ML Model Using Logistic Regression

This project implements a machine learning model to classify messages as either **spam** or **ham** (not spam) using Logistic Regression. It processes a labeled dataset of messages, extracts text features using TF-IDF, and evaluates model performance with various metrics and visualizations. The program also includes a user interface for real-time predictions.

## Features
- Preprocesses text data and extracts features using **TF-IDF**.
- Trains a **Logistic Regression** model for spam classification.
- Evaluates performance with accuracy, confusion matrix, and classification report.
- Includes visualizations for confidence levels and predicted vs. actual results.
- Supports user input for real-time spam predictions.

## Dataset
The project uses a dataset of labeled messages in **CSV format**. The dataset should have two columns:
- `Label`: Indicates whether the message is "ham" or "spam".
- `Message`: The actual message content.

An example dataset is included in the `dataset/` folder as `spamhamdata.csv`.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/spam-detection-logistic-regression.git
    cd spam-detection-logistic-regression
    ```

2. Install required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

    The key dependencies include:
    - `pandas`
    - `scikit-learn`
    - `matplotlib`
    - `seaborn`
    - `numpy`

3. Ensure the dataset is placed in the `dataset/` folder. The default filename is `spamhamdata.csv`.

4. Run the program:
    ```bash
    python spam_detector.py
    ```

## How It Works

1. **Data Preprocessing**:
    - Labels are encoded as `0` (ham) and `1` (spam).
    - Messages are transformed into numerical features using **TF-IDF Vectorizer**.

2. **Model Training**:
    - The Logistic Regression model is trained on 80% of the dataset and tested on the remaining 20%.

3. **Evaluation**:
    - Calculates **accuracy**, **confusion matrix**, and **classification report**.
    - Plots:
        - Confusion matrix.
        - Distribution of spam confidence levels.
        - Predicted vs. actual results.

4. **Real-Time Predictions**:
    - Allows users to input a message and predicts if it's spam or ham with a confidence score.

## Results
After training and testing, the model provides:
- **Accuracy**: Typically around XX% (dependent on the dataset).
- **Confusion Matrix**: Visualizes the number of true/false positives and negatives.
- **Classification Report**: Displays precision, recall, and F1-score.

## Visualizations
The project includes the following visualizations:
1. **Confusion Matrix**:
   ![Confusion Matrix](confusion_matrix_example.png)

2. **Spam Confidence Levels**:
   ![Spam Confidence](spam_confidence_example.png)

3. **Predicted vs. Actual Results**:
   ![Predicted vs Actual](predicted_vs_actual_example.png)

*(Add actual screenshots from your plots for better representation.)*

## Usage
- Modify `spamhamdata.csv` with your own labeled dataset, keeping the same format.
- Run the program to train and test the model.
- Use the real-time input feature to classify custom messages.

## Contributions
Contributions are welcome! Feel free to fork the repository, make improvements, and submit pull requests. For major changes, please open an issue to discuss the proposed updates.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For queries or suggestions, please contact [your-email@example.com].

---

Happy coding! 😊
