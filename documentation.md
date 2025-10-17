# Project Documentation

This document provides an overview of the main modules in the product price prediction system.

## Project Structure

```
D:/LLM Awareness/Phase -1 projects/product-price-prediction/
├───.git/
├───backend/
│   └───main.py
├───data/
│   └───product_data.csv
├───frontend/
│   └───app.py
├───llm/
│   └───qlora_fine_tuning.py
├───.gitignore
├───documentation.md
├───predict.log
├───predict.py
├───Product Price Prediction System - Architecture.docx
├───Product Price Prediction System - Solutions.docx
└───README.md
```

## Modules

### `qlora_fine_tuning.py`

This script is responsible for fine-tuning the language model. It performs the following steps:

1.  **Load a pre-trained model:** It loads a pre-trained language model from the Hugging Face Hub.
2.  **Prepare a dataset:** It loads a dataset of product information from a CSV file and prepares it for training.
3.  **Fine-tune the model:** It uses QLoRA to fine-tune the model on the prepared dataset.
4.  **Save the model:** It saves the fine-tuned model to a directory.

### `backend/main.py`

This script contains the backend API for the product price prediction system. It uses FastAPI to create a web server that exposes the following endpoints:

*   **`/`**: A welcome message to indicate that the API is running.
*   **`/train`**: This endpoint triggers the fine-tuning process in the background.
*   **`/predict`**: This endpoint takes a product name and description as input and returns a predicted price. It loads the fine-tuned model and uses it to generate a price prediction.

### `frontend/app.py`

This script contains the frontend application for the product price prediction system. It uses Streamlit to create a simple web interface that allows users to:

*   Enter a product name and description.
*   Click a button to get a predicted price.
*   View the predicted price.