# Product Price Prediction

This project fine-tunes an open-source model using QLoRA to predict product prices.

## How to Run

1.  **Run the Backend Server:**

    Open a new terminal in the `product-price-prediction` directory and run the following command:

    ```bash
    uvicorn backend.main:app --host 127.0.0.1 --port 8000
    ```

2.  **Run the Frontend Application:**

    Open another new terminal in the `product-price-prediction` directory and run the following command:

    ```bash
    streamlit run frontend.app.py
    ```
