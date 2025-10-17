import streamlit as st
import requests

st.title("Product Price Prediction")

# Add backend status check
def check_backend_status():
    try:
        response = requests.get("http://127.0.0.1:8000/docs", timeout=5)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Backend connection error: {str(e)}")
        return False

# Display backend status
if check_backend_status():
    st.success("✅ Backend is running")
else:
    st.error("❌ Backend is not running")

# Training section
st.header("Train the Model")
if st.button("Start Training"):
    with st.spinner("Training started... This may take a while."):
        try:
            # Make sure the URL is exactly right
            response = requests.post("http://127.0.0.1:8000/train", timeout=30)
            st.write(f"Status Code: {response.status_code}")  # Debug info
            st.write(f"Response: {response.text}")  # Debug info
            
            if response.status_code == 200:
                st.success("Training started successfully in the background!")
                st.json(response.json())  # Show the actual response
            else:
                st.error(f"Error starting training: {response.text}")
        except requests.exceptions.ConnectionError as e:
            st.error("Could not connect to the backend. Please ensure the backend is running.")
            st.error(f"Connection Error Details: {str(e)}")
        except requests.exceptions.Timeout:
            st.error("Request timed out. The backend might be busy.")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

# Prediction section
st.header("Predict Product Price")
product_name = st.text_input("Product Name")
product_description = st.text_area("Product Description")

if st.button("Predict Price"):
    if product_name and product_description:
        with st.spinner("Predicting..."):
            try:
                payload = {"name": product_name, "description": product_description}
                st.write(f"Sending payload: {payload}")  # Debug info
                
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json=payload,
                    timeout=30
                )
                
                st.write(f"Status Code: {response.status_code}")  # Debug info
                # st.write(f"Response: {response.text}")  # Debug info
                
                if response.status_code == 200:
                    result = response.json()
                    if "predicted_price" in result:
                        st.success(f"The predicted price for {result['product']} is: ${result['predicted_price']}")
                    else:
                        st.error(f"Error from backend: {result.get('error', 'Unknown error')}")
                else:
                    st.error(f"Error from backend: {response.text}")
            except requests.exceptions.ConnectionError as e:
                st.error("Could not connect to the backend. Please ensure the backend is running.")
                st.error(f"Connection Error Details: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
    else:
        st.warning("Please enter both product name and description.")