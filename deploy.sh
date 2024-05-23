#!/bin/bash

# Step 1: Pull the latest changes from the repository
echo "Pulling the latest changes from the remote repository..."
git pull

# Step 2: Install the required Python packages
echo "Installing required Python packages..."
pip install -r requirements.txt

# Step 3: Run the training script to update the model
echo "Running the training script..."
python train_and_save_model.py

# Step 4: Start the Flask application
echo "Starting the Flask application..."
FLASK_APP=flask_app.py flask run
