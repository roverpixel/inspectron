# Use an official PyTorch runtime as a parent image
# This base image supports CUDA if available, but falls back to CPU.
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 7001

# Run app.py when the container launches
CMD ["python", "app.py"]
