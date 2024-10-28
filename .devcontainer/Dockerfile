# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /workspace

# Copy all files in the project directory to the container’s working directory
COPY . /workspace
# Print the contents of the requirements.txt file to verify it exists in the container
RUN cat /workspace/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for language support
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Make sure the installed packages are available globally
ENV PATH="/root/.local/bin:${PATH}"