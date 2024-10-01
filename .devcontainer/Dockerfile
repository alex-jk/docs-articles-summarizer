# Use an official Python image from the Docker Hub
FROM python:3.9

# Set the working directory in the container
WORKDIR /workspace

# Copy requirements file into the container
COPY ../requirements.txt .

# Install any dependencies in the requirements.txt file
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Make sure the installed packages are available globally
ENV PATH="/root/.local/bin:${PATH}"
