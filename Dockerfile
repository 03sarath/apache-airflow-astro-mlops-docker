FROM quay.io/astronomer/astro-runtime:12.1.0
# Add these lines to your Astro Dockerfile
RUN pip install scikit-learn joblib

