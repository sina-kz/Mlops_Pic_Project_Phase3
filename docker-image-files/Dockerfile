FROM python:3.10
WORKDIR /app
RUN pip install gdown
RUN gdown --id 1ASv40otDT8YdSp71D-8zh3XU3EtoEngR
RUN pip install Pillow
RUN pip install fastapi
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install uvicorn
RUN pip install numpy
RUN pip install python-multipart
RUN pip install transformers==4.29.2
RUN pip install tqdm
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
