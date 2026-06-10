FROM python:3.14
LABEL authors="SON"

WORKDIR /src
# copy file vao trong docker container

COPY naivebayes_model.pkl /src/naivebayes_model.pkl
COPY requirements.txt /src/requirements.txt
COPY /src ./src/
RUN pip install -r requirements.txt


CMD ["python","src/predict.py"]