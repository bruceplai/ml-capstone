FROM python

WORKDIR /usr/app

COPY . .

RUN pip install -r requirements.txt
RUN chmod +x train.sh run.sh
RUN ./train.sh

EXPOSE 9090

CMD ./run.sh