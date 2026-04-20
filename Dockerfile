FROM public.ecr.aws/lambda/python:3.11

RUN yum install -y gcc10 gcc10-c++ make && \
    ln -sf /usr/bin/gcc10 /usr/bin/gcc && \
    ln -sf /usr/bin/g++10 /usr/bin/g++ && \
    yum clean all

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/        ${LAMBDA_TASK_ROOT}/app/
COPY agent/      ${LAMBDA_TASK_ROOT}/agent/
COPY pipelines/  ${LAMBDA_TASK_ROOT}/pipelines/
COPY monitoring/ ${LAMBDA_TASK_ROOT}/monitoring/
COPY evals/      ${LAMBDA_TASK_ROOT}/evals/

RUN mkdir -p ${LAMBDA_TASK_ROOT}/prompts
RUN mkdir -p /tmp/logs
RUN mkdir -p /tmp/data

CMD ["app.main.handler"]
