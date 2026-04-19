FROM public.ecr.aws/lambda/python:3.11

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