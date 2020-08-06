docker run -d \
  -it \
  --name sturfee \
  --mount type=bind,source=/Users/yuqli/ogl_sandbox,target=/app \
  yuqli:1.0
