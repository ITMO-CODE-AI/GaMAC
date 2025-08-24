# DevOps configs and scripts of GaMAC

* comparison - folder with Dockerfile for Jupyter server over GaMAC and with Jupyter notebooks with different tests of GaMAC in it

* Dockerfile for build GaMAC docker image and run test from test.py in it

* download_minio.py - Python script for download objects from our MinIO

* minio.enc and public.enc - ciphered HTTPS certificate for CI/CD pipeline

* test.py - Python script for run different tests in docker container based on GaMAC docker image

## At the root of GaMAC reposotiry there is a folder .github with CI/CD pipelines based on GitHub Actions. It is recommended to run them on self-hosted runners ##