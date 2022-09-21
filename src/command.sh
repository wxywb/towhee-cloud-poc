#curl  -X POST -d '{"path": ["1", "2"]}' http://127.0.0.1:5000/predict 
#curl -X POST  http://127.0.0.1:5000/predict 
#curl -X POST  http://127.0.0.1:5000/predict
curl -X POST -d '{"path": ["s3://zilliz-vdc-rm-test/cat.png", "s3://zilliz-vdc-rm-test/lion.png", "s3://zilliz-vdc-rm-test/tiger.png"]}' http://127.0.0.1:5000/predict
