#curl  -X POST -d '{"path": ["1", "2"]}' http://127.0.0.1:5000/predict 
#curl -X POST  http://127.0.0.1:5000/predict 
#curl -X POST  http://127.0.0.1:5000/predict
curl -X POST -d '{"path": ["/Users/zilliz/src/pic1.png", "/Users/zilliz/src/pic2.png", "/Users/zilliz/src/pic3.png"]}' http://127.0.0.1:5000/predict
