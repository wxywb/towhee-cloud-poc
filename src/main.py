from concurrent.futures import ThreadPoolExecutor
import os
import json
import shutil
import numpy as np
import towhee
import threading
import logging
import queue
import uuid
import base64
from flask import Flask, request
from io import BytesIO

logging.basicConfig(level=logging.DEBUG)

class InferenceService():
    def construct_towhee_pipline(self): 
        emb_function = (
          towhee.dummy_input['path']()
            .image_decode['path', 'img']()
            .image_embedding.timm['img', 'vec'](model_name='resnet101')
            .select['vec']()
            .as_function()
            )
        return emb_function
    
    def __init__(self):
        self.pipeline = self.construct_towhee_pipline()
        self.q_data = queue.Queue()
        self.q_embs = queue.Queue()
        self.mutex = threading.Semaphore(0)
        self.processed_num = 0
        self.processed_num_td= 0
        self.return_embs = []

    def __call__(self, data):
        if isinstance(data, list):
            self.processed_num_td = len(data)
            self.processed_num = 0
        else:
            self.processed_num_td = 1
            self.processed_num = 0
        self.mutex.acquire()
        return_embs = self.return_embs
        return_embs.sort(key= lambda x: x[0])
        self.return_embs = []
        return return_embs

    def serve_inference(self):
        while True:
            obj = self.q_data.get().result()
            pidx, tmp_path, path = obj 
            emb = self.pipeline(tmp_path)
            self.return_embs.append((pidx, tmp_path, emb.vec, path))
            if len(self.return_embs) == self.processed_num_td:
                self.mutex.release()
            os.remove(tmp_path)


class Service(Flask):
    def __init__(self, pool_capacity=10, *args, **kwargs):
        self.iservice = InferenceService()
        self.executor = ThreadPoolExecutor(max_workers=4)
        super(Service, self).__init__(*args, **kwargs)    
        @self.route('/predict', methods=['POST'])
        def predict():
            data = request.get_data()
            data = json.loads(data)
            len_data = len(data['path'])
            ordered_paths = [(i, data['path'][i]) for i in range(len(data['path']))]
            print('inference {} images.'.format(len_data))
            for opath in ordered_paths:
                fut = self.executor.submit(self.download, opath) 
                fut.add_done_callback(self.download_callback)
            return_embs = self.iservice(ordered_paths)
            ret = self.generate_response(return_embs)
            json_ret = json.dumps(ret)
            return json_ret

    def download(self, path):
        print('downloading {}'.format(path))
        pidx, path_name = path
        ext_name = os.path.splitext(path_name)[-1]
        prefix = str(uuid.uuid4())
        shutil.copyfile(path_name, './tmp/'+prefix+ext_name )
        tmp_path = './tmp/{}{}'.format(prefix, ext_name)
        return (pidx,tmp_path,path_name)

    def download_callback(self, path):
        print('put', path)
        self.iservice.q_data.put(path)

    def generate_response(self, embs):
        ret = []
        for emb in embs:
            shape = emb[2].shape
            #bytesio = BytesIO()
            #np.savetxt(bytesio, emb[2])
            #content = bytesio.getvalue()
            ##b64_code = base64.b64encode(content)
            ##ser_b64_code = b64_code.decode("utf-8")
            #base64_str = str(base64.b64encode(content), 'utf-8')
            ret.append({'vec': emb[2].tolist(), 'shape': shape, 'path': emb[-1]})
        return ret

if __name__ == '__main__':
    service = Service(import_name=__name__)
    tinf = threading.Thread(target=service.iservice.serve_inference, daemon=False, args=())
    tinf.start()
    service.run(debug=True)
    service.logger.setLevel(logging.DEBUG)
    tinf.join()
