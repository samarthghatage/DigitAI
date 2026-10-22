def test_split_sizes(monkeypatch):
    import numpy as np
    import src.data as d
    monkeypatch.setattr(d.tf.keras.datasets.mnist,"load_data",lambda:((np.zeros((60000,28,28),dtype=np.uint8),np.zeros(60000,dtype=np.uint8)),(np.zeros((10000,28,28),dtype=np.uint8),np.zeros(10000,dtype=np.uint8))))
    (x,y),(vx,vy),(tx,ty)=d.load_data(); assert x.shape==(55000,28,28,1) and vx.shape[0]==5000 and tx.shape[0]==10000
