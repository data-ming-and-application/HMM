This is the implementation of the paper **Corporate Relative Valuation using Heterogeneous Multi-Modal Graph Neural Network**.

Basic usage
```bash
python3 main.py --data_root ./sample_data --device cuda:0
```

*Notice: The sample_data is a very small dataset for debug to make sure the training, validation and testing is runnable. This dataset is not a subset of our corporate relative valuation dataset, and the performance does not matter. To evaluate HMM model on other dataset, please refer to the following Custom Dataset section*.

For details of all the arguments and help information, 
```bash
python3 main.py -h
```

Requirements: pytorch scikit-learn

HMM.py: the HMM model.

cv_dataset.py: preprocess and build graph for the **C**orporate Relative **V**aluation dataset.

metrics.py: help functions for several metrics.

train_hmm.py: pytorch training code for HMM.

utils.py: help functions for calculating margin loss.

main.py: help and command line interface, detailed description of parameters.

## Custom dataset

The data set used in this project will be announced after the article is published with the permission of the data owner.

We provide a detailed format here in case of usage on other datasets.

### c_c.txt

(invest company, invested company, invest ratio)
```
('c1', 'c296', 1.0)
('c1', 'c302', 0.51)
('c1', 'c306', 1.0)
('c1', 'c300', 0.224)
('c1', 'c303', 1.0)
```

### c_p.txt

(company, people, position id of this people in this company)
```
('c1', 'p1', [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
('c1', 'p4', [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
('c1', 'p2', [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
```

### p_info.txt

people_id, people attribute
```
p7,0.221102,-0.8070350000000001,0.991509,0.807416,-0.005166,-0.487572,0.501344,-1.416632,0.628414,0.567874,0.360421,0.212601,0.36958,-0.47801499999999997,0.368847,-0.8551709999999999,-0.24893099999999999,0.567555,0.35022,-0.10656500000000001,-0.11441400000000002,-0.23879499999999998,-0.649752,0.722106,0.6276189999999999,-0.632771,-0.675759,-0.091075,1.2500879999999999,-0.808951,0.09175499999999999,0.000155,-0.646505,0.282987,-0.407227,-0.213216,-0.17617,0.181603,-1.038985,0.7805489999999999,0.092196,1.197251,-0.173746,-0.261974,-1.022464,-0.353187,0.538159,0.036936000000000004,-0.021117,0.826942
p8,0.32873600000000003,-1.1261709999999998,0.818144,0.40993100000000005,0.20371199999999998,-0.571954,0.441104,-1.072633,0.814242,0.429817,0.416253,0.29029499999999997,0.187653,-0.63593,0.547273,-0.9295540000000001,-0.28747,0.5485939999999999,0.383912,-0.158203,-0.085238,-0.080536,-0.75967,0.518534,0.532878,-0.452219,-0.629105,0.254504,1.04034,-0.967375,0.048351,0.16630799999999998,-0.8869879999999999,0.328864,-0.070145,0.061027,-0.077671,0.421911,-1.0601209999999999,0.887494,-0.0015630000000000002,1.148855,-0.492716,-0.569242,-0.47010799999999997,-0.383949,0.765278,0.281625,0.11590999999999999,1.1091010000000001
```

### c_info.txt

company_id, company attribute
```
c8,0.03676470588235294,0.010101010101010102,0.5,0.07607607607607608,0.07407407407407407,0.0,0.10101010101010101,0.10101010101010101,0.14444444444444443,0.0,0.001001001001001001,0.05917159763313609,0.0,0.020202020202020204,0.020202020202020204,0.0,0.0,0.0,0.0,0.020202020202020204,0.020202020202020204,0.0,0.0,0.04040404040404041,0.04040404040404041,0.0,0.0,0.0,0.023023023023023025,0.020202020202020204,0.020202020202020204,0.0,0.0,0.5345345345345346,1.0,0.0,0.11983471074380166,0.29365079365079366,1.0,0.208,0.0,0.0,0.0,0.0,0.029411764705882353,0.8,0.01818181818181818,0.0,0.0,0.0,0.25,0.0,0.0,0.0,0.0,0.0,0.04040404040404041,0.04040404040404041,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.3333333333333333,0.2631578947368421,0.08,0.7,0.08708708708708708,0.0,0.005005005005005005,0.002002002002002002,0.011011011011011011,0.041928721174004195,0.657275,-0.862284,0.081813,-0.180645,0.016222,0.550884,0.255101,-0.552221,1.262831,0.915518,-0.199347,0.252942,-0.9568450000000001,0.480204,0.33599,-1.0000959999999999,-0.626958,0.263023,-0.655669,0.240542,-0.325152,-0.767717,-0.466624,0.8902389999999999,-0.37245300000000003,0.682639,0.001047,0.178925,-0.355698,0.738164,0.0033130000000000004,-1.058926,0.448704,0.812701,0.140457,0.07117899999999999,0.339116,0.724097,-0.639592,0.527254,0.31490999999999997,0.392828,0.40092300000000003,0.370839,0.137568,0.396802,1.189057,-0.051448,-0.005986,0.226675
c9,0.13970588235294118,0.08080808080808081,0.25,0.1001001001001001,0.8148148148148148,0.2702702702702703,0.030303030303030304,0.030303030303030304,0.37777777777777777,0.0,0.16616616616616617,0.1893491124260355,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.09090909090909091,0.09090909090909091,0.2222222222222222,0.035035035035035036,0.44344344344344344,1.0,0.0,0.004132231404958678,0.023809523809523808,1.0,0.994,0.0,0.0,0.0,0.0,0.058823529411764705,0.2,0.05454545454545454,0.0,0.0,0.16666666666666666,0.0,0.16666666666666666,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0018975332068311196,0.0,0.0,0.0,0.0,0.0,1.0,0.6111111111111112,0.0,0.10526315789473684,0.04,1.0,0.4694694694694695,0.07607607607607608,0.009009009009009009,0.005005005005005005,0.12412412412412413,0.03773584905660377,-0.682394,-0.7438739999999999,-0.080481,0.27123200000000003,0.41206800000000005,-0.49625600000000003,-0.087916,0.121876,1.084676,-0.09635099999999999,0.532459,0.32848299999999997,0.123735,-0.382792,0.292707,-0.973769,-0.510128,1.159096,-0.565704,0.186474,-0.09529800000000001,0.10375799999999999,-0.301349,0.5155420000000001,0.41341400000000006,0.12383599999999999,-0.45242899999999997,-0.5446,-0.8026260000000001,0.137468,-0.5049739999999999,-0.053739999999999996,-0.022221,0.315576,-0.150534,-0.510097,0.187574,0.52137,-0.398167,0.26422199999999996,-0.470272,0.630664,0.61446,1.325348,-1.107867,-0.347516,0.49695600000000006,-0.550531,-0.967125,-0.397032
```

### c_label.txt

company_id, one-hot encoding of value, one-hot encoding of area
```
c11,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0
c12,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0
c13,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0
```
The first line of this file is ignored.

### splits/

This folder contains several splits of dataset with different training ratio.

```
split_10.pkl
split_30.pkl
split_50.pkl
split_70.pkl
```

Each split is a pickle file contains a dictionary {"train_idx": train_idx, "test_idx": test_idx}

The splits can be modified easily in train_hmm.py