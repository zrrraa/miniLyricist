# miniLyricist

After learning Andrej Karpathy's video about GPT, I made a Song Ci GPT. 

![miniLyricist](./assets/miniLyricist.gif)

Original video link: [从头开始用代码构建GPT](https://www.bilibili.com/video/BV1CP41147Cw/?spm_id_from=333.337.search-card.all.click&vd_source=1d0c07486a3bd3b0adb8ac548bf6453e)

The datasets come from *the Full Song Ci*. I removed the author's information and the name of the song. 

# Usage

Install the pytorch environment according to your cuda version. 

```
pip install torch
```

Run train.py to train the miniLyricist. This takes about twenty minutes on A100. 

```
python train.py
```

Or you can just use my pre-trained model to inference. Click [here](https://pan.baidu.com/s/15CjE5iGwqmo1CK9QTzk_NA?pwd=yia3) to download.

```
python inference.py
```
