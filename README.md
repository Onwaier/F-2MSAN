# Occlusion expression recognition using facial and multi self attention networks

## The framework of FMSAN
![the framework of FMSAN](https://raw.githubusercontent.com/onwaiers/Picture/master/img/20210404155832.png)

## The framework of F2MSAN
![the framework of F2MSAN](https://raw.githubusercontent.com/onwaiers/Picture/master/img/20210404160025.png)

## Visualization of the effect of occlusion on recognition results.
![occlu effect](https://raw.githubusercontent.com/onwaiers/Picture/master/img/20210406230308.GIF)

**The top is the classification result of the baseline (Resnet18), and the bottom is F-2MSAN**

## Notes on the code
1. The code of FAN, MSAN and crop will be seen in **model/f2msan.py**, you can find more details about model.
2. The implementation of the loss function will be found in **model/loss.py**
3. You can see how to get the mask of a face image in [https://github.com/Onwaier/SegfaceAndAlignByDlib](https://github.com/Onwaier/SegfaceAndAlignByDlib)