# CRF-RNN for Semantic Image Segmentation

This fork adds things that I need for my research.

- Batch detect will convert lots of image and also time each conversion. Format of file name is like prefix0001.jpg
- Batch detect only output human as white, any other 20 labels will be black. You can edit this in the python script.

## How to use

First put all your images you want to convert in the same folder. Name it sequentially with four zeroes like human0000.jpg and so on. Make sure each dimension is not exceeding 500 as the network has been designed and trained with dimension 500x500 as per the CRF as RNN paper.

Finally the syntax is :

`python batchConvert.py yourFolderName yourFileNamePrefix startingIndex finalIndex`

Becareful if your sequence starts from 1 and not 0. That is the point of startingIndex.

For more information about CRF-RNN please vist the project website http://crfasrnn.torr.vision.
