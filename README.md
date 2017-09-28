This is a python program to convert a Tensorflow bidirectional LSTM to the equivalent CoreML model. It's still in progress.  In particular, I haven't got the extraction of the weights from Tensorflow completely figured out.

The intent is to use this for recognizing unsegmented handwriting from an iPad + Apple Pencil, but the code should be easily adaptable to other applications.

Thanks to the authors of the below repositories, from which
most of this code was taken:

 1.  https://github.com/igormq/ctc_tensorflow_example
 2.  https://github.com/tbornt/phoneme_ctc
 3.  https://github.com/mitochrome/complex-gestures-demo
