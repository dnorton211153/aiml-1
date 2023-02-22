Traffic sign detection and recognition

This is a bit more complicated than speed estimation and it consists of 2 parts. Firstly you’ll have to train a model to recognize where traffic signs are in the given image or series of images when it comes to video. You can use a pretrain network for that and then retrain it if you find one, this is recommended. After you’ve done this you’ll have to classify the traffic signs by their meaning. We’ll leave the amount of classes that you recognize up to you but it will be good if you have at least half the classes that are in the dataset. The recommended dataset can be found here:

https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset

Bonus: Lane detection

Once that will be done you have the option to also implement recognition of car lanes. This is not an easy task so the results don’t need to be amazing, hence this is also a bonus task. Not doing this task will not hurt your grade but implementing it might raise it. You can find the dataset for this at:

https://bair.berkeley.edu/blog/2018/05/30/bdd/

What to hand in

You’ll be required to hand in all your code, saved models and documentation. On top of that you’ll be required to generate a short video of all your working parts visualized. So it should look something like this:
https://www.youtube.com/watch?v=fKXztwtXaGo

----

**********************************************
The German Traffic Sign Recognition Benchmark
**********************************************

This archive contains the training set of the 
"German Traffic Sign Recognition Benchmark".

This training set is supposed be used for the online competition 
as part of the IJCNN 2011 competition. It is a subset of the final
training set that will be published after the online competition is
closed. 


**********************************************
Archive content
**********************************************
This archive contains the following structure:

There is one directory for each of the 43 classes (0000 - 00043).
Each directory contains the corresponding training images and one 
text file with annotations, eg. GT-00000.csv. 


**********************************************
Image format and naming
**********************************************
The images are PPM images (RGB color). Files are numbered in two parts:

   XXXXX_YYYYY.ppm

The first part, XXXXX, represents the track number. All images of one class 
with identical track numbers originate from one single physical traffic sign.
The second part, YYYYY, is a running number within the track. The temporal order
of the images is preserved.


**********************************************
Annotation format
**********************************************

The annotations are stored in CSV format (field separator
is ";" (semicolon) ). The annotations contain meta information 
about the image and the class id.


In detail, the annotations provide the following fields:

Filename        - Image file the following information applies to
Width, Height   - Dimensions of the image
Roi.x1,Roi.y1,
Roi.x2,Roi.y2   - Location of the sign within the image
		  (Images contain a border around the actual sign
                  of 10 percent of the sign size, at least 5 pixel)
ClassId         - The class of the traffic sign


**********************************************
Further information
**********************************************
For more information on the competition procedures and to obtain the test set, 
please visit the competition website at

	http://benchmark.ini.rub.de

If you have any questions, do not hesitate to contact us 
    
	tsr-benchmark@ini.rub.de