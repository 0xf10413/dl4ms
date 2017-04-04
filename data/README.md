Overview
--------

This is a collection of existing databases and internal captures made at 
Edinburgh University retargetted to a skeleton with common structure 
and joint lengths.

As we do not own all the rights to all the individual parts of this data please 
respect the individual usage terms and licenses of the existing databases if 
you wish to make use of them. These existing databases are as follows:

cmu:   http://mocap.cs.cmu.edu/
hdm05: http://resources.mpi-inf.mpg.de/HDM05/
mhad:  http://tele-immersion.citris-uc.org/berkeley_mhad

The cmu, hdm05, and mhad databases are free to use, modify and redistribute, 
but they do ask for citation.

The data recorded at Edinburgh University is also free to use, modify and 
redistribute but we would also ask that you please include the following 
citations in any published work which uses this data:


    @inproceedings{Holden:2015:LMM,
     author = {Holden, Daniel and Saito, Jun and Komura, Taku and Joyce, Thomas},
     title = {Learning Motion Manifolds with Convolutional Autoencoders},
     booktitle = {SIGGRAPH Asia 2015 Technical Briefs},
     year = {2015},
    } 
    
    @inproceedings{Holden:2016:DLF,
     author = {Holden, Daniel and Saito, Jun and Komura, Taku},
     title = {A Deep Learning Framework for Character Motion Synthesis and Editing},
     booktitle = {SIGGRAPH 2016},
     year = {2016},
    }
    
    

Processing
----------


### Stage 1

The databases are processed in two stages. First the databases in the 
`external` folder are retargetted to a unified skeleton structure defined by 
the first capture in the CMU database. This is performed using all of the 
scripts inside the `processed` folder starting with the word `retarget`.

This stage places a number of subfolders in the `processed` directory 
containing all the .bvh files for all of the databases. These retargetting 
scripts have already been run so you will see these folder already exist. The
scripts only need to be re-run if some issues are found in the retargetting and
it needs to be adjusted.

The edinburgh databases are already included in the `external` folder but the 
cmu, hdm05 and mhad databases should be downloaded separately and placed there
if you wish to reprocess them:

cmu:           http://mocap.cs.cmu.edu/
hdm05:         http://resources.mpi-inf.mpg.de/HDM05/
mhad:          http://tele-immersion.citris-uc.org/berkeley_mhad

To aquire the styletransfer database for processing you must contact the 
authors directly. This database is NOT free to use, modify and redistribute so 
please respect it's license and usage terms.

styletransfer: http://humanmotion.ict.ac.cn/papers/2015P1_StyleTransfer/details.htm


### Stage 2

The second stage is performed by the `export.py` script in the `processed` 
folder. This script converts the data into the format used by the paper and 
does many processes such as placing the motion on the floor, making the pose 
local to the forward direction and annotating the foot contacts.

This stage puts a number of `.npz` files inside the `processed` folder. As 
before, these are included and as such this script only needs to be re-run if 
there is some adjustment to the required processing.

Once processed the script `view.py` in the `processed` folder can be used to 
view data in this format back using `matplotlib`.


Databases 
---------

Here is some more information about the individual databases provided:

`cmu` - This is the full cmu database retargetted to a character with uniform 
joint lengths. It contains a huge variety of motions and has been used in lots 
of different research over the years.

`hdm05` - This is the hdm05 database retargetted to the cmu skeleton structure 
and joint lengths. It contains many small clips of individual motions 
originally intended for motion classification.

`mhad` - This is the mhad database retargetted to the cmu skeleton structure 
and joint lengths. This database contains just a few actions repeated many 
times by many different subjects.

`edin_locomotion` - This is a database containing long clips of locomotion 
data including running, walking, jogging, and various sidestepping motions. It 
contains around 20 minutes of raw data and is not segmented into individual 
strides.

`edin_kinect` - This is a database containing a large variety of motions 
captured standing in a small area using the kinect motion capture system. 
Because this was captured with the kinect it contains many errors and 
artefacts so should not be used as training data, but may be useful to 
researchers for other purposes.

`edin_xsens` - This is a database containing exactly the same motions as in 
the `edin_kinect` database, but captured using an xsens inertia based motion 
capture system. Because there is a frame-by-frame coorespondence between the 
motion in this database and `edin_kinect` this database may be of interested to 
researchers trying to improve the output of the kinect.

`edin_misc` - This is a small database of various miscellaneous captures made 
at the university including some different walking styles.

`edin_punching` - This is a database of punching, kicking, and fighting motions 
segmented into many small sections.

`edin_terrain` - This is a database of walking and jumping on platforms of 
different heights.

For any questions or queries please contact `contact@theorangeduck.com`
