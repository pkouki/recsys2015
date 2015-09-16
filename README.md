# recsys2015
Code for the paper "HyPER: A Flexible and Extensible Probabilistic Framework for Hybrid Recommender Systems" Pigi Kouki, Shobeir Fakhrei, James Foulds, Magdalini Eirinaki, Lise Getoor. Recommender Systems Conference (RecSys) 2015

In order to run this code you need first to install the Probabilistic Soft Logic (PSL) software, available here: https://github.com/linqs/psl. 

Please cite this work as

@InProceedings{kouki:recsys15,
  author       = "Kouki, Pigi and Fakhraei, Shobeir and Foulds, James and Eirinaki, Magdalini and Getoor, Lise",
  title        = "HyPER: A Flexible and Extensible Probabilistic Framework for Hybrid Recommender Systems",
  booktitle    = "9th ACM Conference on Recommender Systems (RecSys 2015)",
  year         = "2015",
  publisher    = "ACM"
}



Installation instructions:

The following assumes everything is down in the same directory e.g. recsys. The instructions are for MacOS.

1. Download and install the Probabilistic Soft Logic (PSL) software from here: https://github.com/linqs/psl. 
Useful info: https://github.com/linqs/psl/wiki/Getting-started

2. Make sure you can run the basic examples. E.g. 
java -cp ./target/classes:\`cat classpath.out\` edu.umd.cs.example.BasicExample
For help check here: https://github.com/linqs/psl/wiki/Running-a-program

3. clone this git repository
git clone https://github.com/pkouki/recsys2015

4. Go into the h2 directory and run build.sh to compile h2. We need to use this version of h2 as the original version coming with PSL has a bug and crashes under certain cases.

5. Change the classpath.out file inside your psl-example to use this newly compiled h2. For example change the path from the default: 
    /Users/user/.m2/repository/com/h2database/h2/1.2.126/h2-1.2.126.jar 
    to something like:
    /Users/antoulas/Desktop/recsys/recsys2015/h2/bin/h2-1.2.126.jar 

6. copy the folders from recsys2015/data into psl_example/data

7. copy the source files from recsys2015/src/main/java/edu/ucsc/cs/model to psl-example/src/main/java/edu/umd/cs/example/

8. Compile: mvn compile

9. You can now run the models as follows from within the psl_example directory:
    java -cp ./target/classes:`cat classpath.out` edu.ucsc.cs.model.LastfmPerceptronWeightLearning
    java -cp ./target/classes:`cat classpath.out` edu.ucsc.cs.model.YelpPerceptronWeightLearning

If the program runs out of memory you may want to increase the java VM heap size.



