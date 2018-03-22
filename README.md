# Spark-scala-files
## IntelliJ Spark/Scala Set up

### New Project:
1) Project SDK => Java SDK
2) Library => Scala 2.11 (2.12 doesn't work with several spark jars 2.11)
3) Java -> Scala

### Go to Project Structure
1) Library 
2) Add Java type
3) Spark folder
4) lib or jars folder

Untick REPL and Interactive modes from worksheet

Settings > Languages & Frameworks > Worksheet > untick all boxes

////////
Script must contain .master
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.master("local[*]").getOrCreate()

////////
GeoSpark

1)Download the jar file from https://github.com/DataSystemsLab/GeoSpark/releases
2)Put the file in the folder you want it.
3)Install from terminal in the chosen folder (2): $ spark-shell --jars GeoSpark_COMPILED.jar
4)Add jars in Project Structure as explained in the beginning of this tutorial
5) Repeat for sql and viz
