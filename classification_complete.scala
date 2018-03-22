//https://spark.apache.org/docs/2.2.0/ml-tuning.html

// libraries
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}

import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().master("local[*]").getOrCreate()

// reduce warnings
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

val data = spark.read.option("header", "true").option("mode", "DROPMALFORMED").option("inferSchema", "true").format("csv").load("titanic.csv")
data.columns

// Exploring data
data.groupBy("Embarked").count().show()
data.printSchema()
data.show()

// df
import spark.implicits._
val labeldf = data.select(data("Survived").as("label"), $"Pclass", $"Sex", $"Age", $"SibSp", $"Parch", $"Fare", $"Embarked").na.drop

val Array(training, testing) = labeldf.randomSplit(Array(0.7, 0.3))

// TRANSFORMATIONS
// One hot encoding
val sexIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("sexIndexed")
val sexEncoded = new OneHotEncoder().setInputCol("sexIndexed").setOutputCol("sexEncoded")

val embarkedIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("embarkedIndexed")
val embarkedEncoded = new OneHotEncoder().setInputCol("embarkedIndexed").setOutputCol("embarkedEncoded")
// checking encoding
val ind = embarkedIndexer.fit(labeldf).transform(labeldf)
val enc = embarkedEncoded.transform(ind)

//Assembler
val assembler = new VectorAssembler().setInputCols(Array("Pclass", "sexEncoded", "Age", "SibSp", "Parch", "Fare", "embarkedEncoded")).setOutputCol("features")
//checking assembler
val ind2 = sexIndexer.fit(enc).transform(enc)
val enc2 = sexEncoded.transform(ind2)
val ass = assembler.transform(enc2)
ass.show()

//Feature scaling
val scaler = new StandardScaler().setInputCol("features").setOutputCol("featureScaled").setWithStd(true).setWithMean(false)
// checking fs
scaler.fit(ass).transform(ass).select("featureScaled").show()

//CLASSIFICATION
val lr = new LogisticRegression()

//PIPELINE
val pipeline = new Pipeline().setStages(Array(sexIndexer, embarkedIndexer, sexEncoded, embarkedEncoded, assembler, scaler, lr))

// PARAMETER TUNING
val pgb = new ParamGridBuilder().addGrid(lr.maxIter, Array(1, 5, 10, 20, 30)).addGrid(lr.regParam,Array(0.01,0.1, 1, 10 ,100)).addGrid(lr.elasticNetParam, Array(0.01, 0.5, 0.1, 1)).build()

// CROSS VALIDATION
val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(new BinaryClassificationEvaluator).setEstimatorParamMaps(pgb).setNumFolds(3)

//RESULTS
val model = cv.fit(training)
val results = model.transform(testing)

//VALIDATION
// get best parameters
println(model.getEstimatorParamMaps.zip(model.avgMetrics).maxBy(_._2)._1)

// Confusion matrix converting to rdd
import org.apache.spark.mllib.evaluation.MulticlassMetrics

val resrdd = results.select($"prediction", $"label").as[(Double, Double)].rdd
val cm = new MulticlassMetrics(resrdd)
println(cm.confusionMatrix)