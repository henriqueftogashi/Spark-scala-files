////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
/// Case exercise: recommendation engine for video streaming product
/// Author: Henri Furstenau
/// Tool: Spark 2 (Scala) / Databricks
/// Inputs: video_count.csv, video_features.csv
/// Outputs: 5 tables
/// Body: Step 1. Import data, libraries, start Spark session
/// Body: Step 2. Profiling data and Pre-processing
/// Body: Step 3. Defining indicators to create buckets
/// Body: Step 4. QA buckets and create working data frame (master)
/// Body: Step 5. Create the Recommendation System
/// Body: Step 6. Model Evaluation
/// Results
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Data Details (data not provided - you can easily generate fake data if you need. I like python's faker)
/// 
/// Table 1: user_table
/// Columns: 
/// video_id : unique video id
/// count : total count of views for each video per day  
/// date : on which day that video was watched 


/// Table 2: video_features
/// Columns: 
/// video_id : video id, unique by video and joinable to the video id in the other table 
/// video_length : length of the video in seconds 
/// video_language : language of the video, as selected by the user when she uploaded the video 
/// video_upload_date : when the video was uploaded 
/// video_quality : quality of the video. It can be [ 240p, 360p, 480p, 720p, 1080p] 
///////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////// Step 1. Import data, libraries, start Spark session /////////////////////
// import libraries
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

// Spark sql session
val spark = SparkSession.builder().master("local[*]").getOrCreate() // May not need to import spark in Databricks

// reduce warnings. Show errors
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// For scala implicits
import spark.implicits._

// import data
val vc = spark.read.option("header","true").
  option("inferSchema", "true").
  format("csv").load("video_count.csv").
  withColumn("user_id", lit(1)) // Model ALS requires user_id and since we are creating a "global"recommendation system, all videos will have the same user_id = 1.

val vf = spark.read.option("header","true").
  option("inferSchema", "true").
  format("csv").load("video_features.csv")
////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////// Step 2. Profiling data and Pre-processing ////////////////////////////////
// Profiling data (video_count)
vc.show() // first rows
vc.printSchema() // data type
vc.describe().show() // summary (count: 41,775, max views in one day: 6,070,570)
vc.select(vc.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*).show() // no null values
vc.groupBy("video_id").sum().describe().show() // 2,785 distinct video_id
vc.groupBy("video_id").sum().show() // 63,178,982 max views per video_id
vc.groupBy("date").count().orderBy($"date".desc).show() // 15 days of count data available for every video_id every. 01 to 15 Jan 2015

// Profiling data (video_features)
vf.show() // first rows
vf.printSchema() // data type
vf.describe().show() // summary: 2785 unique video_ids (same number than vc but needs to confirm matching)
vf.select(vf.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*).show() // no null values
vf.groupBy("video_upload_date").count().orderBy($"video_upload_date".desc).show() // 20 Different uploading dates. 05 to 24 Dec 2014 (8 to 41 days between upload and views)

// Left join vc and vf
val vcf = vc.join(vf, Seq("video_id"), "left_outer")

// Profiling data (join vc and vf)
vcf.show() // first rows
vcf.printSchema() // data type
vcf.describe().show() // summary
vcf.select(vcf.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*).show() // no null values. That means 2785 video_ids match data.
display(vcf) // Plot histograms for further profiling (Databricks syntax)
////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////// Step 3. Defining indicators to create buckets /////////////////////////////
// 2 indicators:
// iTrend: "Count start date" divided by "count end date" (average of 3 last days divided average of 3 first days): Indicates if video count is going up or down. 3 day span to avoid being mistaken by over- or under-performance in one day. iTrend above 1 indicates decline in views.
// Average count: Videos need to have significant number of views
// Assumption: Videos that were really popular and spiked up and down (for example count over date curves in the shape of -xˆ2 polynomial) will still follow the bucket rules.

val dtmin = vcf.select(to_date(min("date"))).head.getDate(0)
val dtplus2 = vcf.select(date_add(to_date(min("date")), 2)).head.getDate(0)

val df_start = vcf.filter(to_date($"date").between(lit(dtmin), lit(dtplus2))).groupBy("video_id").mean().select($"video_id", bround($"avg(count)", 0).alias("count_start"))

val dtmax = vcf.select(to_date(max("date"))).head.getDate(0)
val dtminus2 = vcf.select(date_add(to_date(max("date")), -2)).head.getDate(0)

val df_end = vcf.filter(to_date($"date").between(lit(dtminus2), lit(dtmax))).groupBy("video_id").mean().select($"video_id", bround($"avg(count)", 0).alias("count_end"))

val df_agg = vcf.groupBy("video_id").mean().select($"video_id", bround($"avg(count)", 0).alias("count"))

val indicators = df_agg.join(df_end, Seq("video_id"), "left_outer").join(df_start, Seq("video_id"), "left_outer").select($"video_id", $"count", $"count_end", $"count_start", ($"count_start" / $"count_end").alias("iTrend"))
////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////// Step 3.1 //////////////////////////////////////////////////////
// Classify each video into the following buckets:
// 3) "Hot" - means trending up. Assumptions: iTrend views growth is 100% (iTrend < 0.5); iAvgCount is above the median of the global daily count
// 2) "Stable and Popular" - video view counts are flat, but very high. Assumptions: iTrend views growth is positive and less than 100% (0.5 >= iTrend > 1); iAvgCount is above the median of the global daily count; Views are starting to decline (20%) but the video is still popular (1>= iTrend > 1.25)
// 1) "Everything else" - everything else
val median = vcf.stat.approxQuantile("count", Array(0.5), 0).head

val buckets = indicators.withColumn("rating", when($"iTrend" < 0.5 and $"count" > median, 3).when($"iTrend" >= 0.5 and $"iTrend" < 1.25 and $"count" > median, 2).otherwise(1))

// Buckets defined by number 3, 2, 1 as described above
buckets.show()
buckets.groupBy("rating").count().show // Ratings frequency
////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////// Step 4. QA buckets and create working data frame (master) ////////////////////////
// organising features to create the master data frame
val all_features = vcf.join(buckets.select("video_id", "rating"), Seq("video_id"), "left_outer").
  join(vcf.groupBy($"video_id").agg(max("date")), Seq("video_id"), "left_outer")

// QA: Check if ratings correspond to buckets
all_features.filter($"rating" === 3).show
display(vcf.filter($"video_id" === 1483).orderBy($"date")) // Plot line for QA (Databricks syntax). video_id 1483 shows a clear curve for bucket 1: "Hot". 10 in each rating tested

// master data frame
val df = all_features.select($"video_id", $"count", $"date", $"user_id", $"video_length", $"video_language", $"video_upload_date", $"video_quality", $"rating", (datediff($"max(date)", $"video_upload_date")).alias("days_from_upload_to_max_views"))

// profiling
df.show()
////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////// Profiling results //////////////////////////////////////////////////////
// Main characteristics of the "hot videos"?

// Video length: average between 2:30 to 16:30 min (mean 499sec, stddev 363sec)
df.filter($"rating" === 3).describe().show

// Video language: English 39%, Chinese 33%, Other 14%, Spanish 9%, French 5%
df.filter($"rating" === 3).groupBy($"video_language").count().show()

// Video quality: 720p 46%, 1080p 25%, 480p 18%, 360p 11%, 240p 2%
df.filter($"rating" === 3).groupBy($"video_quality").count().show()

// Uploading date (days from upload to day with max views): Rating 1 from 22 to 31 days. Rating 2 from 28 to 37 days. Rating 3 from 22 to 44 days
df.groupBy($"rating", $"days_from_upload_to_max_views").count().orderBy($"rating", $"days_from_upload_to_max_views").show(100)
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////// Step 5. Create the Recommendation System ////////////////////////////////
// Split the data in train and test set
val Array(training, testing) = df.randomSplit(Array(0.7, 0.3))

// Collaborative filtering algorithm ALS, fit, transform
val als = new ALS().setMaxIter(5).setRegParam(0.01).setImplicitPrefs(false).setUserCol("user_id").setItemCol("video_id").setRatingCol("rating")

val mdl= als.fit(training)

mdl.setColdStartStrategy("drop")

val res = mdl.transform(testing)

// Profiling
res.describe().show()
res.show()
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////// Step 6. Model Evaluation ////////////////////////////////////////////
// Model error is very small. Data set is big enough

// Evaluation
val evaluator = new RegressionEvaluator().setMetricName("rmse").setLabelCol("rating").setPredictionCol("prediction")

val rmse = evaluator.evaluate(res)
println(rmse)

// see residuals
val eval = res.select(abs($"prediction" - $"rating")).as("eval")

eval.show()
eval.describe().show()
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////// Results /////////////////////////////////////////////////////
// Prepare dfs to generate best video recommendations
val pred_df = res.groupBy($"video_id").agg(mean($"prediction"))
val count_df = vc.groupBy($"video_id").agg(mean($"count"))

// Generate best video recommendations
vf.join(pred_df, Seq("video_id"), "left_outer").join(count_df, Seq("video_id"), "left_outer").orderBy($"avg(prediction)".desc).na.drop().withColumn("bucket", when(bround($"avg(prediction)", 0) === 3, "Hot").when(bround($"avg(prediction)", 0) === 2, "Stable and popular").otherwise("Everything  else")).show() // na.drop => coldStartStrategy parameter set to “drop” by default: During cross-validation, the data is split between training and evaluation sets. When using simple random splits as in Spark’s CrossValidator or TrainValidationSplit, it is actually very common to encounter users and/or items in the evaluation set that are not in the training set.

